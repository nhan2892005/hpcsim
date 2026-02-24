"""
HPCGreenEnv — RL Environment for Green-Aware HPC Scheduling.

State space (per decision step):
  [MAX_QUEUE_SIZE × JOB_FEATURES]   — waiting queue (padded)
  [RUN_WIN × RUN_FEATURES]          — running jobs (padded)
  [GREEN_WIN × GREEN_FEATURES]      — 24-h renewable energy forecast

Reward (sparse, at episode end):
  r = ReUtil - η × AvgBSLD

Action:
  MaskablePPO: job_idx  ∈ [0, MAX_QUEUE_SIZE)
  GAS-MARL:   (job_idx, delay_action)  delay ∈ [0, ACTION2_NUM)

Designed to interface with:
  - hpcsim.rl.maskable_ppo.MaskablePPOScheduler
  - hpcsim.rl.gas_marl.GASMARLScheduler

And to be used in training via:
  - hpcsim.rl.train.train_maskable_ppo / train_gas_marl
"""

from __future__ import annotations
import math
import random
from dataclasses import dataclass, field
from typing import Optional
import numpy as np

from ..energy.renewable import RenewableEnergyModule, RenewableConfig
from ..workload.generator import WorkloadGenerator, WorkloadConfig, TRACE_CONFIGS
from ..workload.job import (
    AnyJob, TrainingJob, InferenceJob, LLMJob, HPOJob,
    JobStatus, MODEL_PROFILES,
)
from ..cluster.cluster import Cluster, CLUSTER_CONFIGS


# ─── Constants ────────────────────────────────────────────────────────────────

MAX_QUEUE_SIZE   = 64      # waiting queue window
RUN_WIN          = 32      # running jobs window
GREEN_WIN        = 24      # forecast slots (hours)

JOB_FEATURES     = 8      # features per waiting job
RUN_FEATURES     = 4      # features per running job
GREEN_FEATURES   = 2      # features per energy slot

ACTION2_NUM      = 13     # GAS-MARL delay choices
DELAY_MAX_JOB_NUM = 5     # max running jobs to wait for

# Delay time candidates (seconds)
DELAY_TIMES = [300, 600, 1200, 1800, 2400, 3000, 3600]

# Normalisation constants
MAX_WAIT_SEC     = 86400.0   # 1 day
MAX_RUNTIME_SEC  = 86400.0
MAX_GPUS         = 64.0
MAX_POWER_W      = 50_000.0
BSLD_THRESHOLD   = 10.0      # τ in AvgBSLD


@dataclass
class EnvConfig:
    workload_config: Optional[WorkloadConfig] = None
    cluster_config:  str = "medium_heterogeneous_gavel"
    renewable_config: Optional[RenewableConfig] = None
    sim_duration_sec: float = 86_400.0   # 24 h per episode
    eta: float = 0.005                    # penalty factor for AvgBSLD
    brown_threshold_j: float = 50_000.0  # σ — green-backfill brown energy cap
    seq_len: int = 256                    # jobs per trajectory
    seed: Optional[int] = None


class _RunningJob:
    """Lightweight tracking of a running job."""
    __slots__ = ["job_id", "start_time", "finish_time", "num_gpus",
                 "power_w", "req_runtime", "wait_time", "bsld"]

    def __init__(self, job_id, start_time, finish_time,
                 num_gpus, power_w, req_runtime, wait_time):
        self.job_id = job_id
        self.start_time = start_time
        self.finish_time = finish_time
        self.num_gpus = num_gpus
        self.power_w = power_w
        self.req_runtime = req_runtime
        self.wait_time = wait_time
        self.bsld = 0.0


class HPCGreenEnv:
    """
    Step-based RL environment for green-aware HPC scheduling.

    Episode flow:
        1. Jobs arrive from workload trace
        2. When a scheduling decision is needed, step() is called
        3. Agent selects job (+ optional delay for GAS-MARL)
        4. Simulation advances to next event
        5. Episode ends after seq_len decisions or all jobs scheduled

    Compatible with: PyTorch-based RL agents (MaskablePPO, GAS-MARL)
    """

    def __init__(self, config: Optional[EnvConfig] = None):
        self.cfg = config or EnvConfig()
        self.rng = random.Random(self.cfg.seed or 42)

        # Load cluster info
        self._cluster_cfg = CLUSTER_CONFIGS[self.cfg.cluster_config]
        self._total_gpus = sum(
            n * g for _, n, g in self._cluster_cfg.nodes
        )

        # Renewable energy
        self._re = RenewableEnergyModule(
            config=self.cfg.renewable_config,
            total_gpus=self._total_gpus,
            sim_duration=self.cfg.sim_duration_sec,
        )

        # Episode state (initialised in reset())
        self._current_time: float = 0.0
        self._avail_gpus: int = 0
        self._pending: list[AnyJob] = []
        self._running: list[_RunningJob] = []
        self._completed: list[_RunningJob] = []
        self._jobs_all: list[AnyJob] = []
        self._pending_arrivals: list[AnyJob] = []  # not yet arrived
        self._delayed: list[tuple[float, AnyJob]] = []  # (release_time, job)
        self._power_trace: list[tuple[float, float]] = []
        self._decisions_made: int = 0

    # ── Public API ─────────────────────────────────────────────────────────────

    def seed(self, s: int):
        self.rng = random.Random(s)
        return [s]

    def reset(self) -> np.ndarray:
        """Reset environment and return initial observation."""
        wcfg = self.cfg.workload_config or WorkloadConfig(
            duration=self.cfg.sim_duration_sec,
            rng_seed=self.rng.randint(0, 99999),
        )
        gen = WorkloadGenerator(wcfg)
        self._jobs_all = gen.generate()

        self._current_time = 0.0
        self._avail_gpus = self._total_gpus
        self._pending = []
        self._running = []
        self._completed = []
        self._delayed = []
        self._power_trace = []
        self._decisions_made = 0

        # All jobs sorted by arrival time
        self._pending_arrivals = sorted(self._jobs_all, key=lambda j: j.submit_time)

        # Advance time until first job arrives
        self._advance_arrivals(0.0)
        if not self._pending:
            self._advance_time(0.0)

        return self._get_obs()

    def step(
        self,
        job_action: int,
        delay_action: int = 0,
    ) -> tuple[np.ndarray, float, bool, float, float, float, int, float]:
        """
        Execute one scheduling decision.

        Args:
            job_action:   index into pending queue [0, MAX_QUEUE_SIZE)
            delay_action: 0=immediate, 1-5=wait N jobs, 6-12=delay by time

        Returns:
            obs, reward, done, bsld_reward, sjf_r, f1_r, running_num, green_reward
        """
        # Map action to actual job in queue
        visible = self._pending[:MAX_QUEUE_SIZE]
        if job_action >= len(visible):
            job_action = 0
        if not visible:
            self._advance_time(self._current_time)
            return self._get_obs(), 0.0, False, 0.0, 0.0, 0.0, len(self._running), 0.0

        job = visible[job_action]

        # ── Handle delay decision ────────────────────────────────────────────
        release_time = self._current_time
        if delay_action > 0:
            if delay_action <= DELAY_MAX_JOB_NUM:
                # Wait until N running jobs complete
                n_wait = min(delay_action, len(self._running))
                if n_wait > 0 and self._running:
                    sorted_running = sorted(self._running, key=lambda r: r.finish_time)
                    capped = min(self._current_time + 3600.0,
                                 sorted_running[min(n_wait - 1, len(sorted_running) - 1)].finish_time)
                    release_time = capped
            else:
                # delay_action 6..12 → index into DELAY_TIMES
                dt_idx = delay_action - (DELAY_MAX_JOB_NUM + 1)
                dt_idx = min(dt_idx, len(DELAY_TIMES) - 1)
                release_time = self._current_time + DELAY_TIMES[dt_idx]

        if release_time > self._current_time:
            # Park job for later
            self._pending.remove(job)
            self._delayed.append((release_time, job))
            # Advance to release time
            self._advance_time(release_time)
        else:
            # Try to schedule immediately
            n_gpus = self._gpu_count_for_job(job)
            if n_gpus <= self._avail_gpus:
                self._start_job(job, n_gpus, self._current_time)
                self._pending.remove(job)
            # else: leave in queue, advance to next event
            else:
                self._advance_time(self._current_time)

        self._decisions_made += 1

        # Check episode end
        done = (
            self._decisions_made >= self.cfg.seq_len
            or (not self._pending and not self._pending_arrivals and not self._delayed)
        )

        # Advance until something is schedulable
        if not done:
            self._advance_until_schedulable()

        # Rewards
        bsld_r = 0.0
        green_r = 0.0
        if done:
            # Drain remaining running jobs
            self._finish_all()
            bsld_r = -self._avg_bsld()
            green_r = self._re.compute_utilization(self._power_trace)

        reward = self.cfg.eta * bsld_r + green_r

        obs = self._get_obs()
        return (obs, reward, done, bsld_r, 0.0, 0.0,
                len(self._running), green_r)

    # ── Masks ─────────────────────────────────────────────────────────────────

    def action_mask1(self) -> np.ndarray:
        """Binary mask over job queue: 1 = can select this job."""
        mask = np.zeros(MAX_QUEUE_SIZE, dtype=np.float32)
        visible = self._pending[:MAX_QUEUE_SIZE]
        for i, job in enumerate(visible):
            n = self._gpu_count_for_job(job)
            # Schedulable if resources available OR agent may choose to delay
            mask[i] = 1.0 if n > 0 else 0.0
        if not mask.any():
            mask[0] = 1.0  # fallback
        return mask

    def action_mask2(self) -> np.ndarray:
        """Binary mask for delay actions: 0 = valid, 1 = invalid (masked out)."""
        mask = np.zeros(ACTION2_NUM, dtype=np.float32)
        n_running = len(self._running)
        # Types 2: delay until N running jobs finish — only valid if running
        for i in range(1, DELAY_MAX_JOB_NUM + 1):
            if i > n_running:
                mask[i] = 1.0   # invalid: can't wait for more jobs than running
        # Type 1 (action 0) is always valid
        # Types 3 (actions 6-12) are always valid
        return mask

    # ── Observation ───────────────────────────────────────────────────────────

    def _get_obs(self) -> np.ndarray:
        total_slots = MAX_QUEUE_SIZE + RUN_WIN + GREEN_WIN
        obs = np.zeros((total_slots, JOB_FEATURES), dtype=np.float32)

        # Waiting queue
        visible = self._pending[:MAX_QUEUE_SIZE]
        for i, job in enumerate(visible):
            obs[i] = self._job_features(job)

        # Pad empty slots with sentinel (all-ones = padding)
        for i in range(len(visible), MAX_QUEUE_SIZE):
            obs[i] = np.ones(JOB_FEATURES, dtype=np.float32)

        # Running jobs
        sorted_running = sorted(self._running, key=lambda r: r.finish_time)
        for i, rj in enumerate(sorted_running[:RUN_WIN]):
            row = MAX_QUEUE_SIZE + i
            rem = max(0.0, rj.finish_time - self._current_time)
            obs[row, 0] = min(rj.num_gpus / MAX_GPUS, 1.0)
            obs[row, 1] = min(rj.power_w / MAX_POWER_W, 1.0)
            obs[row, 2] = min(rj.power_w / max(rj.num_gpus, 1) / (MAX_POWER_W / MAX_GPUS), 1.0)
            obs[row, 3] = min(rem / MAX_RUNTIME_SEC, 1.0)

        # Renewable energy forecast
        forecast = self._re.get_forecast(self._current_time)
        for i, (rem_dur, pow_w) in enumerate(forecast[:GREEN_WIN]):
            row = MAX_QUEUE_SIZE + RUN_WIN + i
            obs[row, 0] = min(rem_dur / 3600.0, 1.0)
            obs[row, 1] = min(pow_w / MAX_POWER_W, 1.0)

        return obs.flatten()

    def _job_features(self, job: AnyJob) -> np.ndarray:
        wait = self._current_time - job.submit_time
        n_gpu = self._gpu_count_for_job(job)
        req_rt = self._estimate_runtime(job)
        power = self._estimate_job_power(job)
        power_per_gpu = power / max(n_gpu, 1)
        schedulable = 1.0 if n_gpu <= self._avail_gpus else 0.0

        # Brown energy estimation
        re_avail = self._re.available_power_watts(self._current_time)
        cluster_idle = self._re.idle_power_watts(self._total_gpus)
        current_running_power = sum(r.power_w for r in self._running)
        cluster_power_if_started = cluster_idle + current_running_power + power
        uses_brown = 1.0 if cluster_power_if_started > re_avail else 0.0
        brown_ratio = max(0.0, min(1.0,
            (cluster_power_if_started - re_avail) / max(cluster_power_if_started, 1.0)
        ))

        return np.array([
            min(wait / MAX_WAIT_SEC, 1.0),
            min(n_gpu / MAX_GPUS, 1.0),
            min(req_rt / MAX_RUNTIME_SEC, 1.0),
            min(power / MAX_POWER_W, 1.0),
            min(power_per_gpu / (MAX_POWER_W / MAX_GPUS), 1.0),
            uses_brown,
            brown_ratio,
            schedulable,
        ], dtype=np.float32)

    # ── Simulation helpers ────────────────────────────────────────────────────

    def _gpu_count_for_job(self, job: AnyJob) -> int:
        n = getattr(job, "num_gpus_requested", 1)
        return max(1, min(n, self._total_gpus))

    def _estimate_runtime(self, job: AnyJob) -> float:
        if isinstance(job, (TrainingJob, LLMJob)):
            n_iter = getattr(job, "num_iterations", 10_000)
            n_gpu = max(1, self._gpu_count_for_job(job))
            arch = getattr(job, "arch", None)
            from ..workload.job import MODEL_PROFILES
            profile = MODEL_PROFILES.get(arch)
            tp = (profile.base_throughput_k80 * 2.0) if profile else 5.0
            return n_iter / (tp * n_gpu)
        elif isinstance(job, InferenceJob):
            return job.total_queries / max(job.query_rate_per_sec, 1.0)
        else:
            return 3600.0

    def _estimate_job_power(self, job: AnyJob) -> float:
        n = self._gpu_count_for_job(job)
        return self._re.job_power_watts(n, utilization=0.8)

    def _start_job(self, job: AnyJob, n_gpus: int, t: float):
        req_rt = self._estimate_runtime(job)
        power = self._estimate_job_power(job)
        wait_t = t - job.submit_time
        rj = _RunningJob(
            job_id=job.job_id,
            start_time=t,
            finish_time=t + req_rt,
            num_gpus=n_gpus,
            power_w=power,
            req_runtime=req_rt,
            wait_time=max(0.0, wait_t),
        )
        self._running.append(rj)
        self._avail_gpus -= n_gpus
        self._record_power(t)

    def _finish_job(self, rj: _RunningJob, t: float):
        exec_t = max(t - rj.start_time, 1.0)
        denom = max(BSLD_THRESHOLD, rj.req_runtime)
        rj.bsld = max((rj.wait_time + exec_t) / denom, 1.0)
        self._running.remove(rj)
        self._completed.append(rj)
        self._avail_gpus += rj.num_gpus
        self._record_power(t)

    def _record_power(self, t: float):
        cluster_idle = self._re.idle_power_watts(self._total_gpus)
        running_power = sum(r.power_w for r in self._running)
        self._power_trace.append((t, cluster_idle + running_power))

    def _advance_arrivals(self, until: float):
        while self._pending_arrivals and self._pending_arrivals[0].submit_time <= until:
            job = self._pending_arrivals.pop(0)
            self._pending.append(job)

    def _advance_time(self, target: float):
        """Advance simulation to target time, processing completions."""
        # Release delayed jobs
        still_delayed = []
        for (release_t, job) in self._delayed:
            if release_t <= target:
                self._pending.append(job)
            else:
                still_delayed.append((release_t, job))
        self._delayed = still_delayed

        # Finish completed running jobs
        to_finish = [rj for rj in self._running if rj.finish_time <= target]
        for rj in sorted(to_finish, key=lambda r: r.finish_time):
            self._finish_job(rj, rj.finish_time)

        self._current_time = target
        self._advance_arrivals(target)

    def _advance_until_schedulable(self):
        """Advance time until at least one pending job can be scheduled."""
        max_iters = 200
        for _ in range(max_iters):
            # Check if anything is schedulable
            schedulable = any(
                self._gpu_count_for_job(j) <= self._avail_gpus
                for j in self._pending
            )
            # Also release any delayed jobs past their release time
            delayed_ready = any(rt <= self._current_time for rt, _ in self._delayed)
            if (schedulable and self._pending) or delayed_ready:
                return

            # Nothing pending; advance to next interesting event
            candidates = []
            if self._running:
                candidates.append(min(r.finish_time for r in self._running))
            if self._pending_arrivals:
                candidates.append(self._pending_arrivals[0].submit_time)
            if self._delayed:
                candidates.append(min(rt for rt, _ in self._delayed))
            if not candidates:
                return
            next_t = min(candidates)
            self._advance_time(next_t)
            if not self._pending and not self._pending_arrivals and not self._delayed:
                return

    def _finish_all(self):
        """Drain all running jobs at their scheduled finish times."""
        while self._running or self._pending_arrivals or self._delayed:
            candidates = []
            if self._running:
                candidates.append(min(r.finish_time for r in self._running))
            if self._pending_arrivals:
                candidates.append(self._pending_arrivals[0].submit_time)
            if self._delayed:
                candidates.append(min(rt for rt, _ in self._delayed))
            if not candidates:
                break
            self._advance_time(min(candidates))

            # Start any schedulable pending jobs
            for job in list(self._pending):
                n = self._gpu_count_for_job(job)
                if n <= self._avail_gpus:
                    self._start_job(job, n, self._current_time)
                    self._pending.remove(job)

    def _avg_bsld(self) -> float:
        if not self._completed:
            return 0.0
        return sum(rj.bsld for rj in self._completed) / len(self._completed)

    @property
    def obs_shape(self) -> tuple[int, ...]:
        return ((MAX_QUEUE_SIZE + RUN_WIN + GREEN_WIN) * JOB_FEATURES,)

    @property
    def obs_shape_2d(self) -> tuple[int, int]:
        return (MAX_QUEUE_SIZE + RUN_WIN + GREEN_WIN, JOB_FEATURES)

    @property
    def n_actions(self) -> int:
        return MAX_QUEUE_SIZE

    @property
    def n_delay_actions(self) -> int:
        return ACTION2_NUM

    @property
    def renewable_module(self) -> RenewableEnergyModule:
        return self._re
