"""
HPCGreenEnv — RL Environment for Green-Aware HPC Scheduling.

Supports heterogeneous resource clusters:
  - GPU nodes (physical GPU allocation)
  - CPU nodes (CPU core allocation)
  - MIG nodes (A100/H100 MIG slice allocation)
  - Mixed CPU+GPU hybrid jobs

State space (121 × 12 = 1452 features):
  [MAX_QUEUE_SIZE=64 × JOB_FEATURES=12]  — waiting queue
  [RUN_WIN=32       × JOB_FEATURES=12]   — running jobs
  [GREEN_WIN=24     × JOB_FEATURES=12]   — renewable energy forecast
  [CLUSTER_WIN=1    × JOB_FEATURES=12]   — global cluster state

Job feature vector (12 dim):
  0  wait_time / MAX_WAIT_SEC
  1  n_gpu / MAX_GPUS
  2  req_runtime / MAX_RUNTIME_SEC
  3  power / MAX_POWER_W
  4  power_per_unit / (MAX_POWER_W / MAX_GPUS)
  5  uses_brown (0/1)
  6  brown_ratio
  7  schedulable (ALL resources available: 1/0)
  8  n_cpu / MAX_CPUS
  9  resource_type  (0=GPU 0.25=MIG 0.5=CPU 0.75=HYBRID)
  10 n_mig / MAX_MIGS
  11 cpu_schedulable (CPU cores available: 1/0)

Reward (sparse, end-of-episode):
  r = ReUtil − η × AvgBSLD

Action:
  MaskablePPO : job_idx ∈ [0, MAX_QUEUE_SIZE)
  GAS-MARL    : (job_idx, delay_action),  delay ∈ [0, ACTION2_NUM)
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
    CPUJob, MIGJob, HybridJob,
    ResourceType, JobStatus, MODEL_PROFILES,
)
from ..cluster.cluster import Cluster, CLUSTER_CONFIGS, NodeSpec
from ..cluster.hardware import MIGProfile, MIG_PROFILE_SPECS, NodeType


# ─── Constants ────────────────────────────────────────────────────────────────

MAX_QUEUE_SIZE  = 64       # waiting queue window
RUN_WIN         = 32       # running jobs window
GREEN_WIN       = 24       # forecast slots (hours)
CLUSTER_WIN     = 1        # global cluster state rows
TOTAL_ROWS      = MAX_QUEUE_SIZE + RUN_WIN + GREEN_WIN + CLUSTER_WIN  # = 121
JOB_FEATURES    = 12       # features per row (unified across all sections)

# Backward-compatible aliases used by networks.py
RUN_FEATURES    = JOB_FEATURES
GREEN_FEATURES  = JOB_FEATURES

ACTION2_NUM     = 13       # GAS-MARL delay choices
DELAY_MAX_JOB_NUM = 5      # max running jobs to wait for

DELAY_TIMES = [300, 600, 1200, 1800, 2400, 3000, 3600]   # seconds

# Normalisation constants
MAX_WAIT_SEC    = 86400.0
MAX_RUNTIME_SEC = 86400.0
MAX_GPUS        = 64.0
MAX_CPUS        = 256.0
MAX_MIGS        = 64.0
MAX_POWER_W     = 200_000.0
BSLD_THRESHOLD  = 10.0     # τ in AvgBSLD

# Resource type encoding (continuous)
_RT_ENC = {
    ResourceType.GPU:     0.00,
    ResourceType.MIG:     0.25,
    ResourceType.CPU:     0.50,
    ResourceType.CPU_GPU: 0.75,
}


@dataclass
class EnvConfig:
    workload_config:   Optional[WorkloadConfig]  = None
    cluster_config:    str                        = "hpc_realistic"
    renewable_config:  Optional[RenewableConfig] = None
    sim_duration_sec:  float                      = 86_400.0
    eta:               float                      = 0.002
    brown_threshold_j: float                      = 50_000.0
    seq_len:           int                        = 256
    seed:              Optional[int]              = None


class _RunningJob:
    """Lightweight tracker for a job currently running in the environment."""
    __slots__ = [
        "job_id", "start_time", "finish_time",
        "num_gpus", "num_cpus", "num_mig",
        "power_w", "req_runtime", "wait_time",
        "resource_type", "bsld",
    ]

    def __init__(self, job_id, start_time, finish_time,
                 num_gpus, num_cpus, num_mig,
                 power_w, req_runtime, wait_time,
                 resource_type=ResourceType.GPU):
        self.job_id        = job_id
        self.start_time    = start_time
        self.finish_time   = finish_time
        self.num_gpus      = num_gpus
        self.num_cpus      = num_cpus
        self.num_mig       = num_mig
        self.power_w       = power_w
        self.req_runtime   = req_runtime
        self.wait_time     = wait_time
        self.resource_type = resource_type
        self.bsld          = 0.0


class HPCGreenEnv:
    """
    Step-based RL environment for green-aware HPC scheduling.

    Supports heterogeneous resource pools:
      - Physical GPUs (any type from cluster config)
      - Schedulable CPU cores (CPU-only and mixed nodes)
      - MIG slices (A100/H100 MIG-partitioned nodes)

    Episode flow:
        1. Jobs arrive from workload trace (GPU/CPU/MIG/Hybrid)
        2. When a decision is needed, step() is called
        3. Agent selects job (+ optional delay for GAS-MARL)
        4. Resources are allocated; simulation advances
        5. Episode ends after seq_len decisions or workload exhausted

    Compatible with: MaskablePPOScheduler, GASMARLScheduler
    """

    def __init__(self, config: Optional[EnvConfig] = None):
        self.cfg = config or EnvConfig()
        self.rng = random.Random(self.cfg.seed or 42)

        # Build cluster to get resource totals
        self._cluster_cfg = CLUSTER_CONFIGS[self.cfg.cluster_config]
        _specs = self._cluster_cfg._normalise_nodes()

        self._total_gpus = sum(s.total_gpus() for s in _specs)
        self._total_cpus = sum(s.total_cpu_cores() for s in _specs)
        self._total_mig  = sum(
            self._count_mig_slices(s) for s in _specs
        )

        # Renewable energy module (GPU-centric, with CPU correction)
        self._re = RenewableEnergyModule(
            config=self.cfg.renewable_config,
            total_gpus=max(self._total_gpus, 1),
            sim_duration=self.cfg.sim_duration_sec,
        )

        # Episode state (initialised in reset())
        self._current_time:     float = 0.0
        self._avail_gpus:       int   = 0
        self._avail_cpus:       int   = 0
        self._avail_mig:        int   = 0
        self._pending:          list  = []
        self._running:          list  = []
        self._completed:        list  = []
        self._jobs_all:         list  = []
        self._pending_arrivals: list  = []
        self._delayed:          list  = []
        self._power_trace:      list  = []
        self._decisions_made:   int   = 0

    @staticmethod
    def _count_mig_slices(spec: NodeSpec) -> int:
        if spec.mig_profile == MIGProfile.NONE:
            return 0
        info = MIG_PROFILE_SPECS.get(spec.mig_profile, {})
        return info.get("max_per_gpu", 0) * spec.gpus_per_node * spec.num_nodes

    # ── Public API ─────────────────────────────────────────────────────────────

    def seed(self, s: int):
        self.rng = random.Random(s)
        return [s]

    def reset(self) -> np.ndarray:
        wcfg = self.cfg.workload_config or WorkloadConfig(
            duration=self.cfg.sim_duration_sec,
            rng_seed=self.rng.randint(0, 99999),
        )
        self._jobs_all = WorkloadGenerator(wcfg).generate()

        self._current_time   = 0.0
        self._avail_gpus     = self._total_gpus
        self._avail_cpus     = self._total_cpus
        self._avail_mig      = self._total_mig
        self._pending        = []
        self._running        = []
        self._completed      = []
        self._delayed        = []
        self._power_trace    = []
        self._decisions_made = 0
        self._pending_arrivals = sorted(self._jobs_all, key=lambda j: j.submit_time)

        self._advance_arrivals(0.0)
        # If no jobs at t=0, advance to first arrival
        if not self._pending and self._pending_arrivals:
            first_t = self._pending_arrivals[0].submit_time
            self._advance_time(first_t)
        # Ensure at least one schedulable job is ready
        if self._pending and not any(self._is_schedulable(j) for j in self._pending):
            self._advance_until_schedulable()

        return self._get_obs()

    def step(
        self,
        job_action:   int,
        delay_action: int = 0,
    ) -> tuple[np.ndarray, float, bool, float, float, float, int, float]:
        """
        Execute one scheduling decision.

        Args:
            job_action:   index into pending queue [0, MAX_QUEUE_SIZE)
            delay_action: GAS-MARL only.
                          0 = immediate
                          1-5 = wait for N running jobs to finish
                          6-12 = delay by DELAY_TIMES[action-6]

        Returns:
            obs, reward, done, bsld_reward, sjf_r, f1_r, running_num, green_reward
        """
        visible = self._pending[:MAX_QUEUE_SIZE]
        if not visible:
            self._advance_time(self._current_time)
            return self._get_obs(), 0.0, False, 0.0, 0.0, 0.0, len(self._running), 0.0

        if job_action >= len(visible):
            job_action = 0
        job = visible[job_action]

        # ── Delay handling (GAS-MARL) ─────────────────────────────────────────
        release_time = self._current_time
        if delay_action > 0:
            if delay_action <= DELAY_MAX_JOB_NUM:
                n_wait = min(delay_action, len(self._running))
                if n_wait > 0 and self._running:
                    sorted_r = sorted(self._running, key=lambda r: r.finish_time)
                    release_time = min(
                        self._current_time + 3600.0,
                        sorted_r[min(n_wait - 1, len(sorted_r) - 1)].finish_time,
                    )
            else:
                dt_idx = min(delay_action - DELAY_MAX_JOB_NUM - 1, len(DELAY_TIMES) - 1)
                release_time = self._current_time + DELAY_TIMES[dt_idx]

        if release_time > self._current_time:
            self._pending.remove(job)
            self._delayed.append((release_time, job))
            self._advance_time(release_time)
        else:
            # Try to schedule immediately
            if self._is_schedulable(job):
                self._start_job(job, self._current_time)
                self._pending.remove(job)
            else:
                # Resource unavailable — advance to next freeing event
                self._advance_time(self._current_time)

        self._decisions_made += 1

        done = (
            self._decisions_made >= self.cfg.seq_len
            or (not self._pending and not self._pending_arrivals and not self._delayed)
        )

        if not done:
            self._advance_until_schedulable()

        bsld_r = 0.0
        green_r = 0.0
        if done:
            self._finish_all()
            bsld_r = -self._avg_bsld()
            green_r = self._re.compute_utilization(self._power_trace)

        reward = self.cfg.eta * bsld_r + green_r
        return (self._get_obs(), reward, done, bsld_r, 0.0, 0.0,
                len(self._running), green_r)

    # ── Action Masks ───────────────────────────────────────────────────────────

    def action_mask1(self) -> np.ndarray:
        """
        Binary mask over job queue: 1 = job can be scheduled now.
        A job is schedulable only if ALL required resources are available.
        """
        mask = np.zeros(MAX_QUEUE_SIZE, dtype=np.float32)
        for i, job in enumerate(self._pending[:MAX_QUEUE_SIZE]):
            mask[i] = 1.0 if self._is_schedulable(job) else 0.0
        if not mask.any():
            mask[0] = 1.0   # fallback: allow at least one action
        return mask

    def action_mask2(self) -> np.ndarray:
        """
        Delay action mask for GAS-MARL: 1 = INVALID.
        Job-wait delays (1-5) are invalid if fewer running jobs than requested.
        """
        mask = np.zeros(ACTION2_NUM, dtype=np.float32)
        n_running = len(self._running)
        for i in range(1, DELAY_MAX_JOB_NUM + 1):
            if i > n_running:
                mask[i] = 1.0
        return mask

    # ── Observation ────────────────────────────────────────────────────────────

    def _get_obs(self) -> np.ndarray:
        """
        Build observation tensor of shape (121, 12), then flatten.
        Rows: [64 queue | 32 running | 24 green | 1 cluster]
        """
        total_rows = MAX_QUEUE_SIZE + RUN_WIN + GREEN_WIN + CLUSTER_WIN
        obs = np.zeros((total_rows, JOB_FEATURES), dtype=np.float32)

        # ── Queue slots ───────────────────────────────────────────────────────
        visible = self._pending[:MAX_QUEUE_SIZE]
        for i, job in enumerate(visible):
            obs[i] = self._job_features(job)
        # Padding sentinel: all-ones = "empty slot"
        for i in range(len(visible), MAX_QUEUE_SIZE):
            obs[i] = np.ones(JOB_FEATURES, dtype=np.float32)

        # ── Running slots ──────────────────────────────────────────────────────
        sorted_running = sorted(self._running, key=lambda r: r.finish_time)
        for i, rj in enumerate(sorted_running[:RUN_WIN]):
            row = MAX_QUEUE_SIZE + i
            rem = max(0.0, rj.finish_time - self._current_time)
            obs[row, 0] = min(rj.num_gpus / max(MAX_GPUS, 1), 1.0)
            obs[row, 1] = min(rj.power_w / MAX_POWER_W, 1.0)
            obs[row, 2] = min(rj.power_w / max(rj.num_gpus + rj.num_cpus / 4, 1)
                              / (MAX_POWER_W / max(MAX_GPUS, 1)), 1.0)
            obs[row, 3] = min(rem / MAX_RUNTIME_SEC, 1.0)
            obs[row, 4] = min(rj.num_cpus / MAX_CPUS, 1.0)
            obs[row, 5] = _RT_ENC.get(rj.resource_type, 0.0)
            obs[row, 6] = min(rj.num_mig / max(MAX_MIGS, 1), 1.0)
            # rows 7-11 = 0 (padding)

        # ── Green forecast ─────────────────────────────────────────────────────
        forecast = self._re.get_forecast(self._current_time)
        for i, (rem_dur, pow_w) in enumerate(forecast[:GREEN_WIN]):
            row = MAX_QUEUE_SIZE + RUN_WIN + i
            obs[row, 0] = min(rem_dur / 3600.0, 1.0)
            obs[row, 1] = min(pow_w / MAX_POWER_W, 1.0)
            # rows 2-11 = 0

        # ── Cluster state (1 row) ──────────────────────────────────────────────
        row = MAX_QUEUE_SIZE + RUN_WIN + GREEN_WIN
        running_power  = sum(r.power_w for r in self._running)
        re_avail       = self._re.available_power_watts(self._current_time)
        green_ratio    = min(re_avail / max(running_power, 1.0), 1.0)
        tod            = (self._current_time % 86400) / 86400.0  # time-of-day
        obs[row, 0]    = self._avail_gpus / max(self._total_gpus, 1)
        obs[row, 1]    = self._avail_cpus / max(self._total_cpus, 1) if self._total_cpus else 0.0
        obs[row, 2]    = self._avail_mig  / max(self._total_mig, 1)  if self._total_mig  else 0.0
        obs[row, 3]    = min(running_power / MAX_POWER_W, 1.0)
        obs[row, 4]    = tod
        obs[row, 5]    = len(self._pending)   / MAX_QUEUE_SIZE
        obs[row, 6]    = len(self._running)   / RUN_WIN
        obs[row, 7]    = green_ratio
        obs[row, 8]    = min(self._total_cpus / MAX_CPUS, 1.0)
        obs[row, 9]    = min(self._total_mig  / MAX_MIGS, 1.0)
        obs[row, 10]   = min(self._total_gpus / MAX_GPUS, 1.0)
        # row 11 = 0

        return obs.flatten()

    def _job_features(self, job: AnyJob) -> np.ndarray:
        wait     = self._current_time - job.submit_time
        n_gpu, n_cpu, n_mig, rt = self._resources_for_job(job)
        req_rt   = self._estimate_runtime(job)
        power    = self._estimate_job_power(job, n_gpu, n_cpu, n_mig)
        denom    = max(n_gpu + n_mig + n_cpu / 8, 1)
        power_pu = power / denom

        # Brown energy estimate
        re_avail      = self._re.available_power_watts(self._current_time)
        running_power = sum(r.power_w for r in self._running)
        idle          = self._re.idle_power_watts(max(self._total_gpus, 1))
        total_if_start = idle + running_power + power
        uses_brown    = 1.0 if total_if_start > re_avail else 0.0
        brown_ratio   = max(0.0, min(1.0,
            (total_if_start - re_avail) / max(total_if_start, 1.0)
        ))

        schedulable     = 1.0 if self._is_schedulable(job) else 0.0
        cpu_schedulable = 1.0 if (n_cpu <= self._avail_cpus) else 0.0

        return np.array([
            min(wait   / MAX_WAIT_SEC,    1.0),       # 0
            min(n_gpu  / MAX_GPUS,        1.0),       # 1
            min(req_rt / MAX_RUNTIME_SEC, 1.0),       # 2
            min(power  / MAX_POWER_W,     1.0),       # 3
            min(power_pu / (MAX_POWER_W / max(MAX_GPUS, 1)), 1.0),  # 4
            uses_brown,                                # 5
            brown_ratio,                               # 6
            schedulable,                               # 7
            min(n_cpu  / MAX_CPUS,        1.0),       # 8
            _RT_ENC.get(rt, 0.0),                     # 9
            min(n_mig  / MAX_MIGS,        1.0),       # 10
            cpu_schedulable,                           # 11
        ], dtype=np.float32)

    # ── Resource helpers ───────────────────────────────────────────────────────

    def _resources_for_job(
        self, job: AnyJob
    ) -> tuple[int, int, int, ResourceType]:
        """Return (n_gpu, n_cpu, n_mig, resource_type) for a job."""
        rt = getattr(job, "resource_type", ResourceType.GPU)

        if isinstance(job, CPUJob):
            n_cpu = getattr(job, "num_cpus_requested", 4)
            return 0, n_cpu, 0, ResourceType.CPU

        if isinstance(job, MIGJob):
            n_mig = getattr(job, "num_mig_requested", 1)
            return 0, 0, n_mig, ResourceType.MIG

        if isinstance(job, HybridJob):
            n_gpu = max(1, getattr(job, "num_gpus_requested", 1))
            n_cpu = getattr(job, "num_cpus_requested", 4)
            return n_gpu, n_cpu, 0, ResourceType.CPU_GPU

        # Default: GPU job
        n_gpu = max(1, min(getattr(job, "num_gpus_requested", 1), self._total_gpus))
        return n_gpu, 0, 0, ResourceType.GPU

    def _is_schedulable(self, job: AnyJob) -> bool:
        """True if all required resources are currently available."""
        n_gpu, n_cpu, n_mig, rt = self._resources_for_job(job)
        gpu_ok = (n_gpu <= self._avail_gpus)
        cpu_ok = (n_cpu <= self._avail_cpus) if n_cpu > 0 else True
        mig_ok = (n_mig <= self._avail_mig)  if n_mig > 0 else True
        return gpu_ok and cpu_ok and mig_ok

    def _estimate_runtime(self, job: AnyJob) -> float:
        if isinstance(job, CPUJob):
            return job.effective_duration(job.num_cpus_requested)
        if isinstance(job, MIGJob):
            return job.base_duration_sec
        if isinstance(job, (TrainingJob, HybridJob)):
            n_iter = getattr(job, "num_iterations", 10_000)
            n_gpu  = max(1, getattr(job, "num_gpus_requested", 1))
            arch   = getattr(job, "arch", None)
            profile = MODEL_PROFILES.get(arch)
            tp     = (profile.base_throughput_k80 * 2.0) if profile else 5.0
            return n_iter / (tp * n_gpu)
        if isinstance(job, InferenceJob):
            return job.total_queries / max(job.query_rate_per_sec, 1.0)
        if isinstance(job, LLMJob):
            n_iter = job.num_iterations
            n_gpu  = max(1, job.num_gpus_requested)
            return n_iter / max(n_gpu * 0.1, 0.1)
        return 3600.0

    def _estimate_job_power(
        self, job: AnyJob,
        n_gpu: int = 0, n_cpu: int = 0, n_mig: int = 0,
    ) -> float:
        gpu_power = self._re.job_power_watts(n_gpu, utilization=0.8) if n_gpu > 0 else 0.0
        # CPU power: rough estimate ~20W per busy core
        cpu_power = n_cpu * 20.0
        # MIG power: fraction of one GPU
        mig_power = n_mig * 80.0   # ~200W / (400W/GPU × 1/7 × correction)
        return max(gpu_power + cpu_power + mig_power, 1.0)

    # ── Simulation core ────────────────────────────────────────────────────────

    def _start_job(self, job: AnyJob, t: float):
        n_gpu, n_cpu, n_mig, rt = self._resources_for_job(job)
        req_rt = self._estimate_runtime(job)
        power  = self._estimate_job_power(job, n_gpu, n_cpu, n_mig)
        wait_t = max(0.0, t - job.submit_time)

        rj = _RunningJob(
            job_id=job.job_id,
            start_time=t, finish_time=t + req_rt,
            num_gpus=n_gpu, num_cpus=n_cpu, num_mig=n_mig,
            power_w=power, req_runtime=req_rt, wait_time=wait_t,
            resource_type=rt,
        )
        self._running.append(rj)
        self._avail_gpus -= n_gpu
        self._avail_cpus -= n_cpu
        self._avail_mig  -= n_mig
        self._record_power(t)

    def _finish_job(self, rj: _RunningJob, t: float):
        exec_t = max(t - rj.start_time, 1.0)
        denom  = max(BSLD_THRESHOLD, rj.req_runtime)
        rj.bsld = max((rj.wait_time + exec_t) / denom, 1.0)
        self._running.remove(rj)
        self._completed.append(rj)
        self._avail_gpus += rj.num_gpus
        self._avail_cpus += rj.num_cpus
        self._avail_mig  += rj.num_mig
        self._record_power(t)

    def _record_power(self, t: float):
        idle         = self._re.idle_power_watts(max(self._total_gpus, 1))
        running_power = sum(r.power_w for r in self._running)
        self._power_trace.append((t, idle + running_power))

    def _advance_arrivals(self, until: float):
        while self._pending_arrivals and self._pending_arrivals[0].submit_time <= until:
            self._pending.append(self._pending_arrivals.pop(0))

    def _advance_time(self, target: float):
        still_delayed = []
        for (release_t, job) in self._delayed:
            if release_t <= target:
                self._pending.append(job)
            else:
                still_delayed.append((release_t, job))
        self._delayed = still_delayed

        for rj in sorted(
            [rj for rj in self._running if rj.finish_time <= target],
            key=lambda r: r.finish_time,
        ):
            self._finish_job(rj, rj.finish_time)

        self._current_time = target
        self._advance_arrivals(target)

    def _advance_until_schedulable(self):
        for _ in range(300):
            if any(self._is_schedulable(j) for j in self._pending):
                return
            if any(rt <= self._current_time for rt, _ in self._delayed):
                return
            candidates = []
            if self._running:
                candidates.append(min(r.finish_time for r in self._running))
            if self._pending_arrivals:
                candidates.append(self._pending_arrivals[0].submit_time)
            if self._delayed:
                candidates.append(min(rt for rt, _ in self._delayed))
            if not candidates:
                return
            self._advance_time(min(candidates))
            if not self._pending and not self._pending_arrivals and not self._delayed:
                return

    def _finish_all(self):
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
            for job in list(self._pending):
                if self._is_schedulable(job):
                    self._start_job(job, self._current_time)
                    self._pending.remove(job)

    def _avg_bsld(self) -> float:
        if not self._completed:
            return 0.0
        return sum(rj.bsld for rj in self._completed) / len(self._completed)

    # ── Properties ─────────────────────────────────────────────────────────────

    @property
    def obs_shape(self) -> tuple[int, ...]:
        return ((MAX_QUEUE_SIZE + RUN_WIN + GREEN_WIN + CLUSTER_WIN) * JOB_FEATURES,)

    @property
    def obs_shape_2d(self) -> tuple[int, int]:
        return (MAX_QUEUE_SIZE + RUN_WIN + GREEN_WIN + CLUSTER_WIN, JOB_FEATURES)

    @property
    def n_actions(self) -> int:
        return MAX_QUEUE_SIZE

    @property
    def n_delay_actions(self) -> int:
        return ACTION2_NUM

    @property
    def renewable_module(self) -> RenewableEnergyModule:
        return self._re
