"""
Workload Generator — arrival processes and trace generation.

Supports:
- Poisson arrivals (λ = constant)
- Pareto (heavy-tailed / bursty, Philly trace)
- Diurnal (24-hour pattern, Alibaba trace)

Calibrated to match production cluster traces from the survey.
"""

from __future__ import annotations
from dataclasses import dataclass, field
from enum import Enum
from typing import Optional
import random
import math

from .job import (
    TrainingJob, InferenceJob, LLMJob, HPOJob, AnyJob,
    ModelArch, GPUType, JobType, SchedulingMode,
    MODEL_PROFILES,
)


class ArrivalProcess(str, Enum):
    POISSON = "poisson"
    PARETO  = "pareto"
    DIURNAL = "diurnal"


class TraceType(str, Enum):
    PHILLY   = "philly"
    ALIBABA  = "alibaba"
    GAVEL    = "gavel"
    SYNTHETIC = "synthetic"


@dataclass
class WorkloadConfig:
    duration: float = 3600.0
    arrival_process: ArrivalProcess = ArrivalProcess.POISSON
    mean_arrival_interval: float = 30.0    # seconds
    num_users: int = 10
    # job type mix
    training_fraction: float = 0.70
    inference_fraction: float = 0.15
    llm_fraction: float = 0.10
    hpo_fraction: float = 0.05
    # GPU request distribution (power-law)
    gpu_request_min: int = 1
    gpu_request_max: int = 16
    # model distribution
    arch_weights: Optional[dict] = None   # if None, use defaults
    # deadline probability
    deadline_fraction: float = 0.20
    deadline_slack_factor: float = 3.0    # deadline = estimated_duration × factor
    # scheduling mode mix
    gang_fraction: float = 0.70
    elastic_fraction: float = 0.20
    sharing_fraction: float = 0.10
    # duration distribution (log-normal)
    mean_iter: float = 10_000.0
    std_iter: float = 8_000.0
    rng_seed: Optional[int] = None


# Preset trace configs
TRACE_CONFIGS: dict[str, WorkloadConfig] = {
    "philly": WorkloadConfig(
        arrival_process=ArrivalProcess.PARETO,
        mean_arrival_interval=20.0,
        training_fraction=0.95, inference_fraction=0.03,
        llm_fraction=0.01, hpo_fraction=0.01,
        deadline_fraction=0.10,
    ),
    "alibaba": WorkloadConfig(
        arrival_process=ArrivalProcess.DIURNAL,
        mean_arrival_interval=15.0,
        training_fraction=0.60, inference_fraction=0.25,
        llm_fraction=0.10, hpo_fraction=0.05,
        gpu_request_max=32,
        deadline_fraction=0.30,
    ),
    "gavel": WorkloadConfig(
        arrival_process=ArrivalProcess.POISSON,
        mean_arrival_interval=30.0,
        training_fraction=0.80, inference_fraction=0.10,
        llm_fraction=0.05, hpo_fraction=0.05,
        deadline_fraction=0.15,
    ),
}


def _pareto_sample(scale: float, alpha: float = 1.5, rng: random.Random = None) -> float:
    """Heavy-tailed Pareto: X = scale / U^(1/alpha), U ~ Uniform(0,1)."""
    r = rng or random
    u = r.random()
    return scale / (u ** (1.0 / alpha))


def _diurnal_rate(t: float, mean_rate: float) -> float:
    """
    24-hour diurnal pattern (Alibaba trace).
    Peak at t=10h, trough at t=4h.
    """
    hour = (t / 3600.0) % 24.0
    factor = 0.4 + 0.6 * (0.5 + 0.5 * math.sin(2 * math.pi * (hour - 4) / 24.0))
    return mean_rate * factor


def _power_law_gpus(
    min_g: int, max_g: int, rng: random.Random, alpha: float = 2.0
) -> int:
    """Power-law distributed GPU request count (survey observation)."""
    weights = [1.0 / (k ** alpha) for k in range(min_g, max_g + 1)]
    total = sum(weights)
    r = rng.random() * total
    cumul = 0.0
    for k, w in zip(range(min_g, max_g + 1), weights):
        cumul += w
        if r <= cumul:
            return k
    return max_g


def _sample_arch(arch_weights: Optional[dict], rng: random.Random) -> ModelArch:
    default = {
        ModelArch.RESNET50:    0.25,
        ModelArch.RESNET18:    0.15,
        ModelArch.VGG19:       0.10,
        ModelArch.MOBILENETV3: 0.08,
        ModelArch.BERT:        0.15,
        ModelArch.TRANSFORMER: 0.12,
        ModelArch.GPT2:        0.05,
        ModelArch.RECNET:      0.05,
        ModelArch.PPO:         0.05,
    }
    weights = arch_weights or default
    arches = list(weights.keys())
    probs = [weights[a] for a in arches]
    return rng.choices(arches, weights=probs, k=1)[0]


def _sample_iterations(mean: float, std: float, rng: random.Random) -> int:
    # log-normal duration distribution
    mu = math.log(max(mean, 1.0))
    sigma = math.log(max(1.0 + std / mean, 1.01))
    raw = rng.lognormvariate(mu, sigma)
    return max(100, int(raw))


class WorkloadGenerator:
    """
    Generates a list of DL jobs with configurable arrival processes.
    """

    def __init__(self, config: Optional[WorkloadConfig] = None, trace: str = ""):
        if trace and trace in TRACE_CONFIGS:
            self.config = TRACE_CONFIGS[trace]
        else:
            self.config = config or WorkloadConfig()
        seed = self.config.rng_seed or 42
        self.rng = random.Random(seed)

    def generate(self) -> list[AnyJob]:
        cfg = self.config
        jobs: list[AnyJob] = []
        t = 0.0
        job_idx = 0

        while t < cfg.duration:
            # ── Inter-arrival time ────────────────────────────────────────
            if cfg.arrival_process == ArrivalProcess.POISSON:
                iat = self.rng.expovariate(1.0 / cfg.mean_arrival_interval)
            elif cfg.arrival_process == ArrivalProcess.PARETO:
                iat = _pareto_sample(cfg.mean_arrival_interval * 0.5, 1.5, self.rng)
            else:  # DIURNAL
                rate = _diurnal_rate(t, 1.0 / cfg.mean_arrival_interval)
                iat = self.rng.expovariate(max(rate, 1e-6))

            t += iat
            if t >= cfg.duration:
                break

            # ── Job type selection ────────────────────────────────────────
            r = self.rng.random()
            cumul = 0.0
            job_type = JobType.TRAINING
            for jtype, frac in [
                (JobType.TRAINING,  cfg.training_fraction),
                (JobType.INFERENCE, cfg.inference_fraction),
                (JobType.LLM_TRAIN, cfg.llm_fraction),
                (JobType.HPO,       cfg.hpo_fraction),
            ]:
                cumul += frac
                if r <= cumul:
                    job_type = jtype
                    break

            job = self._make_job(job_type, t, job_idx)
            jobs.append(job)
            job_idx += 1

        return jobs

    def _make_job(self, job_type: JobType, t: float, idx: int) -> AnyJob:
        cfg = self.config
        user_id = f"user_{self.rng.randint(0, cfg.num_users - 1)}"
        arch = _sample_arch(cfg.arch_weights, self.rng)
        profile = MODEL_PROFILES[arch]

        # scheduling mode
        r = self.rng.random()
        if r < cfg.gang_fraction:
            smode = SchedulingMode.GANG
        elif r < cfg.gang_fraction + cfg.elastic_fraction:
            smode = SchedulingMode.ELASTIC if profile.elastic_compatible else SchedulingMode.GANG
        else:
            smode = SchedulingMode.SHARING

        num_gpus = _power_law_gpus(cfg.gpu_request_min, cfg.gpu_request_max, self.rng)
        num_iter  = _sample_iterations(cfg.mean_iter, cfg.std_iter, self.rng)
        mem_gb    = profile.memory_per_replica_gb * (1.0 + self.rng.uniform(0, 0.3))
        bs        = profile.ref_batch_size

        # ── Training ──────────────────────────────────────────────────────
        if job_type == JobType.TRAINING:
            estimated_duration = num_iter / (profile.base_throughput_k80 * 2.0)
            deadline = (
                t + estimated_duration * cfg.deadline_slack_factor
                if self.rng.random() < cfg.deadline_fraction else None
            )
            return TrainingJob(
                job_id=f"T{idx:05d}",
                job_type=job_type,
                submit_time=t,
                arch=arch,
                batch_size=bs,
                num_iterations=num_iter,
                num_gpus_requested=num_gpus,
                memory_per_gpu_gb=mem_gb,
                scheduling_mode=smode,
                min_gpus=1,
                max_gpus=min(num_gpus * 2, cfg.gpu_request_max),
                deadline=deadline,
                checkpoint_overhead_sec=max(30.0, mem_gb * 5.0),
                user_id=user_id,
            )

        # ── Inference ─────────────────────────────────────────────────────
        elif job_type == JobType.INFERENCE:
            slo = self.rng.choice([50.0, 100.0, 200.0, 500.0])
            return InferenceJob(
                job_id=f"I{idx:05d}",
                job_type=job_type,
                submit_time=t,
                arch=arch,
                batch_size=1,
                num_gpus_requested=self.rng.choice([1, 2, 4]),
                memory_per_gpu_gb=mem_gb,
                latency_slo_ms=slo,
                total_queries=self.rng.randint(500, 5000),
                query_rate_per_sec=self.rng.uniform(10.0, 500.0),
                user_id=user_id,
            )

        # ── LLM ──────────────────────────────────────────────────────────
        elif job_type == JobType.LLM_TRAIN:
            size_b = self.rng.choice([1.5, 7.0, 13.0, 70.0])
            pp = self.rng.choice([1, 2, 4])
            tp = self.rng.choice([1, 2, 4])
            dp = max(1, num_gpus // (pp * tp))
            iters = max(500, num_iter // 10)
            estimated_duration = iters / (profile.base_throughput_k80 * 0.5)
            deadline = (
                t + estimated_duration * cfg.deadline_slack_factor
                if self.rng.random() < cfg.deadline_fraction else None
            )
            return LLMJob(
                job_id=f"L{idx:05d}",
                job_type=job_type,
                submit_time=t,
                arch=ModelArch.GPT2,
                num_params_billions=size_b,
                batch_size=8,
                seq_length=2048,
                pipeline_stages=pp,
                tensor_parallel=tp,
                data_parallel=dp,
                num_iterations=iters,
                deadline=deadline,
                user_id=user_id,
            )

        # ── HPO ───────────────────────────────────────────────────────────
        else:
            trials = self.rng.choice([4, 8, 16])
            return HPOJob(
                job_id=f"H{idx:05d}",
                job_type=job_type,
                submit_time=t,
                arch=arch,
                num_trials=trials,
                gpus_per_trial=self.rng.choice([1, 2]),
                memory_per_gpu_gb=mem_gb,
                user_id=user_id,
            )
