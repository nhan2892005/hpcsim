"""
DL Job Models — theoretically accurate.

Implements job characteristics T1-T6 (training) and I1-I2 (inference)
from the survey. Throughput models calibrated on Gavel benchmark dataset.
"""

from __future__ import annotations
from dataclasses import dataclass, field
from enum import Enum
from typing import Optional
import math
import uuid

from ..cluster.hardware import GPUType, GPU_SPECS, ring_allreduce_time_sec


class JobType(str, Enum):
    TRAINING  = "training"
    INFERENCE = "inference"
    LLM_TRAIN = "llm_training"
    LLM_INFER = "llm_inference"
    HPO       = "hpo"


class JobStatus(str, Enum):
    PENDING   = "pending"
    RUNNING   = "running"
    PREEMPTED = "preempted"
    COMPLETED = "completed"
    FAILED    = "failed"


class SchedulingMode(str, Enum):
    GANG    = "gang"
    ELASTIC = "elastic"
    SHARING = "sharing"


class ModelArch(str, Enum):
    RESNET18    = "resnet18"
    RESNET50    = "resnet50"
    VGG19       = "vgg19"
    MOBILENETV3 = "mobilenetv3"
    BERT        = "bert"
    GPT2        = "gpt2"
    TRANSFORMER = "transformer_lm"
    RECNET      = "recommendation"
    PPO         = "ppo_rl"


# Model Profiles 

@dataclass(frozen=True)
class ModelProfile:
    """
    Characterizes compute/memory/communication footprint of a model.

    gpu_affinity: throughput multiplier per GPU type vs K80 baseline.
    Calibrated from Survey Figure 2(f) and Gavel dataset.
    """
    arch: ModelArch
    param_millions: float
    memory_per_replica_gb: float
    ref_batch_size: int
    base_throughput_k80: float    # iter/s solo on K80
    gpu_affinity: dict            # GPUType → float
    comm_compute_ratio: float     # higher = more communication-bound (T2)
    elastic_compatible: bool      # safe for elastic training (T6)
    is_compute_bound: bool        # True=compute, False=memory-bound


MODEL_PROFILES: dict[ModelArch, ModelProfile] = {

    ModelArch.RESNET50: ModelProfile(
        arch=ModelArch.RESNET50,
        param_millions=25.6, memory_per_replica_gb=3.2,
        ref_batch_size=64, base_throughput_k80=1.0,
        gpu_affinity={
            GPUType.K80: 1.0, GPUType.P100: 3.5, GPUType.V100: 6.5,
            GPUType.A100: 19.5, GPUType.RTX3090: 8.2,
            GPUType.K80_UNCONSOL: 0.5, GPUType.P100_UNCONSOL: 1.75,
            GPUType.V100_UNCONSOL: 3.25,
        },
        comm_compute_ratio=0.08, elastic_compatible=True, is_compute_bound=True,
    ),

    ModelArch.RESNET18: ModelProfile(
        arch=ModelArch.RESNET18,
        param_millions=11.7, memory_per_replica_gb=1.8,
        ref_batch_size=64, base_throughput_k80=1.8,
        gpu_affinity={
            GPUType.K80: 1.0, GPUType.P100: 3.2, GPUType.V100: 6.0,
            GPUType.A100: 17.0, GPUType.RTX3090: 7.5,
            GPUType.K80_UNCONSOL: 0.5, GPUType.P100_UNCONSOL: 1.6,
            GPUType.V100_UNCONSOL: 3.0,
        },
        comm_compute_ratio=0.06, elastic_compatible=True, is_compute_bound=True,
    ),

    ModelArch.VGG19: ModelProfile(
        arch=ModelArch.VGG19,
        param_millions=143.7, memory_per_replica_gb=8.0,
        ref_batch_size=32, base_throughput_k80=0.5,
        gpu_affinity={
            GPUType.K80: 1.0, GPUType.P100: 3.0, GPUType.V100: 6.3,
            GPUType.A100: 17.0, GPUType.RTX3090: 8.2,
            GPUType.K80_UNCONSOL: 0.5, GPUType.P100_UNCONSOL: 1.5,
            GPUType.V100_UNCONSOL: 3.15,
        },
        comm_compute_ratio=0.15, elastic_compatible=True, is_compute_bound=True,
    ),

    ModelArch.MOBILENETV3: ModelProfile(
        arch=ModelArch.MOBILENETV3,
        param_millions=5.4, memory_per_replica_gb=1.2,
        ref_batch_size=64, base_throughput_k80=2.5,
        gpu_affinity={
            GPUType.K80: 1.0, GPUType.P100: 1.6, GPUType.V100: 1.9,
            GPUType.A100: 1.8, GPUType.RTX3090: 1.6,
            GPUType.K80_UNCONSOL: 0.5, GPUType.P100_UNCONSOL: 0.8,
            GPUType.V100_UNCONSOL: 0.95,
        },
        comm_compute_ratio=0.04, elastic_compatible=True, is_compute_bound=False,
    ),

    ModelArch.BERT: ModelProfile(
        arch=ModelArch.BERT,
        param_millions=110.0, memory_per_replica_gb=6.0,
        ref_batch_size=32, base_throughput_k80=0.3,
        gpu_affinity={
            GPUType.K80: 1.0, GPUType.P100: 4.0, GPUType.V100: 8.0,
            GPUType.A100: 30.0, GPUType.RTX3090: 12.0,
            GPUType.K80_UNCONSOL: 0.5, GPUType.P100_UNCONSOL: 2.0,
            GPUType.V100_UNCONSOL: 4.0,
        },
        comm_compute_ratio=0.12, elastic_compatible=True, is_compute_bound=True,
    ),

    ModelArch.TRANSFORMER: ModelProfile(
        arch=ModelArch.TRANSFORMER,
        param_millions=350.0, memory_per_replica_gb=12.0,
        ref_batch_size=16, base_throughput_k80=0.15,
        gpu_affinity={
            GPUType.K80: 1.0, GPUType.P100: 4.5, GPUType.V100: 9.0,
            GPUType.A100: 35.0, GPUType.RTX3090: 14.0,
            GPUType.K80_UNCONSOL: 0.5, GPUType.P100_UNCONSOL: 2.25,
            GPUType.V100_UNCONSOL: 4.5,
        },
        comm_compute_ratio=0.20, elastic_compatible=True, is_compute_bound=True,
    ),

    ModelArch.GPT2: ModelProfile(
        arch=ModelArch.GPT2,
        param_millions=1500.0, memory_per_replica_gb=24.0,
        ref_batch_size=8, base_throughput_k80=0.05,
        gpu_affinity={
            GPUType.K80: 1.0, GPUType.P100: 5.0, GPUType.V100: 12.0,
            GPUType.A100: 50.0, GPUType.RTX3090: 18.0,
            GPUType.K80_UNCONSOL: 0.0, GPUType.P100_UNCONSOL: 0.0,
            GPUType.V100_UNCONSOL: 0.5,
        },
        comm_compute_ratio=0.30, elastic_compatible=False, is_compute_bound=True,
    ),

    ModelArch.RECNET: ModelProfile(
        arch=ModelArch.RECNET,
        param_millions=50.0, memory_per_replica_gb=10.0,
        ref_batch_size=512, base_throughput_k80=1.5,
        gpu_affinity={
            GPUType.K80: 1.0, GPUType.P100: 1.3, GPUType.V100: 1.5,
            GPUType.A100: 2.0, GPUType.RTX3090: 1.4,
            GPUType.K80_UNCONSOL: 0.5, GPUType.P100_UNCONSOL: 0.65,
            GPUType.V100_UNCONSOL: 0.75,
        },
        comm_compute_ratio=0.05, elastic_compatible=True, is_compute_bound=False,
    ),

    ModelArch.PPO: ModelProfile(
        arch=ModelArch.PPO,
        param_millions=2.0, memory_per_replica_gb=1.5,
        ref_batch_size=64, base_throughput_k80=2.0,
        gpu_affinity={
            GPUType.K80: 1.0, GPUType.P100: 1.9, GPUType.V100: 1.8,
            GPUType.A100: 1.8, GPUType.RTX3090: 1.6,
            GPUType.K80_UNCONSOL: 0.5, GPUType.P100_UNCONSOL: 0.95,
            GPUType.V100_UNCONSOL: 0.9,
        },
        comm_compute_ratio=0.03, elastic_compatible=True, is_compute_bound=False,
    ),
}


# Throughput functions 

def solo_throughput(arch: ModelArch, gpu_type: GPUType, batch_size: int) -> float:
    """
    Throughput (iter/s) of one job on one GPU.
    T1 heterogeneous affinity × batch-size scaling.
    """
    p = MODEL_PROFILES.get(arch)
    if not p:
        return 1.0
    affinity = p.gpu_affinity.get(gpu_type, 0.5)
    alpha = 0.85 if p.is_compute_bound else 0.60
    bs_factor = (batch_size / p.ref_batch_size) ** alpha
    return p.base_throughput_k80 * affinity * bs_factor


def multi_gpu_throughput(
    arch: ModelArch,
    gpu_type: GPUType,
    batch_size: int,
    num_gpus: int,
    bandwidth_gbps: float,
) -> float:
    """
    Data-parallel throughput on k GPUs.

    t(k) = t(1) · k · η(k)
    η(k) = t_compute / (t_compute + t_comm)   [Amdahl + ring-allreduce]
    Survey T2 placement sensitivity model.
    """
    if num_gpus <= 0:
        return 0.0
    t1 = solo_throughput(arch, gpu_type, batch_size)
    if num_gpus == 1:
        return t1
    p = MODEL_PROFILES.get(arch)
    if not p or t1 <= 0:
        return t1 * num_gpus * 0.8
    t_compute = 1.0 / t1
    t_comm = ring_allreduce_time_sec(
        int(p.param_millions * 1e6), num_gpus, bandwidth_gbps
    )
    efficiency = t_compute / (t_compute + t_comm)
    return t1 * num_gpus * efficiency


def goodput(
    arch: ModelArch,
    gpu_type: GPUType,
    batch_size: int,
    num_gpus: int,
    bandwidth_gbps: float,
) -> float:
    """
    Goodput = throughput × statistical_efficiency(global_batch_size).
    Pollux [Qiao et al., OSDI'21] Eq.(1).
    SE = sqrt(B_crit / (bs·k))  for bs·k > B_crit
    """
    tp = multi_gpu_throughput(arch, gpu_type, batch_size, num_gpus, bandwidth_gbps)
    p = MODEL_PROFILES.get(arch)
    if not p:
        return tp
    global_batch = batch_size * num_gpus
    B_crit = p.ref_batch_size * 32
    se = 1.0 if global_batch <= B_crit else math.sqrt(B_crit / global_batch)
    return tp * se


def colocation_throughput(
    arch1: ModelArch, arch2: ModelArch,
    gpu_type: GPUType,
    batch1: int, batch2: int,
) -> tuple[float, float]:
    """
    Throughput of two co-located jobs.  Survey T5 / Space-Time model.
    Interference depends on both jobs' memory-boundedness.
    """
    p1 = MODEL_PROFILES.get(arch1)
    p2 = MODEL_PROFILES.get(arch2)
    t1 = solo_throughput(arch1, gpu_type, batch1)
    t2 = solo_throughput(arch2, gpu_type, batch2)
    if not p1 or not p2:
        return t1 * 0.70, t2 * 0.70

    if not p1.is_compute_bound and not p2.is_compute_bound:
        factor = 0.55   # both memory-bound → high contention
    elif p1.is_compute_bound and p2.is_compute_bound:
        factor = 0.75   # both compute-bound → moderate contention
    else:
        factor = 0.65   # mixed

    spec = GPU_SPECS.get(gpu_type)
    if spec:
        mem_pressure = (p1.memory_per_replica_gb + p2.memory_per_replica_gb) / spec.memory_gb
        factor *= max(0.5, 1.0 - 0.3 * mem_pressure)
    return t1 * factor, t2 * factor


# Job Base 

@dataclass
class BaseJob:
    job_id: str = field(default_factory=lambda: str(uuid.uuid4())[:8])
    job_type: JobType = JobType.TRAINING
    status: JobStatus = JobStatus.PENDING
    submit_time: float = 0.0
    start_time: Optional[float] = None
    end_time: Optional[float] = None
    preempt_count: int = 0
    allocated_gpus: list = field(default_factory=list)
    user_id: str = "user_0"
    priority: int = 0

    @property
    def queue_time(self) -> float:
        if self.start_time is None:
            return float("inf")
        return self.start_time - self.submit_time

    @property
    def jct(self) -> float:
        if self.end_time is None:
            return float("inf")
        return self.end_time - self.submit_time


# Training Job 

@dataclass
class TrainingJob(BaseJob):
    """
    DL Training Job — implements T1-T6 from the survey.
    Duration model: JCT = queue + remaining_iter / throughput + overhead
    """
    job_type: JobType = JobType.TRAINING
    arch: ModelArch = ModelArch.RESNET50
    batch_size: int = 64
    num_iterations: int = 10_000
    total_epochs: int = 100
    num_gpus_requested: int = 1
    gpu_type_preference: Optional[GPUType] = None
    memory_per_gpu_gb: float = 4.0
    scheduling_mode: SchedulingMode = SchedulingMode.GANG
    min_gpus: int = 1
    max_gpus: int = 8
    deadline: Optional[float] = None
    completed_iterations: int = 0
    attained_service: float = 0.0   # GPU-seconds (Tiresias LAS metric)
    accumulated_work: float = 0.0
    checkpoint_overhead_sec: float = 30.0
    profiled_throughput: Optional[float] = None

    def remaining_iterations(self) -> int:
        return max(0, self.num_iterations - self.completed_iterations)

    def progress(self) -> float:
        return self.completed_iterations / max(1, self.num_iterations)


# Inference Job 

@dataclass
class InferenceJob(BaseJob):
    """
    DL Inference Job — implements I1-I2 from the survey.
    Deterministic execution, SLO-aware.
    """
    job_type: JobType = JobType.INFERENCE
    arch: ModelArch = ModelArch.RESNET50
    batch_size: int = 1
    num_replicas: int = 1
    latency_slo_ms: float = 100.0
    accuracy_threshold: float = 0.9
    query_rate_per_sec: float = 100.0
    total_queries: int = 1000
    completed_queries: int = 0
    slo_violations: int = 0
    max_batch_size: int = 32
    current_batch_size: int = 1
    memory_per_gpu_gb: float = 2.0
    num_gpus_requested: int = 1

    def per_query_latency_ms(self, gpu_type: GPUType, batch_size: int) -> float:
        """
        Deterministic per-query latency (Survey I1).
        Latency = batch_size / throughput + fixed_overhead
        """
        tp = solo_throughput(self.arch, gpu_type, batch_size)
        if tp <= 0:
            return float("inf")
        p = MODEL_PROFILES.get(self.arch)
        overhead_ms = 2.0 + (p.memory_per_replica_gb * 0.5 if p else 2.0)
        return (batch_size / tp) * 1000.0 + overhead_ms

    def optimal_batch_size(self, gpu_type: GPUType) -> int:
        """Largest batch meeting SLO (AIMD from Clipper survey)."""
        for bs in [32, 16, 8, 4, 2, 1]:
            if self.per_query_latency_ms(gpu_type, bs) <= self.latency_slo_ms:
                return bs
        return 1


# LLM Job 

@dataclass
class LLMJob(BaseJob):
    """LLM Training/Inference with pipeline + tensor parallelism."""
    job_type: JobType = JobType.LLM_TRAIN
    arch: ModelArch = ModelArch.GPT2
    num_params_billions: float = 1.5
    batch_size: int = 8
    seq_length: int = 2048
    pipeline_stages: int = 1
    tensor_parallel: int = 1
    data_parallel: int = 1
    num_iterations: int = 5000
    completed_iterations: int = 0
    attained_service: float = 0.0
    accumulated_work: float = 0.0
    checkpoint_overhead_sec: float = 300.0
    deadline: Optional[float] = None
    profiled_throughput: Optional[float] = None
    scheduling_mode: SchedulingMode = SchedulingMode.GANG
    min_gpus: int = 1
    max_gpus: int = 64
    gpu_type_preference: Optional[GPUType] = None

    @property
    def num_gpus_requested(self) -> int:
        return max(1, self.pipeline_stages * self.tensor_parallel * self.data_parallel)

    @property
    def memory_per_gpu_gb(self) -> float:
        # 10 bytes/param (model + optimizer + gradients, mixed precision)
        total_bytes = self.num_params_billions * 1e9 * 10
        n = max(1, self.pipeline_stages * self.tensor_parallel)
        return total_bytes / (n * 1e9)

    def progress(self) -> float:
        return self.completed_iterations / max(1, self.num_iterations)

    def remaining_iterations(self) -> int:
        return max(0, self.num_iterations - self.completed_iterations)


# HPO Job 

@dataclass
class HPOJob(BaseJob):
    """
    Hyperparameter Optimization job (Survey Section 5.1).
    Manages multiple training trials with ASHA early stopping.
    """
    job_type: JobType = JobType.HPO
    arch: ModelArch = ModelArch.RESNET50
    num_trials: int = 16
    max_epochs_per_trial: int = 50
    budget_epochs: int = 200
    active_trials: dict = field(default_factory=dict)
    completed_trials: list = field(default_factory=list)
    best_accuracy: float = 0.0
    gpus_per_trial: int = 1
    memory_per_gpu_gb: float = 4.0

    @property
    def num_gpus_requested(self) -> int:
        return self.num_trials * self.gpus_per_trial


# Union type
AnyJob = TrainingJob | InferenceJob | LLMJob | HPOJob


# 
# NEW: Resource type enum + CPU/MIG job classes
# 

from ..cluster.hardware import CPUType, MIGProfile   # noqa — appended to file


class ResourceType(str, Enum):
    """
    Declares what primary resource a job requires.
    Scheduler uses this to route jobs to the correct pool.
    """
    GPU      = "gpu"       # Full physical GPU(s)
    MIG      = "mig"       # MIG slice(s) on A100/H100
    CPU      = "cpu"       # CPU cores only  (no GPU)
    CPU_GPU  = "cpu_gpu"   # Both CPU cores AND GPU(s) — e.g. data-parallel w/ local PS


# CPU Job 

@dataclass
class CPUJob(BaseJob):
    """
    CPU-only job: data preprocessing, feature extraction, post-processing,
    parameter-server workloads, traditional HPC simulation (non-GPU).

    Runtime model:
        duration = base_duration / (cpu_perf_factor × num_cores / ref_cores)
        where cpu_perf_factor comes from CPUSpec.mt_perf.
    """
    job_type:         JobType      = JobType.TRAINING    # repurposed for CPU jobs
    resource_type:    ResourceType = ResourceType.CPU

    # Resource request
    num_cpus_requested: int        = 4     # total CPU cores needed
    min_cpus:          int         = 1     # elastic minimum
    max_cpus:          int         = 64    # elastic maximum
    cpu_type_preference: Optional[CPUType] = None

    # Duration model
    base_duration_sec: float       = 600.0  # duration at ref_cores
    ref_cores:         int         = 4      # reference parallelism
    memory_gb:         float       = 8.0    # RAM required

    # SLO
    deadline:          Optional[float] = None

    # Runtime state
    allocated_cpus:    list        = field(default_factory=list)  # ["cpu_id:N", ...]
    attained_service:  float       = 0.0   # CPU-core-seconds (for fairness)

    @property
    def num_gpus_requested(self) -> int:
        """Compatibility with schedulers that check num_gpus_requested."""
        return 0

    @property
    def memory_per_gpu_gb(self) -> float:
        return 0.0

    def effective_duration(self, actual_cores: int, cpu_perf: float = 1.0) -> float:
        """
        Estimated runtime given actual_cores and cpu_perf_factor.
        Assumes perfect scaling up to num_cpus_requested, then Amdahl-bounded.
        """
        if actual_cores <= 0:
            return float("inf")
        # Simple model: linear speedup with 80% parallel fraction
        parallel_frac = 0.80
        speedup = 1.0 / ((1 - parallel_frac) + parallel_frac / actual_cores * self.ref_cores)
        return self.base_duration_sec / (speedup * cpu_perf)

    def progress(self) -> float:
        if self.start_time is None or self.end_time is None:
            return 0.0
        return 1.0  # simplified: CPU jobs run to completion


# MIG Job 

@dataclass
class MIGJob(BaseJob):
    """
    Job that requests MIG slice(s) instead of full GPUs.
    Useful for small inference tasks, light training, or multi-tenant scenarios.

    MIG slices provide strong isolation: dedicated compute + memory partition.
    Typical use cases:
      - Small fine-tuning runs (1g.10gb = 1/7 of A100)
      - Inference serving with isolation guarantees
      - Multi-tenant research clusters
    """
    job_type:         JobType      = JobType.INFERENCE
    resource_type:    ResourceType = ResourceType.MIG

    # MIG resource request
    num_mig_requested: int         = 1
    mig_profile:      MIGProfile   = MIGProfile.G1_10GB
    gpu_type_preference: Optional[GPUType] = None

    # Model / task
    arch:             ModelArch    = ModelArch.RESNET50
    batch_size:       int          = 1
    num_iterations:   int          = 1000
    completed_iterations: int      = 0

    # Duration
    base_duration_sec: float       = 120.0
    deadline:          Optional[float] = None

    # Runtime
    allocated_mig:    list         = field(default_factory=list)  # mig_ids
    attained_service: float        = 0.0

    @property
    def num_gpus_requested(self) -> int:
        """MIG jobs do not consume full GPUs."""
        return 0

    @property
    def memory_per_gpu_gb(self) -> float:
        from ..cluster.hardware import MIG_PROFILE_SPECS
        return MIG_PROFILE_SPECS.get(self.mig_profile, {}).get("memory_gb", 10.0)

    def progress(self) -> float:
        return self.completed_iterations / max(1, self.num_iterations)


# Mixed CPU+GPU Job 

@dataclass
class HybridJob(BaseJob):
    """
    Job that requires both CPU cores AND GPUs simultaneously.

    Real-world examples:
      - Data-parallel training with on-node data loaders (CPU) + GPU compute
      - Reinforcement learning: CPU environment + GPU policy network
      - Simulation + ML: CPU physics + GPU neural network
      - PyTorch DataLoader workers (CPU) alongside GPU training

    The scheduler must allocate BOTH resources before the job can start.
    """
    job_type:          JobType      = JobType.TRAINING
    resource_type:     ResourceType = ResourceType.CPU_GPU

    # GPU request
    num_gpus_requested: int         = 4
    gpu_type_preference: Optional[GPUType] = None
    memory_per_gpu_gb:  float       = 8.0
    scheduling_mode:    SchedulingMode = SchedulingMode.GANG
    min_gpus:           int         = 1
    max_gpus:           int         = 8

    # CPU request (e.g. DataLoader workers)
    num_cpus_requested: int         = 8   # typically 2× num_gpus
    cpu_type_preference: Optional[CPUType] = None

    # Task
    arch:              ModelArch    = ModelArch.RESNET50
    num_iterations:    int          = 10_000
    completed_iterations: int       = 0
    attained_service:  float        = 0.0
    accumulated_work:  float        = 0.0
    deadline:          Optional[float] = None

    # Runtime
    allocated_cpus:    list         = field(default_factory=list)  # ["cpu_id:N"]

    def remaining_iterations(self) -> int:
        return max(0, self.num_iterations - self.completed_iterations)

    def progress(self) -> float:
        return self.completed_iterations / max(1, self.num_iterations)


# Update AnyJob union
AnyJob = TrainingJob | InferenceJob | LLMJob | HPOJob | CPUJob | MIGJob | HybridJob
