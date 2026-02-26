"""
Scheduling Algorithms for HPC GPU Cluster.

Implemented algorithms (12 total):
1.  FIFO             — First In First Out (baseline)
2.  SJF              — Shortest Job First
3.  LAS (Tiresias)   — Least Attained Service, MLFQ [Gu et al., NSDI'19]
4.  E-LAS            — Epoch-progress LAS [Sultana et al., ICPP'20]
5.  MLFQ             — Multi-Level Feedback Queue (HPC standard)
6.  Gavel            — Heterogeneity-aware round-based [Narayanan et al., OSDI'20]
7.  Pollux           — Goodput-optimized elastic [Qiao et al., OSDI'21]
8.  Themis           — Finish-time fairness ρ [Mahajan et al., NSDI'20]
9.  Chronus          — Deadline-aware SLO/BE [Gao et al., SoCC'21]
10. ElasticFlow      — Serverless elastic [Gu et al., ASPLOS'23]
11. MaxMinFairness   — Dominant Resource Fairness [Ghodsi et al., NSDI'11]
12. Backfill         — Conservative backfill EASY (HPC standard)
"""

from __future__ import annotations
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Optional, TYPE_CHECKING
import math

if TYPE_CHECKING:
    from ..cluster.cluster import Cluster

from ..workload.job import (
    AnyJob, TrainingJob, InferenceJob, LLMJob, HPOJob,
    CPUJob, MIGJob, HybridJob, ResourceType,
    ModelArch, GPUType, JobType, SchedulingMode,
    goodput, solo_throughput, MODEL_PROFILES,
)
from ..cluster.hardware import CPUType, MIGProfile


# ─────────────────────────────────────────────────────────────────────────────
# Scheduling Decision — supports GPU, MIG, CPU, and Hybrid allocations
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class Allocation:
    job:         AnyJob
    gpu_ids:     list[str]              = field(default_factory=list)
    mig_ids:     list[str]              = field(default_factory=list)
    cpu_alloc:   list[str]              = field(default_factory=list)  # ["cpu_id:N"]

    @property
    def resource_type(self) -> str:
        if self.mig_ids:
            return "mig"
        if self.cpu_alloc and not self.gpu_ids:
            return "cpu"
        if self.cpu_alloc and self.gpu_ids:
            return "hybrid"
        return "gpu"


@dataclass
class SchedulingDecision:
    allocations: list[Allocation] = field(default_factory=list)
    preemptions: list[str]        = field(default_factory=list)  # job_ids
    # Optional metadata set by GASMARLScheduler when it issues a delay action.
    # BackfillWrapper reads this to compute the correct backfill window.
    # Schema: {"delay_type": int, "release_time": float, "head_job_id": str,
    #           "head_req_gpus": int}
    delay_info: Optional[dict]    = field(default=None)

    def add(self, job: AnyJob, gpu_ids: list[str],
            mig_ids: list[str] | None = None,
            cpu_alloc: list[str] | None = None):
        self.allocations.append(Allocation(
            job=job,
            gpu_ids=gpu_ids,
            mig_ids=mig_ids or [],
            cpu_alloc=cpu_alloc or [],
        ))

    def add_cpu(self, job: AnyJob, cpu_alloc: list[str]):
        self.allocations.append(Allocation(job=job, cpu_alloc=cpu_alloc))

    def add_mig(self, job: AnyJob, mig_ids: list[str]):
        self.allocations.append(Allocation(job=job, mig_ids=mig_ids))

    def add_hybrid(self, job: AnyJob, gpu_ids: list[str], cpu_alloc: list[str]):
        self.allocations.append(Allocation(job=job, gpu_ids=gpu_ids, cpu_alloc=cpu_alloc))

    def preempt(self, job_id: str):
        self.preemptions.append(job_id)


# ─────────────────────────────────────────────────────────────────────────────
# BaseScheduler
# ─────────────────────────────────────────────────────────────────────────────

class BaseScheduler(ABC):
    """
    Abstract base class for all schedulers.

    Subclasses implement schedule() which returns a SchedulingDecision.
    Helper methods handle GPU, MIG, and CPU resource discovery.
    """

    name: str = "base"

    def __init__(self, cluster: "Cluster"):
        self.cluster = cluster

    # ── GPU helpers ───────────────────────────────────────────────────────────

    def _find_gpus(
        self,
        job: AnyJob,
        prefer_consolidated: bool = True,
        gpu_type: Optional[GPUType] = None,
    ) -> Optional[list[str]]:
        """Find physical GPUs for a GPU or Hybrid job."""
        n = getattr(job, "num_gpus_requested", 0)
        if n <= 0:
            return None
        gt = gpu_type or getattr(job, "gpu_type_preference", None)
        return self.cluster.find_best_placement(n, gt, prefer_consolidated)

    # ── MIG helpers ───────────────────────────────────────────────────────────

    def _find_mig(
        self,
        job: AnyJob,
        profile: Optional[MIGProfile] = None,
        gpu_type: Optional[GPUType] = None,
    ) -> Optional[list[str]]:
        """Find MIG slices for a MIGJob."""
        if not self.cluster.has_mig():
            return None
        n       = getattr(job, "num_mig_requested", 1)
        prof    = profile or getattr(job, "mig_profile", None)
        gt      = gpu_type or getattr(job, "gpu_type_preference", None)
        return self.cluster.find_mig_slices(n, prof, gt)

    # ── CPU helpers ───────────────────────────────────────────────────────────

    def _find_cpus(
        self,
        job: AnyJob,
        cpu_type: Optional[CPUType] = None,
        prefer_consolidated: bool = True,
    ) -> Optional[list[str]]:
        """Find CPU cores for a CPUJob or HybridJob."""
        if not self.cluster.has_cpu_nodes():
            return None
        n  = getattr(job, "num_cpus_requested", 0)
        if n <= 0:
            return None
        ct = cpu_type or getattr(job, "cpu_type_preference", None)
        return self.cluster.find_cpu_cores(n, ct, prefer_consolidated)

    # ── Unified resource finder ───────────────────────────────────────────────

    def _find_resources(
        self, job: AnyJob
    ) -> Optional[tuple[list[str], list[str], list[str]]]:
        """
        Find resources for any job type.
        Returns (gpu_ids, mig_ids, cpu_alloc) or None if unavailable.
        """
        rt = getattr(job, "resource_type", ResourceType.GPU)

        if rt == ResourceType.CPU:
            cpu = self._find_cpus(job)
            return ([], [], cpu) if cpu else None

        if rt == ResourceType.MIG:
            mig = self._find_mig(job)
            return ([], mig, []) if mig else None

        if rt == ResourceType.CPU_GPU:
            gpus = self._find_gpus(job)
            cpus = self._find_cpus(job)
            if gpus and cpus:
                return (gpus, [], cpus)
            return None

        # Default: GPU
        gpus = self._find_gpus(job)
        return (gpus, [], []) if gpus else None

    @abstractmethod
    def schedule(
        self,
        pending: list[AnyJob],
        running: list[AnyJob],
        current_time: float,
    ) -> SchedulingDecision:
        ...


SCHEDULER_REGISTRY: dict[str, type] = {}

def register(cls: type) -> type:
    SCHEDULER_REGISTRY[cls.name.lower()] = cls
    return cls


# ─────────────────────────────────────────────────────────────────────────────
# 1. FIFO
# ─────────────────────────────────────────────────────────────────────────────

@register
class FIFOScheduler(BaseScheduler):
    """First In First Out — baseline for JCT comparisons."""
    name = "fifo"

    def schedule(self, pending, running, current_time):
        decision = SchedulingDecision()
        ordered = sorted(pending, key=lambda j: j.submit_time)
        for job in ordered:
            res = self._find_resources(job)
            if res:
                gpus, migs, cpus = res
                decision.add(job, gpus, migs, cpus)
        return decision


# ─────────────────────────────────────────────────────────────────────────────
# 2. SJF
# ─────────────────────────────────────────────────────────────────────────────

@register
class SJFScheduler(BaseScheduler):
    """
    Shortest Job First (preemptive SJF = SRPT).
    Optimal for avg JCT when job durations are known.
    """
    name = "sjf"

    def _est_duration(self, job: AnyJob) -> float:
        if isinstance(job, TrainingJob):
            tp = solo_throughput(job.arch, GPUType.V100, job.batch_size)
            return job.remaining_iterations() / max(tp, 1e-9)
        if isinstance(job, LLMJob):
            p = MODEL_PROFILES.get(job.arch)
            tp = p.base_throughput_k80 * 2.0 if p else 0.5
            return job.remaining_iterations() / max(tp, 1e-9)
        return getattr(job, "total_queries", 1000)

    def schedule(self, pending, running, current_time):
        decision = SchedulingDecision()
        ordered = sorted(pending, key=self._est_duration)
        for job in ordered:
            res = self._find_resources(job)
            if res:
                gpus, migs, cpus = res
                decision.add(job, gpus, migs, cpus)
        return decision


# ─────────────────────────────────────────────────────────────────────────────
# 3. LAS / Tiresias
# ─────────────────────────────────────────────────────────────────────────────

# MLFQ thresholds in GPU-seconds (Tiresias paper Table 3)
_TIRESIAS_THRESHOLDS = [60, 180, 600, 1800, float("inf")]

@register
class TiresiasLASScheduler(BaseScheduler):
    """
    Least Attained Service (LAS) via MLFQ discretization.
    Tiresias [Gu et al., NSDI'19]: prioritize jobs with less GPU-seconds.
    """
    name = "tiresias"

    def _queue_level(self, attained: float) -> int:
        for i, thr in enumerate(_TIRESIAS_THRESHOLDS):
            if attained <= thr:
                return i
        return len(_TIRESIAS_THRESHOLDS) - 1

    def schedule(self, pending, running, current_time):
        decision = SchedulingDecision()
        for job in pending:
            svc = getattr(job, "attained_service", 0.0)
            job._las_level = self._queue_level(svc)

        ordered = sorted(pending, key=lambda j: (
            getattr(j, "_las_level", 0),
            j.submit_time,
        ))
        for job in ordered:
            res = self._find_resources(job)
            if res:
                gpus, migs, cpus = res
                decision.add(job, gpus, migs, cpus)
        return decision


# ─────────────────────────────────────────────────────────────────────────────
# 4. E-LAS
# ─────────────────────────────────────────────────────────────────────────────

@register
class ELASScheduler(BaseScheduler):
    """
    Epoch-progress-rate LAS. [Sultana et al., ICPP'20]
    Priority ∝ remaining_work / attained_service.
    """
    name = "e-las"

    def _urgency(self, job: AnyJob) -> float:
        attained = getattr(job, "attained_service", 1.0) or 1.0
        remaining = (
            job.remaining_iterations() if hasattr(job, "remaining_iterations")
            else getattr(job, "total_queries", 1000)
        )
        return remaining / attained

    def schedule(self, pending, running, current_time):
        decision = SchedulingDecision()
        for job in sorted(pending, key=self._urgency, reverse=True):
            res = self._find_resources(job)
            if res:
                gpus, migs, cpus = res
                decision.add(job, gpus, migs, cpus)
        return decision


# ─────────────────────────────────────────────────────────────────────────────
# 5. MLFQ
# ─────────────────────────────────────────────────────────────────────────────

@register
class MLFQScheduler(BaseScheduler):
    """
    Multi-Level Feedback Queue — standard HPC job scheduler.
    Priority decays with accumulated runtime.
    """
    name = "mlfq"

    LEVELS = 4
    QUANTA = [300, 900, 2700, float("inf")]  # seconds

    def _level(self, job: AnyJob) -> int:
        svc = getattr(job, "accumulated_work", 0.0)
        for i, q in enumerate(self.QUANTA):
            if svc <= q:
                return i
        return self.LEVELS - 1

    def schedule(self, pending, running, current_time):
        decision = SchedulingDecision()
        ordered = sorted(pending, key=lambda j: (self._level(j), j.submit_time))
        for job in ordered:
            res = self._find_resources(job)
            if res:
                gpus, migs, cpus = res
                decision.add(job, gpus, migs, cpus)
        return decision


# ─────────────────────────────────────────────────────────────────────────────
# 6. Gavel
# ─────────────────────────────────────────────────────────────────────────────

@register
class GavelScheduler(BaseScheduler):
    """
    Gavel: Heterogeneity-aware round-based scheduling. [Narayanan et al., OSDI'20]

    Algorithm:
    1. Compute throughput of each job on each GPU type.
    2. Assign jobs to GPU types to maximise total normalised throughput.
    3. Allocate in round-robin across types (fairness).
    """
    name = "gavel"

    def _best_gpu_type(self, job: AnyJob) -> GPUType:
        arch = getattr(job, "arch", ModelArch.RESNET50)
        bs   = getattr(job, "batch_size", 64)
        best_tp, best_type = 0.0, GPUType.V100
        free = self.cluster.free_gpus_by_type()
        for gtype, gpus in free.items():
            if not gpus:
                continue
            tp = solo_throughput(arch, gtype, bs)
            if tp > best_tp:
                best_tp, best_type = tp, gtype
        return best_type

    def schedule(self, pending, running, current_time):
        decision = SchedulingDecision()
        # Sort by Gavel priority: normalised throughput deficit
        ordered = sorted(
            pending,
            key=lambda j: -solo_throughput(
                getattr(j, "arch", ModelArch.RESNET50),
                GPUType.V100,
                getattr(j, "batch_size", 64),
            )
        )
        for job in ordered:
            best_type = self._best_gpu_type(job)
            gpu_ids = self._find_gpus(job, gpu_type=best_type)
            if gpu_ids:
                decision.add(job, gpu_ids)
            else:
                res = self._find_resources(job)
                if res:
                    gpus, migs, cpus = res
                    decision.add(job, gpus, migs, cpus)
        return decision


# ─────────────────────────────────────────────────────────────────────────────
# 7. Pollux
# ─────────────────────────────────────────────────────────────────────────────

@register
class PolluxScheduler(BaseScheduler):
    """
    Pollux: Goodput-optimised elastic training. [Qiao et al., OSDI'21]

    Joint optimisation of (k, bs, lr) to maximise goodput.
    Elastic jobs may be re-allocated with different GPU counts.
    """
    name = "pollux"

    def _best_k_and_gpus(self, job: AnyJob) -> tuple[int, Optional[list[str]]]:
        arch = getattr(job, "arch", ModelArch.RESNET50)
        min_k = getattr(job, "min_gpus", 1)
        max_k = getattr(job, "max_gpus", 8)
        free_all = self.cluster.find_scattered_gpus(1)  # check any availability
        if not free_all:
            return 1, None

        best_gp, best_k, best_ids = 0.0, min_k, None
        for k in [min_k, min_k*2, max_k]:
            k = max(min_k, min(k, max_k))
            gpu_ids = self.cluster.find_best_placement(k)
            if not gpu_ids:
                continue
            bw = self.cluster.effective_bandwidth_gbps(gpu_ids)
            gp = goodput(
                arch, self.cluster.gpus[gpu_ids[0]].gpu_type,
                getattr(job, "batch_size", 64), k, bw,
            )
            if gp > best_gp:
                best_gp, best_k, best_ids = gp, k, gpu_ids
        return best_k, best_ids

    def schedule(self, pending, running, current_time):
        decision = SchedulingDecision()
        elastic = [j for j in pending if getattr(j, "scheduling_mode", None) == SchedulingMode.ELASTIC]
        rest    = [j for j in pending if j not in elastic]

        # Pollux elastic jobs get priority with optimal k
        for job in sorted(elastic, key=lambda j: j.submit_time):
            _, gpu_ids = self._best_k_and_gpus(job)
            if gpu_ids:
                decision.add(job, gpu_ids, [], [])

        # Remaining: FIFO
        for job in sorted(rest, key=lambda j: j.submit_time):
            res = self._find_resources(job)
            if res:
                gpus, migs, cpus = res
                decision.add(job, gpus, migs, cpus)
        return decision


# ─────────────────────────────────────────────────────────────────────────────
# 8. Themis
# ─────────────────────────────────────────────────────────────────────────────

@register
class ThemisScheduler(BaseScheduler):
    """
    Themis: Finish-time fairness (ρ-fairness). [Mahajan et al., NSDI'20]

    Allocates to minimise max(actual_finish_time / solo_finish_time) across users.
    """
    name = "themis"

    def _solo_finish_time(self, job: AnyJob, current_time: float) -> float:
        arch = getattr(job, "arch", ModelArch.RESNET50)
        tp = solo_throughput(arch, GPUType.V100, getattr(job, "batch_size", 64))
        if hasattr(job, "remaining_iterations"):
            remaining = job.remaining_iterations()
        else:
            remaining = getattr(job, "total_queries", 1000)
        return (remaining / max(tp, 1e-9)) + current_time

    def schedule(self, pending, running, current_time):
        decision = SchedulingDecision()

        # Group by user; compute ρ ratio: (elapsed) / solo_time
        user_rho: dict[str, float] = {}
        for job in running:
            uid = job.user_id
            ft  = self._solo_finish_time(job, current_time)
            rho = (current_time - job.submit_time) / max(ft - job.submit_time, 1.0)
            user_rho[uid] = min(user_rho.get(uid, float("inf")), rho)

        # Prioritise users with smallest ρ (most behind their fair share)
        def key(job: AnyJob) -> float:
            return user_rho.get(job.user_id, 0.0)

        for job in sorted(pending, key=key):
            res = self._find_resources(job)
            if res:
                gpus, migs, cpus = res
                decision.add(job, gpus, migs, cpus)
        return decision


# ─────────────────────────────────────────────────────────────────────────────
# 9. Chronus
# ─────────────────────────────────────────────────────────────────────────────

@register
class ChronusScheduler(BaseScheduler):
    """
    Chronus: Deadline-aware SLO + Best-Effort separation. [Gao et al., SoCC'21]

    SLO jobs scheduled first by tightest deadline;
    remaining capacity fills with BE jobs (FIFO).
    """
    name = "chronus"

    def _slack(self, job: AnyJob, current_time: float) -> float:
        deadline = getattr(job, "deadline", None)
        if deadline is None:
            return float("inf")
        if hasattr(job, "remaining_iterations"):
            arch = job.arch
            tp = solo_throughput(arch, GPUType.V100, job.batch_size)
            est = job.remaining_iterations() / max(tp, 1e-9)
        else:
            est = 0.0
        return deadline - current_time - est

    def schedule(self, pending, running, current_time):
        decision = SchedulingDecision()
        slo_jobs = [j for j in pending if getattr(j, "deadline", None) is not None]
        be_jobs  = [j for j in pending if getattr(j, "deadline", None) is None]

        # SLO: tightest slack first (Earliest Deadline First variant)
        for job in sorted(slo_jobs, key=lambda j: self._slack(j, current_time)):
            res = self._find_resources(job)
            if res:
                gpus, migs, cpus = res
                decision.add(job, gpus, migs, cpus)

        # BE: FIFO on remaining capacity
        for job in sorted(be_jobs, key=lambda j: j.submit_time):
            res = self._find_resources(job)
            if res:
                gpus, migs, cpus = res
                decision.add(job, gpus, migs, cpus)
        return decision


# ─────────────────────────────────────────────────────────────────────────────
# 10. ElasticFlow
# ─────────────────────────────────────────────────────────────────────────────

@register
class ElasticFlowScheduler(BaseScheduler):
    """
    ElasticFlow: Elastic serverless DNN with minimum satisfactory share.
    [Gu et al., ASPLOS'23]

    Assigns each job the minimum GPU count to meet its deadline;
    gives surplus GPUs to maximise throughput.
    """
    name = "elasticflow"

    def _min_gpus_for_deadline(
        self, job: AnyJob, current_time: float
    ) -> int:
        deadline = getattr(job, "deadline", None)
        if deadline is None:
            return getattr(job, "min_gpus", 1)
        slack = max(1.0, deadline - current_time)
        arch = getattr(job, "arch", ModelArch.RESNET50)
        bs   = getattr(job, "batch_size", 64)
        if hasattr(job, "remaining_iterations"):
            remaining = job.remaining_iterations()
        else:
            remaining = 1000

        # Find smallest k such that throughput * slack >= remaining
        for k in range(1, 65):
            tp = solo_throughput(arch, GPUType.V100, bs) * k * 0.85
            if tp * slack >= remaining:
                return k
        return getattr(job, "min_gpus", 1)

    def schedule(self, pending, running, current_time):
        decision = SchedulingDecision()
        ordered = sorted(pending, key=lambda j: (
            getattr(j, "deadline", float("inf")),
            j.submit_time,
        ))
        for job in ordered:
            min_k = self._min_gpus_for_deadline(job, current_time)
            old_req = getattr(job, "num_gpus_requested", min_k)
            object.__setattr__(job, "num_gpus_requested", max(min_k, old_req)) if hasattr(job, "__dataclass_fields__") else None
            try:
                job.num_gpus_requested = max(min_k, old_req)
            except AttributeError:
                pass
            res = self._find_resources(job)
            if not res and min_k > 1:
                try:
                    job.num_gpus_requested = min_k
                except AttributeError:
                    pass
                res = self._find_resources(job)
            if res:
                gpus, migs, cpus = res
                decision.add(job, gpus, migs, cpus)
        return decision


# ─────────────────────────────────────────────────────────────────────────────
# 11. MaxMinFairness (DRF)
# ─────────────────────────────────────────────────────────────────────────────

@register
class MaxMinFairnessScheduler(BaseScheduler):
    """
    Dominant Resource Fairness. [Ghodsi et al., NSDI'11]
    Maximises the minimum dominant resource share across users.
    """
    name = "maxminfairness"

    def _dominant_share(self, user_id: str, running: list[AnyJob]) -> float:
        total_gpus = self.cluster.total_gpus()
        if total_gpus == 0:
            return 0.0
        used = sum(
            len(j.allocated_gpus)
            for j in running
            if j.user_id == user_id
        )
        return used / total_gpus

    def schedule(self, pending, running, current_time):
        decision = SchedulingDecision()
        # Give priority to user with least dominant share
        for job in sorted(pending, key=lambda j: (
            self._dominant_share(j.user_id, running),
            j.submit_time,
        )):
            res = self._find_resources(job)
            if res:
                gpus, migs, cpus = res
                decision.add(job, gpus, migs, cpus)
        return decision


# ─────────────────────────────────────────────────────────────────────────────
# 12. Backfill — deprecated stub (use BackfillWrapper from scheduler.backfill)
# ─────────────────────────────────────────────────────────────────────────────

@register
class BackfillScheduler(BaseScheduler):
    """
    DEPRECATED — kept for backward compatibility only.

    Use BackfillWrapper from hpcsim.scheduler.backfill instead:

        from hpcsim.scheduler.backfill import BackfillWrapper, EASYBackfillPolicy
        sched = BackfillWrapper(FIFOScheduler(cluster), EASYBackfillPolicy())

    This stub now behaves as FIFO + EASY-Backfilling via the wrapper.
    """
    name = "backfill"

    def schedule(self, pending, running, current_time):
        import warnings
        warnings.warn(
            "BackfillScheduler is deprecated. "
            "Use BackfillWrapper(primary, EASYBackfillPolicy()) instead.",
            DeprecationWarning, stacklevel=2,
        )
        decision = SchedulingDecision()
        if not pending:
            return decision
        ordered = sorted(pending, key=lambda j: j.submit_time)
        # Head job
        head = ordered[0]
        head_res = self._find_resources(head)
        if head_res:
            gpus, migs, cpus = head_res
            decision.add(head, gpus, migs, cpus)
        # Simple EASY fill: smaller jobs that fit
        free_gpus = self.cluster.free_gpu_count()
        for job in ordered[1:]:
            n_req = getattr(job, "num_gpus_requested", 1)
            if n_req > free_gpus:
                continue
            res = self._find_resources(job)
            if res:
                gpus, migs, cpus = res
                decision.add(job, gpus, migs, cpus)
                free_gpus -= n_req
        return decision



        ordered = sorted(pending, key=lambda j: j.submit_time)

        # Try to schedule head-of-queue job
        head = ordered[0]
        head_res = self._find_resources(head)
        if head_res:
            _hg, _hm, _hc = head_res
            decision.add(head, _hg, _hm, _hc)
            head_gpus = _hg  # for backfill window calc
            ordered = ordered[1:]

        # Backfill: schedule smaller jobs that fit without delaying head
        for job in ordered:
            if job is head:
                continue
            n_req = getattr(job, "num_gpus_requested", 1)
            # heuristic: only backfill if job needs <= half head's GPUs
            head_req = getattr(head, "num_gpus_requested", 1)
            if n_req > head_req:
                continue
            res = self._find_resources(job)
            if res:
                gpus, migs, cpus = res
                decision.add(job, gpus, migs, cpus)
        return decision


# ─────────────────────────────────────────────────────────────────────────────
# Factory
# ─────────────────────────────────────────────────────────────────────────────

def create_scheduler(name: str, cluster: "Cluster") -> BaseScheduler:
    """Instantiate a scheduler by name (case-insensitive)."""
    key = name.lower().replace("-", "").replace("_", "")
    # Alias normalisation
    aliases = {
        "las": "tiresias", "elas": "e-las", "las2": "e-las",
        "drf": "maxminfairness", "fair": "maxminfairness",
        "fcfs": "fifo", "easy": "backfill",
        # RL scheduler aliases
        "ppo": "maskableppo", "maskppo": "maskableppo",
        "gasmarl": "gasmarl", "marl": "gasmarl",
    }
    key = aliases.get(key, key)
    factory = SCHEDULER_REGISTRY.get(key)
    if factory is None:
        available = list(SCHEDULER_REGISTRY.keys())
        raise ValueError(f"Unknown scheduler '{name}'. Available: {available}")
    # Factory can be a class (callable with cluster) or a plain function
    return factory(cluster)


# Expose registry under both names for compatibility
_REGISTRY = SCHEDULER_REGISTRY


def list_schedulers() -> list[str]:
    return sorted(SCHEDULER_REGISTRY.keys())


def register_factory(name: str, factory_fn):
    """Register a callable factory for dynamic/RL schedulers.

    factory_fn(cluster: Cluster) -> BaseScheduler

    Example:
        from hpcsim.scheduler.schedulers import register_factory
        register_factory("my_ppo", lambda cl: MyPPOScheduler(cl, model_dir="models/"))
    """
    SCHEDULER_REGISTRY[name.lower()] = factory_fn
