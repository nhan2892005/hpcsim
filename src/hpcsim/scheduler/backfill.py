"""
Backfill Policies for HPC Scheduling.

Backfilling is a *policy* applied on top of any primary scheduler, not a
scheduler itself.  This module provides:

  - BackfillPolicy   — abstract base class
  - EASYBackfillPolicy  — standard EASY / Conservative Backfilling (HPC default)
  - GreenBackfillPolicy — Green-Backfilling from GAS-MARL (Chen et al., FGCS 2025)
  - BackfillWrapper  — wraps any BaseScheduler with a BackfillPolicy

Architecture (two-layer):
  ┌─────────────────────────────────────────────────┐
  │           Primary Scheduler                      │
  │  (FIFO / Gavel / GAS-MARL / PPO / ...)          │
  │  → picks job + optional delay decision           │
  └────────────────────┬────────────────────────────┘
                       │  blocked or delay
                       ▼
  ┌─────────────────────────────────────────────────┐
  │           BackfillPolicy                         │
  │  EASY: fill gaps in submit-time order            │
  │  Green: fill gaps, prefer low-brown-energy jobs  │
  └─────────────────────────────────────────────────┘

Usage::

    from hpcsim.scheduler.backfill import BackfillWrapper, EASYBackfillPolicy, GreenBackfillPolicy
    from hpcsim.energy.renewable import RenewableEnergyModule

    # FIFO + EASY  (classic HPC production setup)
    sched = BackfillWrapper(FIFOScheduler(cluster), EASYBackfillPolicy())

    # Gavel + Green-Backfilling  (green-aware)
    re    = RenewableEnergyModule(total_gpus=cluster.total_gpus(), sim_duration=86400)
    sched = BackfillWrapper(GavelScheduler(cluster), GreenBackfillPolicy(re))

    # GAS-MARL + Green-Backfilling  (paper's best combination)
    sched = BackfillWrapper(GASMARLScheduler(cluster), GreenBackfillPolicy(re))

Reference: Chen et al., "GAS-MARL: Green-Aware job Scheduling algorithm for HPC
clusters based on Multi-Action Deep Reinforcement Learning", FGCS 2025, §4.3.
"""

from __future__ import annotations
from abc import ABC, abstractmethod
from typing import Optional, TYPE_CHECKING

if TYPE_CHECKING:
    from ..cluster.cluster import Cluster
    from ..energy.renewable import RenewableEnergyModule

from ..workload.job import AnyJob, ResourceType
from .schedulers import BaseScheduler, SchedulingDecision


# ─── Brown-energy threshold (Algorithm 2, GAS-MARL paper §4.3) ───────────────
BROWN_THRESHOLD_J: float = 50_000.0   # σ = 50,000 J


# ─────────────────────────────────────────────────────────────────────────────
# Abstract BackfillPolicy
# ─────────────────────────────────────────────────────────────────────────────

class BackfillPolicy(ABC):
    """
    Abstract backfill policy.

    ``select_backfill_jobs`` receives the set of candidate jobs (pending, minus
    the blocked head job) and returns the subset that may start early without
    violating the backfill window constraint.
    """

    @abstractmethod
    def select_backfill_jobs(
        self,
        candidates:      list[AnyJob],   # pending jobs (excluding head)
        head_job:        AnyJob,          # job blocked at head-of-queue
        running:         list[AnyJob],    # currently running jobs
        cluster:         "Cluster",
        current_time:    float,
        max_finish_time: float,           # shadow time — backfill window limit
    ) -> list[AnyJob]:
        """Return jobs permitted to backfill (in preferred execution order)."""
        ...


# ─────────────────────────────────────────────────────────────────────────────
# EASY-Backfilling
# ─────────────────────────────────────────────────────────────────────────────

class EASYBackfillPolicy(BackfillPolicy):
    """
    Standard EASY / Conservative Backfilling.

    A job is allowed to backfill if:
      1. It fits in currently available resources.
      2. Its estimated finish time does not exceed the shadow time
         (earliest time the blocked head job can start).

    Jobs are considered in *submit-time* (FCFS) order.
    """

    def select_backfill_jobs(
        self,
        candidates:      list[AnyJob],
        head_job:        AnyJob,
        running:         list[AnyJob],
        cluster:         "Cluster",
        current_time:    float,
        max_finish_time: float,
    ) -> list[AnyJob]:
        result: list[AnyJob] = []
        free_gpus = cluster.free_gpu_count()
        free_cpus = cluster.free_cpu_cores()
        free_migs = cluster.free_mig_slices()

        # Sort candidates by submit time (EASY ordering)
        for job in sorted(candidates, key=lambda j: j.submit_time):
            rt       = getattr(job, "resource_type", ResourceType.GPU)
            req_gpu  = getattr(job, "num_gpus_requested", 0)
            req_cpu  = getattr(job, "num_cpus_requested", 0)
            req_mig  = getattr(job, "num_mig_requested", 0)
            runtime  = getattr(job, "requested_runtime", None) or \
                       getattr(job, "base_duration_sec", 3600.0)
            est_finish = current_time + runtime

            # Check window constraint
            if est_finish > max_finish_time:
                continue

            # Check resource availability (optimistic — actual allocation done by wrapper)
            fits = False
            if rt == ResourceType.CPU:
                fits = (req_cpu > 0) and (free_cpus >= req_cpu)
            elif rt == ResourceType.MIG:
                fits = (req_mig > 0) and (free_migs >= req_mig)
            elif rt == ResourceType.CPU_GPU:
                fits = (free_gpus >= req_gpu) and (free_cpus >= req_cpu)
            else:  # GPU (default)
                fits = (req_gpu > 0) and (free_gpus >= req_gpu)

            if fits:
                result.append(job)
                # Update optimistic free counts
                free_gpus -= req_gpu
                free_cpus -= req_cpu
                free_migs -= req_mig

        return result


# ─────────────────────────────────────────────────────────────────────────────
# Green-Backfilling
# ─────────────────────────────────────────────────────────────────────────────

class GreenBackfillPolicy(BackfillPolicy):
    """
    Green-Backfilling (GAS-MARL, Algorithm 2).

    Extends EASY-Backfilling with two modifications:

    1. **Priority function** (Eq. 15):
       L_j = re_j × q_j × p_j
       where re = requested runtime, q = processors, p = power_watts.
       Jobs are sorted ascending by L_j (smaller = higher priority).
       This favours jobs that are short, use few GPUs, and consume little power.

    2. **Brown-energy acceptance criterion** (Algorithm 2, Line 7):
       Estimated brown energy consumed by the job < σ (default 50,000 J).
       If a job would consume too much brown energy, it is rejected.
    """

    def __init__(
        self,
        renewable: "RenewableEnergyModule",
        brown_threshold_j: float = BROWN_THRESHOLD_J,
    ):
        self.re = renewable
        self.sigma = brown_threshold_j

    # ── priority ─────────────────────────────────────────────────────────────

    def _priority(self, job: AnyJob) -> float:
        """L_j = re_j × q_j × p_j  (Eq. 15) — ascending sort."""
        re_j  = getattr(job, "requested_runtime", None) or \
                getattr(job, "base_duration_sec", 3600.0)
        q_j   = getattr(job, "num_gpus_requested", 1) or \
                getattr(job, "num_cpus_requested", 1) or 1
        # Prefer jobs with explicit power; fall back to 300W × GPU count
        p_j   = getattr(job, "power_watts", None) or (300.0 * max(q_j, 1))
        return re_j * q_j * p_j

    # ── brown energy estimate ─────────────────────────────────────────────────

    def _brown_energy_j(self, job: AnyJob, current_time: float) -> float:
        """
        Estimated brown (non-renewable) energy if job starts now (Joules).
        E_B = max(0, P_job - P_green_available) × runtime
        """
        power_w  = getattr(job, "power_watts", None) or \
                   (300.0 * max(getattr(job, "num_gpus_requested", 1), 1))
        runtime  = getattr(job, "requested_runtime", None) or \
                   getattr(job, "base_duration_sec", 3600.0)
        re_avail = self.re.available_power_watts(current_time)
        brown_w  = max(0.0, power_w - re_avail)
        return brown_w * runtime

    # ── select ───────────────────────────────────────────────────────────────

    def select_backfill_jobs(
        self,
        candidates:      list[AnyJob],
        head_job:        AnyJob,
        running:         list[AnyJob],
        cluster:         "Cluster",
        current_time:    float,
        max_finish_time: float,
    ) -> list[AnyJob]:
        result: list[AnyJob] = []
        free_gpus = cluster.free_gpu_count()
        free_cpus = cluster.free_cpu_cores()
        free_migs = cluster.free_mig_slices()

        # Sort by ascending L_j (smaller resource × power → higher priority)
        for job in sorted(candidates, key=self._priority):
            rt       = getattr(job, "resource_type", ResourceType.GPU)
            req_gpu  = getattr(job, "num_gpus_requested", 0)
            req_cpu  = getattr(job, "num_cpus_requested", 0)
            req_mig  = getattr(job, "num_mig_requested", 0)
            runtime  = getattr(job, "requested_runtime", None) or \
                       getattr(job, "base_duration_sec", 3600.0)
            est_finish = current_time + runtime

            # Algorithm 2, Line 7:  pe_j + ts_c < ts_LM  AND  E_B < σ
            if est_finish > max_finish_time:
                continue
            if self._brown_energy_j(job, current_time) >= self.sigma:
                continue

            # Resource availability (optimistic)
            fits = False
            if rt == ResourceType.CPU:
                fits = (req_cpu > 0) and (free_cpus >= req_cpu)
            elif rt == ResourceType.MIG:
                fits = (req_mig > 0) and (free_migs >= req_mig)
            elif rt == ResourceType.CPU_GPU:
                fits = (free_gpus >= req_gpu) and (free_cpus >= req_cpu)
            else:
                fits = (req_gpu > 0) and (free_gpus >= req_gpu)

            if fits:
                result.append(job)
                free_gpus -= req_gpu
                free_cpus -= req_cpu
                free_migs -= req_mig

        return result


# ─────────────────────────────────────────────────────────────────────────────
# BackfillWrapper
# ─────────────────────────────────────────────────────────────────────────────

class BackfillWrapper(BaseScheduler):
    """
    Wraps any primary scheduler with a backfill policy.

    Scheduling flow
    ───────────────
    1. Call ``primary.schedule()`` → SchedulingDecision.
    2. Identify the "head" job: highest-priority pending job that was NOT
       scheduled (blocked because of insufficient resources).
    3. Compute *shadow time*: earliest timestamp the head job can start.
    4. Run ``policy.select_backfill_jobs()`` on remaining candidates.
    5. For each selected backfill job, call ``_find_resources()`` and add to
       the decision if resources are actually available.

    GAS-MARL special case
    ──────────────────────
    If the primary scheduler attaches ``decision.delay_info`` (set by
    GASMARLScheduler when it issues a delay action), the wrapper uses that
    metadata to compute the backfill window as per paper §4.3:

    - delay type 2 (wait N jobs complete):
        max_finish = max(ts_RN, ts_ES)  capped at current_time + 3600
    - delay type 3 (wait D seconds):
        max_finish = max(ts_ES, current_time + DT)
    """

    name = "backfill_wrapper"

    def __init__(
        self,
        primary: BaseScheduler,
        policy:  BackfillPolicy,
    ):
        # BackfillWrapper delegates resource helpers to the primary's cluster
        super().__init__(primary.cluster)
        self.primary = primary
        self.policy  = policy

    # ── main schedule ─────────────────────────────────────────────────────────

    def schedule(
        self,
        pending:      list[AnyJob],
        running:      list[AnyJob],
        current_time: float,
    ) -> SchedulingDecision:
        # 1. Primary decision
        decision = self.primary.schedule(pending, running, current_time)

        # 2. Jobs the primary already scheduled
        scheduled_ids = {a.job.job_id for a in decision.allocations}

        # 3. Remaining pending jobs
        remaining = [j for j in pending if j.job_id not in scheduled_ids]
        if not remaining:
            return decision

        # 4. Head job: highest-priority unscheduled job (by submit time)
        head = min(remaining, key=lambda j: j.submit_time)

        # 5. Compute max_finish_time (backfill window)
        delay_info = getattr(decision, "delay_info", None)
        if delay_info:
            max_finish = self._window_from_delay_info(delay_info, running, current_time)
        else:
            # Standard EASY: shadow time = when head can next start
            head_req   = getattr(head, "num_gpus_requested", 1)
            max_finish = self._shadow_time(head_req, running, current_time)

        # 6. Backfill candidates = remaining minus head itself
        candidates = [j for j in remaining if j.job_id != head.job_id]
        backfill_jobs = self.policy.select_backfill_jobs(
            candidates, head, running, self.cluster, current_time, max_finish
        )

        # 7. Actually allocate resources and extend decision
        for job in backfill_jobs:
            res = self._find_resources(job)
            if res:
                gpus, migs, cpus = res
                decision.add(job, gpus, migs, cpus)

        return decision

    # ── shadow time ───────────────────────────────────────────────────────────

    def _shadow_time(
        self,
        n_gpus_needed: int,
        running:       list[AnyJob],
        current_time:  float,
    ) -> float:
        """
        Earliest time at which n_gpus_needed GPUs become free.
        (Optimistic estimate based on requested runtimes.)
        """
        free = self.cluster.free_gpu_count()
        if free >= n_gpus_needed:
            return current_time  # head can start now

        # Sort running by estimated completion time
        def _est_done(j: AnyJob) -> float:
            start = getattr(j, "start_time", None) or current_time
            rt    = getattr(j, "requested_runtime", None) or \
                    getattr(j, "base_duration_sec", 3600.0)
            return start + rt

        sorted_run = sorted(running, key=_est_done)
        needed = n_gpus_needed - free
        for j in sorted_run:
            needed -= len(getattr(j, "allocated_gpus", [])) or \
                      getattr(j, "num_gpus_requested", 1)
            if needed <= 0:
                return _est_done(j)
        return current_time + 7200.0  # conservative fallback

    # ── GAS-MARL delay window (paper §4.3) ───────────────────────────────────

    def _window_from_delay_info(
        self,
        delay_info:   dict,
        running:      list[AnyJob],
        current_time: float,
    ) -> float:
        """
        Compute max_finish_time from GAS-MARL delay metadata.

        delay_info keys:
          delay_type   : int  (0 = no delay, 1-5 = wait N jobs, 6-12 = wait DT s)
          release_time : float
          head_job_id  : str  (job being delayed)
        """
        delay_type   = delay_info.get("delay_type", 0)
        release_time = delay_info.get("release_time", current_time)
        head_job_id  = delay_info.get("head_job_id", None)

        if delay_type == 0:
            # No delay — use standard shadow time
            head_req = delay_info.get("head_req_gpus", 1)
            return self._shadow_time(head_req, running, current_time)

        # ts_ES = earliest time head job CAN run (resource-wise)
        head_req   = delay_info.get("head_req_gpus", 1)
        ts_es      = self._shadow_time(head_req, running, current_time)

        # ts_LM = max(ts_release, ts_ES)  [paper Algorithm 2]
        ts_lm = max(release_time, ts_es)

        # Paper caps type-2 at current + 3600
        if delay_type <= 5:  # type 2: wait N jobs complete
            ts_lm = min(ts_lm, current_time + 3600.0)

        return ts_lm

    # ── delegate attribute access to primary ──────────────────────────────────

    def __getattr__(self, item):
        # Allow transparent attribute access on the primary scheduler
        # (e.g. _delayed_until for GASMARLScheduler)
        if item.startswith("_"):
            raise AttributeError(item)
        return getattr(self.primary, item)

    def __repr__(self) -> str:
        return (f"BackfillWrapper(primary={self.primary.__class__.__name__}, "
                f"policy={self.policy.__class__.__name__})")


# ─────────────────────────────────────────────────────────────────────────────
# Factory helpers
# ─────────────────────────────────────────────────────────────────────────────

def create_backfill_policy(
    name: str,
    renewable: Optional["RenewableEnergyModule"] = None,
    brown_threshold_j: float = BROWN_THRESHOLD_J,
) -> BackfillPolicy:
    """
    Instantiate a BackfillPolicy by name.

    Args:
        name:             "none" | "easy" | "green"
        renewable:        RenewableEnergyModule (required for "green")
        brown_threshold_j: σ threshold for green backfill (default 50,000 J)

    Returns:
        BackfillPolicy instance, or ``None`` if name is "none".
    """
    key = name.lower().strip()
    if key in ("none", "off", "disabled", ""):
        return None
    if key in ("easy", "conservative", "standard"):
        return EASYBackfillPolicy()
    if key in ("green", "green_backfill", "green-backfill"):
        if renewable is None:
            raise ValueError(
                "GreenBackfillPolicy requires a RenewableEnergyModule. "
                "Pass renewable= argument or disable green backfilling."
            )
        return GreenBackfillPolicy(renewable, brown_threshold_j)
    raise ValueError(
        f"Unknown backfill policy '{name}'. "
        f"Available: none, easy, green"
    )


def wrap_with_backfill(
    scheduler:    BaseScheduler,
    backfill:     str,
    renewable:    Optional["RenewableEnergyModule"] = None,
) -> BaseScheduler:
    """
    Optionally wrap a scheduler with a backfill policy.

    If backfill is "none" (default), returns the scheduler unchanged.
    Otherwise wraps it with BackfillWrapper + the chosen policy.

    Example::

        sched = create_scheduler("gavel", cluster)
        sched = wrap_with_backfill(sched, "green", renewable=re_module)
    """
    policy = create_backfill_policy(backfill, renewable=renewable)
    if policy is None:
        return scheduler
    return BackfillWrapper(scheduler, policy)
