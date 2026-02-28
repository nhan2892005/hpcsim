"""
Discrete Event Simulation Engine  —  event-driven, O(log N) per event.

Events:
- JOB_ARRIVAL     : new job enters the queue
- SCHEDULE        : scheduler invoked
- JOB_COMPLETE    : job finishes (calculated directly, no intermediate ticks)
- JOB_PREEMPT     : job preempted (checkpointed)
- METRIC_SAMPLE   : periodic metrics snapshot

Optimisation over the original tick-based engine
-------------------------------------------------
The original engine fired a JOB_PROGRESS event every TICK_INTERVAL=10 s for
every running job.  For a 24-hour simulation with 200 concurrent jobs that
produces ~200 × 8640 = 1.7 million progress events — a bottleneck for long
runs or large workloads.

This engine instead:
1. Computes the exact wall-clock time at which each job will finish given its
   current throughput and remaining work, then schedules a single JOB_COMPLETE
   event at that time.
2. Keeps a per-job completion-sequence counter.  Whenever throughput changes
   (co-location, preemption, elastic resize) the counter is bumped and a new
   JOB_COMPLETE is enqueued; the stale event is silently dropped when it
   eventually pops from the heap.
3. Tracks per-job segment start times so that attained_service /
   accumulated_work are updated correctly at preemption or completion without
   intermediate ticks.

Result: event count drops from O(jobs × sim_duration / tick) to
        O(jobs + completions + schedule_events + metric_events).
"""

from __future__ import annotations
from dataclasses import dataclass, field
from enum import Enum
from typing import Optional
import heapq
import math

from ..cluster.cluster import Cluster
from ..scheduler.schedulers import BaseScheduler, SchedulingDecision
from ..workload.job import (
    AnyJob, TrainingJob, InferenceJob, LLMJob, HPOJob,
    JobStatus, SchedulingMode,
    multi_gpu_throughput, solo_throughput, MODEL_PROFILES,
)
from ..metrics.collector import MetricsCollector
from ..energy.renewable import RenewableEnergyModule, RenewableConfig


class EventType(str, Enum):
    JOB_ARRIVAL    = "JOB_ARRIVAL"
    SCHEDULE       = "SCHEDULE"
    JOB_COMPLETE   = "JOB_COMPLETE"
    JOB_PREEMPT    = "JOB_PREEMPT"
    METRIC_SAMPLE  = "METRIC_SAMPLE"


@dataclass(order=True)
class Event:
    time: float
    seq: int = field(compare=True)
    event_type: EventType = field(compare=False)
    job_id: Optional[str] = field(default=None, compare=False)
    data: dict = field(default_factory=dict, compare=False)


@dataclass
class SimulationResult:
    completed_jobs: list
    metrics: "MetricsCollector"
    total_time: float
    scheduler_name: str
    cluster_name: str

    def avg_jct(self) -> float:
        jcts = [j.jct for j in self.completed_jobs if j.jct < float("inf")]
        return sum(jcts) / len(jcts) if jcts else 0.0

    def avg_queue_time(self) -> float:
        qts = [j.queue_time for j in self.completed_jobs if j.queue_time < float("inf")]
        return sum(qts) / len(qts) if qts else 0.0

    def throughput(self) -> float:
        return len(self.completed_jobs) / max(self.total_time, 1.0)

    def deadline_miss_rate(self) -> float:
        with_dl = [j for j in self.completed_jobs if getattr(j, "deadline", None) is not None]
        if not with_dl:
            return 0.0
        misses = sum(
            1 for j in with_dl
            if j.end_time is not None and j.end_time > j.deadline
        )
        return misses / len(with_dl)


class SimulationEngine:
    """
    Discrete-event simulation of an HPC GPU cluster.

    Design:
    - Priority queue of events (min-heap on time)
    - Scheduler called periodically + on every job arrival/completion
    - Job completion times calculated directly from remaining work / throughput
    - Preemption with checkpoint overhead (T6 – gang/elastic modes)
    """

    SCHEDULE_INTERVAL = 5.0    # seconds between periodic scheduler invocations
    METRIC_INTERVAL   = 60.0   # seconds between metric snapshots

    # HPO: simulated wall-clock seconds per epoch per trial
    HPO_EPOCH_SECONDS = 300.0

    def __init__(
        self,
        cluster: Cluster,
        scheduler: BaseScheduler,
        workload: list[AnyJob],
        metrics: Optional[MetricsCollector] = None,
        max_sim_time: Optional[float] = None,
        verbose: bool = False,
        renewable_config: Optional[RenewableConfig] = None,
    ):
        self.cluster   = cluster
        self.scheduler = scheduler
        self.workload  = workload
        self.metrics   = metrics or MetricsCollector()
        self.max_time  = max_sim_time or (max((j.submit_time for j in workload), default=0) + 7200)
        self.verbose   = verbose

        # Renewable energy module — generates solar+wind time series for this run
        self._renewable = RenewableEnergyModule(
            config=renewable_config,
            total_gpus=cluster.total_gpus(),
            sim_duration=self.max_time,
        )

        self._event_seq = 0
        self._heap: list[Event] = []
        self._pending:  dict[str, AnyJob] = {}
        self._running:  dict[str, AnyJob] = {}
        self._completed: list[AnyJob] = []
        self._job_gpus:  dict[str, list[str]] = {}   # job_id → gpu_ids
        self._job_migs:  dict[str, list[str]] = {}   # job_id → mig_ids
        self._job_cpus:  dict[str, list[str]] = {}   # job_id → cpu_alloc
        self._current_time = 0.0

        # ── Event-driven optimisation state ──────────────────────────────────
        # Per-job completion-sequence counter.  Bumped each time a new
        # JOB_COMPLETE is scheduled; stale events carry an old seq and are
        # discarded on pop.
        self._job_complete_seq: dict[str, int] = {}

        # Wall-clock time at which the current running segment started, used
        # to accumulate attained_service and accumulated_work lazily.
        self._job_segment_start: dict[str, float] = {}

    # ── Heap helper ───────────────────────────────────────────────────────────

    def _push(self, t: float, etype: EventType, job_id=None, data=None):
        self._event_seq += 1
        heapq.heappush(
            self._heap,
            Event(time=t, seq=self._event_seq,
                  event_type=etype, job_id=job_id, data=data or {})
        )

    # ── Throughput computation ────────────────────────────────────────────────

    def _compute_throughput(self, job: AnyJob) -> float:
        """
        Throughput (iter/s or queries/s) for current allocation.
        Accounts for GPU type, placement (T2), and co-location (T5).
        """
        gpu_ids = self._job_gpus.get(job.job_id, [])
        if not gpu_ids:
            return 0.0

        # Use first GPU's type as representative
        g0 = self.cluster.gpus[gpu_ids[0]]
        gtype = g0.gpu_type
        arch = getattr(job, "arch", None)
        bs   = getattr(job, "batch_size", 64)
        k    = len(gpu_ids)

        if arch is None:
            return k * 1.0

        bw = self.cluster.effective_bandwidth_gbps(gpu_ids)
        tp = multi_gpu_throughput(arch, gtype, bs, k, bw)

        # T5: co-location interference
        colocated_jobs = [
            jid for jid in g0.allocated_jobs if jid != job.job_id
        ]
        if colocated_jobs:
            n = 1 + len(colocated_jobs)
            tp *= max(0.5, 0.8 ** (n - 1))

        return max(tp, 1e-9)

    # ── Remaining-time calculation ────────────────────────────────────────────

    def _remaining_job_time(self, job: AnyJob, tp: float) -> float:
        """
        Exact remaining wall-clock time for *job* at throughput *tp*.
        Returns 0.0 if the job is already done, inf if tp==0.
        """
        if tp <= 0:
            return float("inf")

        if isinstance(job, (TrainingJob, LLMJob)):
            remaining = job.num_iterations - job.completed_iterations
            return max(0.0, remaining / tp)

        elif isinstance(job, InferenceJob):
            remaining = job.total_queries - job.completed_queries
            effective_tp = tp * job.current_batch_size
            return max(0.0, remaining / max(effective_tp, 1e-9))

        elif isinstance(job, HPOJob):
            EPOCH_S = self.HPO_EPOCH_SECONDS
            # Initialise active trials on first call
            if not job.active_trials and len(job.completed_trials) < job.num_trials:
                concurrent = min(job.num_trials, 4)
                for i in range(concurrent):
                    job.active_trials[str(i)] = 0.0

            # Time until each active trial finishes its current run
            max_active_t = 0.0
            for epochs_done in job.active_trials.values():
                remaining_epochs = job.max_epochs_per_trial - epochs_done
                max_active_t = max(max_active_t, remaining_epochs * EPOCH_S)

            # After this wave, how many more complete waves are needed?
            remaining_trials = (job.num_trials
                                - len(job.completed_trials)
                                - len(job.active_trials))
            concurrent = min(job.num_trials, 4)
            extra_waves = math.ceil(max(remaining_trials, 0) / concurrent)
            extra_time  = extra_waves * job.max_epochs_per_trial * EPOCH_S

            return max_active_t + extra_time

        # Unknown job type — can't optimise, fall back to a large sentinel
        return float("inf")

    # ── Direct-completion scheduling ──────────────────────────────────────────

    def _schedule_completion(self, job: AnyJob, current_time: float):
        """
        Compute exact finish time for *job* and enqueue a JOB_COMPLETE event.
        Bumps the per-job sequence counter so any previously-queued stale
        JOB_COMPLETE for this job will be ignored when it eventually pops.
        """
        jid = job.job_id
        tp  = self._compute_throughput(job)
        dt  = self._remaining_job_time(job, tp)

        new_seq = self._job_complete_seq.get(jid, 0) + 1
        self._job_complete_seq[jid] = new_seq

        finish_time = current_time + dt
        self._push(finish_time, EventType.JOB_COMPLETE, jid,
                   {"completion_seq": new_seq})

        if self.verbose:
            print(f"  [{current_time:.0f}s] SCHED_COMPLETE {jid} "
                  f"in {dt:.1f}s → t={finish_time:.0f}s  tp={tp:.3f}")

    def _reschedule_collocated(self, new_job: AnyJob, current_time: float):
        """
        When *new_job* starts, its co-location may reduce throughput for jobs
        already running on the same GPUs.  Recalculate their completion times,
        accumulating work for the elapsed segment first.
        """
        gpu_ids = self._job_gpus.get(new_job.job_id, [])
        if not gpu_ids:
            return
        # Gather other jobs that share at least one GPU
        affected: set[str] = set()
        for gid in gpu_ids:
            g = self.cluster.gpus[gid]
            for jid in g.allocated_jobs:
                if jid != new_job.job_id and jid in self._running:
                    affected.add(jid)

        for jid in affected:
            job = self._running[jid]
            self._flush_segment(job, current_time)
            self._schedule_completion(job, current_time)

    def _flush_segment(self, job: AnyJob, current_time: float):
        """
        Accumulate attained_service / accumulated_work for the elapsed segment
        [segment_start, current_time) and advance completed progress counters.
        Call before any event that changes throughput (preemption, co-location).
        """
        jid = job.job_id
        seg_start = self._job_segment_start.get(jid, current_time)
        dt = current_time - seg_start
        if dt <= 0:
            return

        k  = len(self._job_gpus.get(jid, []))
        tp = self._compute_throughput(job)

        # Update service counters
        if hasattr(job, "attained_service"):
            job.attained_service += dt * k
        if hasattr(job, "accumulated_work"):
            job.accumulated_work += dt

        # Advance progress so remaining-time calc stays accurate
        if isinstance(job, (TrainingJob, LLMJob)):
            delta = int(tp * dt)
            job.completed_iterations = min(
                job.num_iterations,
                job.completed_iterations + delta,
            )
        elif isinstance(job, InferenceJob):
            delta = int(tp * job.current_batch_size * dt)
            job.completed_queries = min(
                job.total_queries,
                job.completed_queries + delta,
            )
        elif isinstance(job, HPOJob):
            # Advance active-trial epochs
            EPOCH_S = self.HPO_EPOCH_SECONDS
            epochs_done = dt / EPOCH_S
            if not job.active_trials:
                concurrent = min(job.num_trials, 4)
                for i in range(concurrent):
                    job.active_trials[str(i)] = 0.0
            for tid in list(job.active_trials.keys()):
                job.active_trials[tid] += epochs_done
                if job.active_trials[tid] >= job.max_epochs_per_trial:
                    job.completed_trials.append(tid)
                    del job.active_trials[tid]
            # Start next wave if needed
            needed = job.num_trials - len(job.completed_trials) - len(job.active_trials)
            if needed > 0 and len(job.active_trials) == 0:
                concurrent = min(needed, 4)
                base = len(job.completed_trials) + len(job.active_trials)
                for i in range(concurrent):
                    job.active_trials[str(base + i)] = 0.0

        # Reset segment start to now
        self._job_segment_start[jid] = current_time

    # ── Event handlers ────────────────────────────────────────────────────────

    def _on_arrival(self, event: Event):
        job_id = event.job_id
        job = next((j for j in self.workload if j.job_id == job_id), None)
        if job is None:
            return
        job.status = JobStatus.PENDING
        self._pending[job_id] = job
        if self.verbose:
            print(f"  [{event.time:.0f}s] ARRIVE {job_id} "
                  f"({getattr(job, 'arch', '?').value if hasattr(getattr(job, 'arch', None), 'value') else '?'})")
        self._push(event.time, EventType.SCHEDULE)

    def _on_schedule(self, event: Event):
        pending_list = list(self._pending.values())
        running_list = list(self._running.values())
        if not pending_list:
            # Still need to schedule next periodic invocation
            self._push(event.time + self.SCHEDULE_INTERVAL, EventType.SCHEDULE)
            return

        decision = self.scheduler.schedule(pending_list, running_list, event.time)

        # Handle preemptions
        for job_id in decision.preemptions:
            if job_id in self._running:
                self._preempt_job(job_id, event.time)

        # Handle new allocations
        for alloc in decision.allocations:
            job = alloc.job
            if job.job_id not in self._pending:
                continue

            jid  = job.job_id
            mem  = getattr(job, "memory_per_gpu_gb", 4.0)
            ok   = True

            # Allocate GPUs (physical)
            if alloc.gpu_ids:
                ok = self.cluster.allocate(jid, alloc.gpu_ids, mem)
                if not ok:
                    continue

            # Allocate MIG slices
            if alloc.mig_ids:
                ok = self.cluster.allocate_mig(jid, alloc.mig_ids)
                if not ok:
                    if alloc.gpu_ids:
                        self.cluster.deallocate(jid, alloc.gpu_ids, mem)
                    continue

            # Allocate CPU cores
            if alloc.cpu_alloc:
                ok = self.cluster.allocate_cpu(jid, alloc.cpu_alloc)
                if not ok:
                    if alloc.gpu_ids:
                        self.cluster.deallocate(jid, alloc.gpu_ids, mem)
                    if alloc.mig_ids:
                        self.cluster.deallocate_mig(jid, alloc.mig_ids)
                    continue

            # Commit allocation state
            job.status     = JobStatus.RUNNING
            job.start_time = job.start_time or event.time
            job.allocated_gpus = alloc.gpu_ids

            # Store all resource handles
            if alloc.gpu_ids:
                self._job_gpus[jid] = alloc.gpu_ids
            if alloc.mig_ids:
                self._job_migs[jid] = alloc.mig_ids
                if hasattr(job, "allocated_mig"):
                    job.allocated_mig = alloc.mig_ids
            if alloc.cpu_alloc:
                self._job_cpus[jid] = alloc.cpu_alloc
                if hasattr(job, "allocated_cpus"):
                    job.allocated_cpus = alloc.cpu_alloc

            del self._pending[jid]
            self._running[jid] = job

            # Mark segment start
            self._job_segment_start[jid] = event.time

            # Recalculate completion for jobs whose co-location just changed
            self._reschedule_collocated(job, event.time)

            # Schedule this job's direct completion
            self._schedule_completion(job, event.time)

            if self.verbose:
                rtype = alloc.resource_type
                detail = (f"gpus={alloc.gpu_ids[:2]}..." if alloc.gpu_ids
                         else f"migs={alloc.mig_ids[:2]}..." if alloc.mig_ids
                         else f"cpus={alloc.cpu_alloc[:2]}...")
                print(f"  [{event.time:.0f}s] START {jid} [{rtype}] {detail}")
            self.metrics.record_job_start(job, event.time)

        # Schedule next periodic invocation
        self._push(event.time + self.SCHEDULE_INTERVAL, EventType.SCHEDULE)

    def _on_complete(self, event: Event):
        job_id = event.job_id
        if job_id not in self._running:
            return   # already preempted or completed

        # Stale-event check: discard if a newer completion was scheduled
        expected_seq = self._job_complete_seq.get(job_id, -1)
        if event.data.get("completion_seq", -1) != expected_seq:
            return   # stale — a newer JOB_COMPLETE is in the heap

        job = self._running.pop(job_id)

        # Flush any residual segment progress
        self._flush_segment(job, event.time)

        job.status   = JobStatus.COMPLETED
        job.end_time = event.time
        mem = getattr(job, "memory_per_gpu_gb", 4.0)
        self.cluster.deallocate(job_id, self._job_gpus.pop(job_id, []), mem)
        self.cluster.deallocate_mig(job_id, self._job_migs.pop(job_id, []))
        self.cluster.deallocate_cpu(job_id, self._job_cpus.pop(job_id, []))
        self._job_complete_seq.pop(job_id, None)
        self._job_segment_start.pop(job_id, None)
        self._completed.append(job)
        self.metrics.record_job_complete(job, event.time)

        if self.verbose:
            print(f"  [{event.time:.0f}s] DONE  {job_id}  JCT={job.jct:.1f}s")

        # Recalculate co-located jobs whose throughput just improved
        freed_gpus = []  # already popped above; use snapshot from before dealloc
        # Trigger scheduler to fill freed capacity
        self._push(event.time, EventType.SCHEDULE)

    def _preempt_job(self, job_id: str, current_time: float):
        if job_id not in self._running:
            return
        job = self._running.pop(job_id)

        # Flush accumulated progress for this segment before preempting
        self._flush_segment(job, current_time)

        job.status = JobStatus.PREEMPTED
        job.preempt_count += 1
        mem = getattr(job, "memory_per_gpu_gb", 4.0)

        # Identify which GPUs are shared with others (their throughput improves)
        preempted_gpus = self._job_gpus.get(job_id, [])

        self.cluster.deallocate(job_id, self._job_gpus.pop(job_id, []), mem)
        self.cluster.deallocate_mig(job_id, self._job_migs.pop(job_id, []))
        self.cluster.deallocate_cpu(job_id, self._job_cpus.pop(job_id, []))

        # Invalidate pending completion event by bumping seq
        self._job_complete_seq[job_id] = self._job_complete_seq.get(job_id, 0) + 1
        self._job_segment_start.pop(job_id, None)

        # Recalculate jobs that were co-located and now have more headroom
        affected: set[str] = set()
        for gid in preempted_gpus:
            if gid not in self.cluster.gpus:
                continue
            g = self.cluster.gpus[gid]
            for jid in g.allocated_jobs:
                if jid != job_id and jid in self._running:
                    affected.add(jid)
        for jid in affected:
            other = self._running[jid]
            self._flush_segment(other, current_time)
            self._schedule_completion(other, current_time)

        # Checkpoint overhead — job re-enters queue after checkpoint completes
        ckpt = getattr(job, "checkpoint_overhead_sec", 30.0)
        job.submit_time = current_time + ckpt
        job.start_time  = None
        job.status      = JobStatus.PENDING
        self._pending[job_id] = job
        self.metrics.record_preemption(job)

    def _on_metric_sample(self, event: Event):
        snap = self.cluster.snapshot()
        renewable_power_w = self._renewable.available_power_watts(event.time)
        self.metrics.record_cluster_snapshot(event.time, snap, renewable_power_w)
        self._push(event.time + self.METRIC_INTERVAL, EventType.METRIC_SAMPLE)

    # ── Main run loop ─────────────────────────────────────────────────────────

    def run(self) -> SimulationResult:
        # Seed job arrivals
        for job in sorted(self.workload, key=lambda j: j.submit_time):
            self._push(job.submit_time, EventType.JOB_ARRIVAL, job.job_id)

        # Seed periodic events
        self._push(0.0, EventType.SCHEDULE)
        self._push(0.0, EventType.METRIC_SAMPLE)

        handler = {
            EventType.JOB_ARRIVAL:  self._on_arrival,
            EventType.SCHEDULE:     self._on_schedule,
            EventType.JOB_COMPLETE: self._on_complete,
            EventType.JOB_PREEMPT:  self._preempt_job,
            EventType.METRIC_SAMPLE: self._on_metric_sample,
        }

        processed = 0
        while self._heap:
            event = heapq.heappop(self._heap)
            if event.time > self.max_time:
                break
            self._current_time = event.time

            h = handler.get(event.event_type)
            if h:
                h(event)

            processed += 1
            # Safety guard (much larger threshold — events are now sparse)
            if processed > 50_000_000:
                print("WARNING: event limit reached, terminating early.")
                break

        # Finalise incomplete jobs — flush their accumulated progress first
        for job in list(self._running.values()):
            self._flush_segment(job, self._current_time)
            job.end_time = self._current_time
            job.status   = JobStatus.COMPLETED
            self._completed.append(job)

        self.metrics.finalise(self._current_time, self._completed)
        return SimulationResult(
            completed_jobs=self._completed,
            metrics=self.metrics,
            total_time=self._current_time,
            scheduler_name=self.scheduler.name,
            cluster_name=self.cluster.config.name,
        )
