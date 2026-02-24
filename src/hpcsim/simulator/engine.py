"""
Discrete Event Simulation Engine.

Events:
- JOB_ARRIVAL     : new job enters the queue
- SCHEDULE        : scheduler invoked
- JOB_PROGRESS    : advance job progress tick
- JOB_COMPLETE    : job finishes
- JOB_PREEMPT     : job preempted (checkpointed)
- METRIC_SAMPLE   : periodic metrics snapshot
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
    JOB_PROGRESS   = "JOB_PROGRESS"
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
    - Progress ticks model training iterations and inference queries
    - Preemption with checkpoint overhead (T6 – gang/elastic modes)
    """

    TICK_INTERVAL     = 10.0   # seconds per progress tick
    SCHEDULE_INTERVAL = 5.0    # seconds between scheduler invocations
    METRIC_INTERVAL   = 60.0   # seconds between metric snapshots

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
        self._current_time = 0.0

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

    # ── Event handlers ────────────────────────────────────────────────────────

    def _on_arrival(self, event: Event):
        job_id = event.job_id
        job = next((j for j in self.workload if j.job_id == job_id), None)
        if job is None:
            return
        job.status = JobStatus.PENDING
        self._pending[job_id] = job
        if self.verbose:
            print(f"  [{event.time:.0f}s] ARRIVE {job_id} ({getattr(job, 'arch', '?').value if hasattr(getattr(job, 'arch', None), 'value') else '?'})")
        self._push(event.time, EventType.SCHEDULE)

    def _on_schedule(self, event: Event):
        pending_list = list(self._pending.values())
        running_list = list(self._running.values())
        if not pending_list:
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
            mem = getattr(job, "memory_per_gpu_gb", 4.0)
            ok  = self.cluster.allocate(job.job_id, alloc.gpu_ids, mem)
            if not ok:
                continue
            job.status      = JobStatus.RUNNING
            job.start_time  = job.start_time or event.time
            job.allocated_gpus = alloc.gpu_ids
            self._job_gpus[job.job_id] = alloc.gpu_ids
            del self._pending[job.job_id]
            self._running[job.job_id] = job
            # Schedule first progress tick
            self._push(event.time + self.TICK_INTERVAL, EventType.JOB_PROGRESS, job.job_id)
            if self.verbose:
                print(f"  [{event.time:.0f}s] START {job.job_id} gpus={alloc.gpu_ids[:2]}...")
            self.metrics.record_job_start(job, event.time)

        # Schedule next invocation
        self._push(event.time + self.SCHEDULE_INTERVAL, EventType.SCHEDULE)

    def _on_progress(self, event: Event):
        job_id = event.job_id
        if job_id not in self._running:
            return
        job = self._running[job_id]
        tp  = self._compute_throughput(job)

        dt = self.TICK_INTERVAL
        delta_iter = tp * dt

        # Update attained service and accumulated work
        k = len(self._job_gpus.get(job_id, []))
        atttained_delta = dt * k
        if hasattr(job, "attained_service"):
            job.attained_service  += atttained_delta
        if hasattr(job, "accumulated_work"):
            job.accumulated_work  += dt

        finished = False

        if isinstance(job, (TrainingJob, LLMJob)):
            job.completed_iterations = min(
                job.num_iterations,
                job.completed_iterations + int(delta_iter),
            )
            finished = job.completed_iterations >= job.num_iterations

        elif isinstance(job, InferenceJob):
            batch_tp = tp * job.current_batch_size
            job.completed_queries = min(
                job.total_queries,
                job.completed_queries + int(batch_tp * dt),
            )
            finished = job.completed_queries >= job.total_queries

        elif isinstance(job, HPOJob):
            # HPO: advance epoch budgets
            epochs_done = dt / 300.0   # ~300s per epoch
            if not job.active_trials:
                # Spawn trials
                for i in range(min(job.num_trials, 4)):
                    job.active_trials[str(i)] = 0.0
            for tid in list(job.active_trials.keys()):
                job.active_trials[tid] += epochs_done
                if job.active_trials[tid] >= job.max_epochs_per_trial:
                    job.completed_trials.append(tid)
                    del job.active_trials[tid]
            finished = len(job.completed_trials) >= job.num_trials

        if finished:
            self._push(event.time + dt, EventType.JOB_COMPLETE, job_id)
        else:
            self._push(event.time + dt, EventType.JOB_PROGRESS, job_id)

    def _on_complete(self, event: Event):
        job_id = event.job_id
        if job_id not in self._running:
            return
        job = self._running.pop(job_id)
        job.status   = JobStatus.COMPLETED
        job.end_time = event.time
        mem = getattr(job, "memory_per_gpu_gb", 4.0)
        self.cluster.deallocate(job_id, self._job_gpus.pop(job_id, []), mem)
        self._completed.append(job)
        self.metrics.record_job_complete(job, event.time)
        if self.verbose:
            print(f"  [{event.time:.0f}s] DONE  {job_id}  JCT={job.jct:.1f}s")
        # Trigger scheduler to fill freed capacity
        self._push(event.time, EventType.SCHEDULE)

    def _preempt_job(self, job_id: str, current_time: float):
        if job_id not in self._running:
            return
        job = self._running.pop(job_id)
        job.status = JobStatus.PREEMPTED
        job.preempt_count += 1
        mem = getattr(job, "memory_per_gpu_gb", 4.0)
        self.cluster.deallocate(job_id, self._job_gpus.pop(job_id, []), mem)
        # Checkpoint overhead
        ckpt = getattr(job, "checkpoint_overhead_sec", 30.0)
        job.submit_time = current_time + ckpt   # re-enters queue after checkpoint
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
            EventType.JOB_PROGRESS: self._on_progress,
            EventType.JOB_COMPLETE: self._on_complete,
            EventType.JOB_PREEMPT:  self._preempt_job,
            EventType.METRIC_SAMPLE:self._on_metric_sample,
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
            # Safety guard: infinite loop protection
            if processed > 5_000_000:
                print("WARNING: event limit reached, terminating early.")
                break

        # Finalise incomplete jobs
        for job in list(self._running.values()):
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
