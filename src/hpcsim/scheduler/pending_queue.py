"""
Optimised pending-job priority queue data structures for HPC simulation.

Problem solved

Every ``scheduler.schedule()`` call previously:
  1. Created a list of all J pending jobs     → O(J)
  2. Called ``sorted(pending, key=...)``       → O(J log J)

When J = 50 000 and the scheduler is invoked on every JOB_ARRIVAL /
JOB_COMPLETE event, the cumulative sort cost dominates the simulation.

Solution

PendingJobQueue
    Min-heap keyed by (submit_time, insertion_seq).

    Complexity budget
    
    push()        O(log J)        — heap insert
    remove()      O(1) amortised  — lazy deletion via _jobs dict
    pop()         O(log J)        — heap pop + ghost purge
    peek()        O(log J)        — same
    iter_fifo()   O(J log J)      — non-destructive heap snapshot
                  ↑ same asymptotic as sorted(), but the heap is already
                    partially sorted so the constant is smaller, and the
                    caller can stop early (generator) saving O((J-K) log J)
                    when only K < J jobs fit the cluster.
    values()      O(J)

    Lazy deletion works by keeping an authoritative membership dict
    ``_jobs`` (job_id → job).  Ghost entries in the heap (job_ids no
    longer in ``_jobs``) are discarded silently on the next pop/peek.

MultiLevelQueue
    N separate PendingJobQueues — one per priority level.
    Used by TiresiasLASScheduler and MLFQScheduler to eliminate the
    O(J log J) re-sort on every scheduling event.

    sync(pending_list)  O(J)   — reconcile with fresh engine snapshot
    iter_by_priority()         — yield level-0 first, FIFO within level

    The service metric can be customised via service_fn.

Backward compatibility

PendingJobQueue exposes the same mapping interface as the ``dict`` that
the engine previously used for ``_pending``:

    __contains__, __len__, __bool__, __iter__ (yields job objects),
    __getitem__, __setitem__ (→ push), __delitem__ (→ remove)

This means the only engine change required is the type declaration and
removing the now-redundant ``list(...)`` conversion in _do_schedule.
"""

from __future__ import annotations

import heapq
from typing import Callable, Iterator, Optional

from ..workload.job import AnyJob


__all__ = ["PendingJobQueue", "MultiLevelQueue"]


# 
# PendingJobQueue
# 

class PendingJobQueue:
    """
    Priority queue for pending HPC jobs.

    Internal layout
    
    _heap : list[(key: float, seq: int, job_id: str)]
        Min-heap.  A job may have multiple entries (one per push call), but
        only the newest one is "live" — the rest are ghosts (see _jobs).

    _jobs : dict[str, AnyJob]
        Authoritative membership set.
        A job is pending iff ``job_id in _jobs``.
        Ghost purging skips heap entries whose job_id is missing from _jobs.

    _seq  : monotone integer that breaks submit_time ties deterministically
            and that uniquely identifies each push call.

    Ghost purging
    
    A ghost arises when remove() is called (sets _jobs entry to absent) while
    a matching heap entry still exists.  Ghosts are silently skipped in pop(),
    peek(), and iter_fifo().  The heap therefore grows at most by a factor of
    2× in pathological preemption scenarios, but in typical use (remove is
    called once per job) it stays lean.

    Re-push safety
    
    Calling push() for a job_id that is already present just updates the
    object reference in _jobs without adding a duplicate heap entry.
    Use repush() when the heap key (submit_time) has genuinely changed.
    """

    __slots__ = ("_heap", "_jobs", "_seq")

    def __init__(self) -> None:
        self._heap: list[tuple[float, int, str]] = []   # (key, seq, job_id)
        self._jobs: dict[str, AnyJob]            = {}   # job_id → job
        self._seq:  int                          = 0

    # Insertion 

    def push(self, job: AnyJob, key: Optional[float] = None) -> None:
        """
        O(log J) amortised.

        Insert *job* with priority *key* (defaults to ``job.submit_time``).
        If *job_id* is already present the object reference is updated but
        **no duplicate heap entry is created** (use repush() if the key
        changed, e.g. after preemption updates submit_time).
        """
        if job.job_id not in self._jobs:
            k = key if key is not None else job.submit_time
            self._seq += 1
            heapq.heappush(self._heap, (k, self._seq, job.job_id))
        # Always refresh the object reference (handles in-place mutations)
        self._jobs[job.job_id] = job

    def repush(self, job: AnyJob, key: Optional[float] = None) -> None:
        """
        Remove and re-insert *job* with an (optionally) new heap key.

        Use this when ``job.submit_time`` changed (e.g. a preempted job
        that was re-queued with ``submit_time = current_time + checkpoint``).
        O(log J) amortised.
        """
        self.remove(job.job_id)   # lazy-delete old entry
        self.push(job, key)       # fresh insert with new key

    # Removal 

    def remove(self, job_id: str) -> Optional[AnyJob]:
        """
        O(1) amortised (lazy deletion).

        Drops *job_id* from the authoritative ``_jobs`` dict.  The
        corresponding heap entry becomes a ghost that is silently skipped
        the next time it surfaces during pop / peek / iter_fifo.

        Returns the removed job, or None if it was not present.
        """
        return self._jobs.pop(job_id, None)

    # Access 

    def peek(self) -> Optional[AnyJob]:
        """
        O(log J) amortised.  Return (without removing) the min-key job.
        Purges ghosts from the heap front as a side-effect.
        """
        self._purge_ghosts()
        if not self._heap:
            return None
        return self._jobs[self._heap[0][2]]

    def pop(self) -> Optional[AnyJob]:
        """
        O(log J).  Remove and return the min-key job.
        Returns None if the queue is empty.
        """
        self._purge_ghosts()
        if not self._heap:
            return None
        _, _, jid = heapq.heappop(self._heap)
        return self._jobs.pop(jid, None)

    # Internal ghost purge 

    def _purge_ghosts(self) -> None:
        """Discard heap entries whose job_id is no longer in _jobs."""
        while self._heap and self._heap[0][2] not in self._jobs:
            heapq.heappop(self._heap)

    # Iteration 

    def iter_fifo(self) -> Iterator[AnyJob]:
        """
        O(J log J).  Yield all pending jobs in ascending submit_time order.

        Non-destructive: operates on a heap snapshot so the queue is
        unchanged after iteration.

        Supports early-exit: callers that stop iterating once the cluster
        fills pay only O(K log J) rather than O(J log J), where K is the
        number of jobs successfully scheduled before resources ran out.
        """
        # Build a compact snapshot: heap entries whose job is still pending.
        snapshot: list[tuple[float, int, str]] = [
            entry for entry in self._heap if entry[2] in self._jobs
        ]
        heapq.heapify(snapshot)
        seen: set[str] = set()
        while snapshot:
            _, _, jid = heapq.heappop(snapshot)
            if jid in self._jobs and jid not in seen:
                seen.add(jid)
                yield self._jobs[jid]

    def sorted_by(
        self,
        key: Callable[[AnyJob], float],
        reverse: bool = False,
    ) -> list[AnyJob]:
        """O(J log J).  Return all pending jobs sorted by *key* function."""
        return sorted(self._jobs.values(), key=key, reverse=reverse)

    def values(self) -> list[AnyJob]:
        """O(J).  Return all pending jobs in arbitrary order."""
        return list(self._jobs.values())

    # Mapping interface (backward-compat with engine's _pending dict API) 

    def __contains__(self, job_id: str) -> bool:
        """O(1).  Test membership by job_id string."""
        return job_id in self._jobs

    def __len__(self) -> int:
        return len(self._jobs)

    def __bool__(self) -> bool:
        return bool(self._jobs)

    def __iter__(self) -> Iterator[AnyJob]:
        """Iterate over job *objects* (not job_ids).
        Enables ``for job in pending`` in schedulers and BackfillWrapper.
        Note: ``jid in pending`` still uses __contains__ (checks by string key).
        """
        return iter(self._jobs.values())

    def __getitem__(self, job_id: str) -> AnyJob:
        return self._jobs[job_id]

    def __setitem__(self, job_id: str, job: AnyJob) -> None:
        """Backward-compat: ``_pending[jid] = job`` → push."""
        self.push(job)

    def __delitem__(self, job_id: str) -> None:
        """Backward-compat: ``del _pending[jid]`` → remove (lazy delete)."""
        self.remove(job_id)

    def __repr__(self) -> str:
        return (f"PendingJobQueue("
                f"pending={len(self._jobs)}, "
                f"heap={len(self._heap)}, "
                f"ghosts={len(self._heap) - len(self._jobs)})")


# 
# MultiLevelQueue
# 

class MultiLevelQueue:
    """
    N-level priority queue for MLFQ and Tiresias LAS schedulers.

    Motivation
    
    Both MLFQ and Tiresias sort the pending list by a service metric
    (accumulated_work or attained_service) on every schedule() call.
    With J = 50 000 pending jobs this is O(J log J) per call.

    MultiLevelQueue maintains N separate PendingJobQueues (one per level).
    The scheduling loop calls ``iter_by_priority()`` which yields level-0
    jobs first, then level-1, etc., FIFO within each level.  The key win:

    - No full re-sort.  The heap in each level is already ordered by
      submit_time; crossing level boundaries is O(1).
    - Jobs only migrate between levels when their service crosses a
      threshold — not on every scheduling call.
    - sync() reconciles with the engine's pending list in O(J) rather
      than O(J log J).

    Parameters
    
    thresholds  : sorted list of float boundaries for the service metric.
                  A job is placed in level i if
                  thresholds[i-1] < service ≤ thresholds[i].
                  The last level covers (thresholds[-1], ∞).

    service_fn  : callable (AnyJob) → float  (default: attained_service).
                  Override with ``lambda j: getattr(j, "accumulated_work", 0.)``
                  for MLFQ.

    Example — Tiresias (GPU-seconds, Table 3)
    
    ::
        _TIRESIAS_THRESHOLDS = [60, 180, 600, 1800]
        mlq = MultiLevelQueue(_TIRESIAS_THRESHOLDS)
        # → 5 levels:
        #  0: [0, 60]        highest priority
        #  1: (60, 180]
        #  2: (180, 600]
        #  3: (600, 1800]
        #  4: (1800, ∞)      lowest priority
    """

    def __init__(
        self,
        thresholds: list[float],
        service_fn: Optional[Callable[[AnyJob], float]] = None,
    ) -> None:
        self._thresholds: list[float] = sorted(thresholds)
        self._n: int = len(thresholds) + 1
        self._queues: list[PendingJobQueue] = [
            PendingJobQueue() for _ in range(self._n)
        ]
        self._level_of: dict[str, int] = {}   # job_id → current level index
        self._service_fn: Callable[[AnyJob], float] = (
            service_fn or (lambda j: getattr(j, "attained_service", 0.0))
        )

    # Level assignment 

    def _level_for(self, job: AnyJob) -> int:
        """O(levels).  Determine which level a job belongs to."""
        svc = self._service_fn(job)
        for i, thr in enumerate(self._thresholds):
            if svc <= thr:
                return i
        return self._n - 1

    # Mutation 

    def push(self, job: AnyJob) -> None:
        """
        Insert or re-classify *job* based on its current service value.

        - New job: inserted at its correct level.  O(log J).
        - Existing job, same level: object reference updated in-place.  O(1).
        - Existing job, level changed: moved to new level.  O(log J).
        """
        jid = job.job_id
        new_lvl = self._level_for(job)
        old_lvl = self._level_of.get(jid)

        if old_lvl is not None:
            if old_lvl == new_lvl:
                # Fast path: no level migration, just refresh object reference
                self._queues[new_lvl]._jobs[jid] = job
                return
            # Level promotion / demotion
            self._queues[old_lvl].remove(jid)

        self._level_of[jid] = new_lvl
        self._queues[new_lvl].push(job)

    def remove(self, job_id: str) -> Optional[AnyJob]:
        """O(1) amortised.  Remove job from its current level."""
        lvl = self._level_of.pop(job_id, None)
        if lvl is None:
            return None
        return self._queues[lvl].remove(job_id)

    def sync(self, pending: list[AnyJob]) -> None:
        """
        O(J).  Reconcile the MLQ with a fresh pending list from the engine.

        Step 1 — drop stale entries (jobs that started running / completed).
        Step 2 — upsert remaining jobs (new arrivals + re-classify if service
                 crossed a threshold while the job was waiting in the queue).
        """
        incoming_ids: set[str] = {j.job_id for j in pending}

        # Remove jobs no longer pending
        for jid in list(self._level_of):
            if jid not in incoming_ids:
                self.remove(jid)

        # Insert new jobs and re-classify existing ones
        for job in pending:
            self.push(job)

    # Iteration 

    def iter_by_priority(self) -> Iterator[AnyJob]:
        """
        Yield all jobs in priority order.

        Order: level-0 first (highest priority, least service), then
        level-1, …, level-(N-1).  Within each level: ascending submit_time.

        Non-destructive.  Supports early-exit (generator).
        """
        for q in self._queues:
            yield from q.iter_fifo()

    def values(self) -> list[AnyJob]:
        """All pending jobs in arbitrary order."""
        out: list[AnyJob] = []
        for q in self._queues:
            out.extend(q.values())
        return out

    # Sizing 

    def __len__(self) -> int:
        return len(self._level_of)

    def __bool__(self) -> bool:
        return bool(self._level_of)

    def __contains__(self, job_id: str) -> bool:
        return job_id in self._level_of

    def __repr__(self) -> str:
        sizes = [len(q) for q in self._queues]
        return f"MultiLevelQueue(levels={self._n}, sizes={sizes})"