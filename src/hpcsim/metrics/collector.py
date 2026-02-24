"""
Metrics Collector — updated with renewable energy utilization (ReUtil).

New metrics vs original:
  - renewable_energy_utilization  (0–1 ratio)
  - avg_bsld                      (Average Bounded Slowdown, GAS-MARL metric)
  - renewable_energy_wh           (total renewable Wh consumed)
  - brown_energy_wh               (total brown/grid Wh consumed)

Existing metrics retained:
  - JCT (mean, median, p90, p99, std)
  - Queue time statistics
  - GPU utilization (time series + average)
  - Jain's Fairness Index
  - Deadline miss rate / SLO violations
  - Total energy consumption (kWh)
  - Preemption counts
"""

from __future__ import annotations
from dataclasses import dataclass, field
from typing import Optional
import math
import statistics


@dataclass
class ClusterSnapshot:
    time: float
    total_gpus: int
    busy_gpus: int
    free_gpus: int
    utilization: float
    power_watts: float
    renewable_power_watts: float = 0.0    # ← NEW: green power available at this tick


@dataclass
class MetricsCollector:
    job_starts:     dict = field(default_factory=dict)
    job_events:     list = field(default_factory=list)
    snapshots:      list[ClusterSnapshot] = field(default_factory=list)
    preemption_count: int = 0
    slo_violations: int = 0

    # ── Recording hooks ───────────────────────────────────────────────────────

    def record_job_start(self, job, current_time: float):
        self.job_starts[job.job_id] = current_time

    def record_job_complete(self, job, current_time: float):
        exec_time = max(1.0, (job.end_time or current_time) - (job.start_time or current_time))
        wait_time = job.queue_time if job.queue_time < float("inf") else 0.0
        tau       = 10.0   # BSLD threshold τ (GAS-MARL §3.2)
        bsld      = max((wait_time + exec_time) / max(tau, exec_time), 1.0)

        self.job_events.append({
            "job_id":     job.job_id,
            "user_id":    job.user_id,
            "jct":        job.jct,
            "queue_time": queue_time if (queue_time := job.queue_time) < float("inf") else 0.0,
            "exec_time":  exec_time,
            "bsld":       bsld,
            "preempts":   job.preempt_count,
            "deadline":   getattr(job, "deadline", None),
            "end_time":   current_time,
            "missed_dl":  (
                current_time > job.deadline
                if getattr(job, "deadline", None) else False
            ),
        })
        violations = getattr(job, "slo_violations", 0)
        self.slo_violations += violations

    def record_preemption(self, job):
        self.preemption_count += 1

    def record_cluster_snapshot(
        self,
        t: float,
        snap: dict,
        renewable_power_w: float = 0.0,   # ← NEW parameter
    ):
        self.snapshots.append(ClusterSnapshot(
            time=t,
            total_gpus=snap["total_gpus"],
            busy_gpus=snap["busy_gpus"],
            free_gpus=snap["free_gpus"],
            utilization=snap["utilization"],
            power_watts=snap["power_watts"],
            renewable_power_watts=renewable_power_w,
        ))

    def finalise(self, end_time: float, completed_jobs: list):
        pass   # hook for post-processing

    # ── JCT statistics ────────────────────────────────────────────────────────

    def jct_stats(self) -> dict:
        jcts = sorted(e["jct"] for e in self.job_events if e["jct"] < float("inf"))
        if not jcts:
            return {"mean": 0, "median": 0, "p90": 0, "p99": 0, "std": 0, "n": 0}
        n = len(jcts)
        return {
            "mean":   statistics.mean(jcts),
            "median": statistics.median(jcts),
            "p90":    jcts[int(0.90 * n)],
            "p99":    jcts[min(int(0.99 * n), n - 1)],
            "std":    statistics.stdev(jcts) if n > 1 else 0.0,
            "n":      n,
        }

    def queue_time_stats(self) -> dict:
        qts = sorted(e["queue_time"] for e in self.job_events if e["queue_time"] < float("inf"))
        if not qts:
            return {"mean": 0, "median": 0, "p90": 0, "p99": 0}
        n = len(qts)
        return {
            "mean":   statistics.mean(qts),
            "median": statistics.median(qts),
            "p90":    qts[int(0.90 * n)],
            "p99":    qts[min(int(0.99 * n), n - 1)],
        }

    # ── GPU utilisation ───────────────────────────────────────────────────────

    def avg_gpu_utilization(self) -> float:
        if not self.snapshots:
            return 0.0
        return statistics.mean(s.utilization for s in self.snapshots)

    def utilization_time_series(self) -> tuple[list[float], list[float]]:
        return [s.time for s in self.snapshots], [s.utilization for s in self.snapshots]

    # ── BSLD (GAS-MARL §3.2) ─────────────────────────────────────────────────

    def avg_bsld(self) -> float:
        """
        Average Bounded Slowdown:
        AvgBSLD = (1/N) × Σ max((w_i + e_i) / max(τ, e_i), 1)
        where τ = 10 s (threshold to prevent short jobs dominating).
        """
        bslds = [e["bsld"] for e in self.job_events if e.get("bsld") is not None]
        return statistics.mean(bslds) if bslds else 0.0

    # ── Renewable Energy Utilization (ReUtil) ─────────────────────────────────

    def renewable_energy_utilization(self) -> float:
        """
        ReUtil = ∫ min(P_green, P_cluster) dt / ∫ P_cluster dt
        (GAS-MARL §3.2, Eq. 4)

        Uses cluster snapshots which now carry renewable_power_watts.
        """
        if len(self.snapshots) < 2:
            return 0.0
        green_ws  = 0.0
        total_ws  = 0.0
        for i in range(1, len(self.snapshots)):
            dt        = self.snapshots[i].time - self.snapshots[i - 1].time
            avg_clust = (self.snapshots[i].power_watts +
                         self.snapshots[i - 1].power_watts) / 2.0
            avg_green = (self.snapshots[i].renewable_power_watts +
                         self.snapshots[i - 1].renewable_power_watts) / 2.0
            green_ws  += min(avg_green, avg_clust) * dt
            total_ws  += avg_clust * dt
        return green_ws / max(total_ws, 1.0)

    def renewable_energy_wh(self) -> float:
        """Total renewable energy consumed (Wh)."""
        if len(self.snapshots) < 2:
            return 0.0
        wh = 0.0
        for i in range(1, len(self.snapshots)):
            dt        = self.snapshots[i].time - self.snapshots[i - 1].time
            avg_clust = (self.snapshots[i].power_watts +
                         self.snapshots[i - 1].power_watts) / 2.0
            avg_green = (self.snapshots[i].renewable_power_watts +
                         self.snapshots[i - 1].renewable_power_watts) / 2.0
            wh += min(avg_green, avg_clust) * dt / 3600.0
        return wh

    def brown_energy_wh(self) -> float:
        """Total brown (grid) energy consumed (Wh)."""
        if len(self.snapshots) < 2:
            return 0.0
        wh = 0.0
        for i in range(1, len(self.snapshots)):
            dt        = self.snapshots[i].time - self.snapshots[i - 1].time
            avg_clust = (self.snapshots[i].power_watts +
                         self.snapshots[i - 1].power_watts) / 2.0
            avg_green = (self.snapshots[i].renewable_power_watts +
                         self.snapshots[i - 1].renewable_power_watts) / 2.0
            wh += max(0.0, avg_clust - avg_green) * dt / 3600.0
        return wh

    # ── Fairness ─────────────────────────────────────────────────────────────

    def jains_fairness_index(self) -> float:
        user_jcts: dict[str, list[float]] = {}
        for e in self.job_events:
            if e["jct"] < float("inf"):
                user_jcts.setdefault(e["user_id"], []).append(e["jct"])
        if not user_jcts:
            return 1.0
        means = [statistics.mean(v) for v in user_jcts.values()]
        n  = len(means)
        s1 = sum(means)
        s2 = sum(m * m for m in means)
        return (s1 * s1) / (n * s2) if s2 > 0 else 1.0

    # ── Deadline + SLO ────────────────────────────────────────────────────────

    def deadline_miss_rate(self) -> float:
        with_dl = [e for e in self.job_events if e["deadline"] is not None]
        if not with_dl:
            return 0.0
        return sum(1 for e in with_dl if e["missed_dl"]) / len(with_dl)

    def slo_violation_rate(self) -> float:
        total_queries = sum(e.get("total_queries", 0) for e in self.job_events)
        return self.slo_violations / max(total_queries, 1)

    # ── Legacy energy (trapezoid from power samples) ─────────────────────────

    def total_energy_kwh(self) -> float:
        if len(self.snapshots) < 2:
            return 0.0
        energy_ws = 0.0
        for i in range(1, len(self.snapshots)):
            dt = self.snapshots[i].time - self.snapshots[i - 1].time
            avg_pw = (self.snapshots[i].power_watts +
                      self.snapshots[i - 1].power_watts) / 2.0
            energy_ws += avg_pw * dt
        return energy_ws / 3_600_000.0

    # ── Summary dict ─────────────────────────────────────────────────────────

    def summary(self) -> dict:
        jct = self.jct_stats()
        return {
            "jobs_completed":               jct["n"],
            "avg_jct_s":                    jct["mean"],
            "median_jct_s":                 jct["median"],
            "p90_jct_s":                    jct["p90"],
            "p99_jct_s":                    jct["p99"],
            "std_jct_s":                    jct["std"],
            "avg_queue_s":                  self.queue_time_stats()["mean"],
            "avg_bsld":                     self.avg_bsld(),             # ← NEW
            "avg_gpu_util":                 self.avg_gpu_utilization(),
            "jains_fairness":               self.jains_fairness_index(),
            "deadline_miss_pct":            self.deadline_miss_rate() * 100,
            "slo_violations":               self.slo_violations,
            "total_energy_kwh":             self.total_energy_kwh(),
            "renewable_energy_utilization": self.renewable_energy_utilization(),  # ← NEW
            "renewable_energy_wh":          self.renewable_energy_wh(),           # ← NEW
            "brown_energy_wh":              self.brown_energy_wh(),               # ← NEW
            "preemptions":                  self.preemption_count,
        }
