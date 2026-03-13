"""
Metrics Collector.

Metrics tracked:
  JCT (mean, median, p90, p99, std), Queue time, GPU utilization,
  CPU utilization, Jain's Fairness Index, Deadline miss rate,
  SLO violations, Total energy (kWh), Renewable energy utilization (ReUtil),
  AvgBSLD (GAS-MARL §3.2), Preemption counts,
  CPU/MIG/Hybrid job breakdown.
"""

from __future__ import annotations
from dataclasses import dataclass, field
from typing import Optional
import statistics


# ClusterSnapshot 

@dataclass
class ClusterSnapshot:
    """One time-stamped cluster state sample."""
    time:                   float
    total_gpus:             int
    busy_gpus:              int
    free_gpus:              int
    utilization:            float   # GPU utilization [0, 1]
    power_watts:            float
    renewable_power_watts:  float = 0.0
    cpu_utilization:        float = 0.0   # CPU core utilization [0, 1]
    total_cpu_cores:        int   = 0
    free_cpu_cores:         int   = 0


# MetricsCollector 

@dataclass
class MetricsCollector:
    job_starts:       dict  = field(default_factory=dict)
    job_events:       list  = field(default_factory=list)
    snapshots:        list  = field(default_factory=list)   # list[ClusterSnapshot]
    preemption_count: int   = 0
    slo_violations:   int   = 0

    # Recording hooks 

    def record_job_start(self, job, current_time: float):
        self.job_starts[job.job_id] = current_time

    def record_job_complete(self, job, current_time: float):
        exec_time = max(1.0, (job.end_time or current_time) - (job.start_time or current_time))
        wait_time = job.queue_time if job.queue_time < float("inf") else 0.0
        tau       = 10.0
        bsld      = max((wait_time + exec_time) / max(tau, exec_time), 1.0)

        # resource_type: value string ("gpu", "cpu", "mig", "cpu_gpu")
        rt_obj = getattr(job, "resource_type", None)
        rt_str = rt_obj.value if rt_obj is not None else "gpu"

        self.job_events.append({
            "job_id":        job.job_id,
            "user_id":       job.user_id,
            "jct":           job.jct,
            "queue_time":    wait_time,
            "exec_time":     exec_time,
            "bsld":          bsld,
            "preempts":      job.preempt_count,
            "deadline":      getattr(job, "deadline", None),
            "end_time":      current_time,
            "missed_dl":     (current_time > job.deadline
                              if getattr(job, "deadline", None) else False),
            "resource_type": rt_str,
            "total_queries": getattr(job, "total_queries", 0),
        })
        self.slo_violations += getattr(job, "slo_violations", 0)

    def record_preemption(self, job):
        self.preemption_count += 1

    def record_cluster_snapshot(
        self,
        t: float,
        snap: dict,
        renewable_power_w: float = 0.0,
    ):
        # Support both old key ("utilization") and new key ("gpu_utilization")
        gpu_util = snap.get("gpu_utilization", snap.get("utilization", 0.0))
        self.snapshots.append(ClusterSnapshot(
            time=t,
            total_gpus=snap.get("total_gpus", 0),
            busy_gpus=snap.get("busy_gpus", 0),
            free_gpus=snap.get("free_gpus", 0),
            utilization=gpu_util,
            power_watts=snap.get("power_watts", 0.0),
            renewable_power_watts=renewable_power_w,
            cpu_utilization=snap.get("cpu_utilization", 0.0),
            total_cpu_cores=snap.get("total_cpu_cores", 0),
            free_cpu_cores=snap.get("free_cpu_cores", 0),
        ))

    def finalise(self, end_time: float, completed_jobs: list):
        pass

    # JCT statistics 

    def jct_stats(self) -> dict:
        jcts = sorted(e["jct"] for e in self.job_events if e["jct"] < float("inf"))
        if not jcts:
            return {"mean": 0.0, "median": 0.0, "p90": 0.0, "p99": 0.0, "std": 0.0, "n": 0}
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
        qts = sorted(e["queue_time"] for e in self.job_events
                     if e["queue_time"] < float("inf"))
        if not qts:
            return {"mean": 0.0, "median": 0.0, "p90": 0.0, "p99": 0.0}
        n = len(qts)
        return {
            "mean":   statistics.mean(qts),
            "median": statistics.median(qts),
            "p90":    qts[int(0.90 * n)],
            "p99":    qts[min(int(0.99 * n), n - 1)],
        }

    # GPU utilization 

    def avg_gpu_utilization(self) -> float:
        if not self.snapshots:
            return 0.0
        return statistics.mean(s.utilization for s in self.snapshots)

    def utilization_time_series(self) -> tuple[list, list]:
        return ([s.time for s in self.snapshots],
                [s.utilization for s in self.snapshots])

    # CPU utilization 

    def avg_cpu_utilization(self) -> float:
        if not self.snapshots:
            return 0.0
        vals = [s.cpu_utilization for s in self.snapshots]
        return statistics.mean(vals)

    # BSLD 

    def avg_bsld(self) -> float:
        bslds = [e["bsld"] for e in self.job_events if e.get("bsld") is not None]
        return statistics.mean(bslds) if bslds else 0.0

    # Renewable energy 

    def renewable_energy_utilization(self) -> float:
        if len(self.snapshots) < 2:
            return 0.0
        green_ws = total_ws = 0.0
        for i in range(1, len(self.snapshots)):
            dt         = self.snapshots[i].time - self.snapshots[i - 1].time
            avg_clust  = (self.snapshots[i].power_watts +
                          self.snapshots[i - 1].power_watts) / 2.0
            avg_green  = (self.snapshots[i].renewable_power_watts +
                          self.snapshots[i - 1].renewable_power_watts) / 2.0
            green_ws  += min(avg_green, avg_clust) * dt
            total_ws  += avg_clust * dt
        return green_ws / max(total_ws, 1.0)

    def renewable_energy_wh(self) -> float:
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

    # Fairness 

    def jains_fairness_index(self) -> float:
        user_jcts: dict[str, list] = {}
        for e in self.job_events:
            if e["jct"] < float("inf"):
                user_jcts.setdefault(e["user_id"], []).append(e["jct"])
        if not user_jcts:
            return 1.0
        means = [statistics.mean(v) for v in user_jcts.values()]
        n = len(means)
        s1, s2 = sum(means), sum(m * m for m in means)
        return (s1 * s1) / (n * s2) if s2 > 0 else 1.0

    # Deadline + SLO 

    def deadline_miss_rate(self) -> float:
        with_dl = [e for e in self.job_events if e["deadline"] is not None]
        if not with_dl:
            return 0.0
        return sum(1 for e in with_dl if e["missed_dl"]) / len(with_dl)

    def slo_violation_rate(self) -> float:
        total_q = sum(e.get("total_queries", 0) for e in self.job_events)
        return self.slo_violations / max(total_q, 1)

    # Job type breakdown 

    def cpu_jobs_completed(self) -> int:
        return sum(1 for e in self.job_events if e.get("resource_type") == "cpu")

    def mig_jobs_completed(self) -> int:
        return sum(1 for e in self.job_events if e.get("resource_type") == "mig")

    def hybrid_jobs_completed(self) -> int:
        return sum(1 for e in self.job_events if e.get("resource_type") == "cpu_gpu")

    # Summary 

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
            "avg_bsld":                     self.avg_bsld(),
            "avg_gpu_util":                 self.avg_gpu_utilization(),
            "avg_cpu_util":                 self.avg_cpu_utilization(),
            "jains_fairness":               self.jains_fairness_index(),
            "deadline_miss_pct":            self.deadline_miss_rate() * 100,
            "slo_violations":               self.slo_violations,
            "total_energy_kwh":             self.total_energy_kwh(),
            "renewable_energy_utilization": self.renewable_energy_utilization(),
            "renewable_energy_wh":          self.renewable_energy_wh(),
            "brown_energy_wh":              self.brown_energy_wh(),
            "preemptions":                  self.preemption_count,
            "cpu_jobs_completed":           self.cpu_jobs_completed(),
            "mig_jobs_completed":           self.mig_jobs_completed(),
            "hybrid_jobs_completed":        self.hybrid_jobs_completed(),
        }
