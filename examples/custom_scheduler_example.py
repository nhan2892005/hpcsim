"""
Custom Scheduler Example — GOGH-inspired correlation-guided scheduler.

GOGH [Raeisi et al., 2025]: assign jobs to the GPU type that maximises
estimated throughput based on (model, GPU) correlation matrix.

Run:
  cd hpc-gpu-sim
  PYTHONPATH=src python examples/custom_scheduler_example.py
"""

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from hpcsim.scheduler.schedulers import BaseScheduler, SchedulingDecision, register
from hpcsim.workload.job import solo_throughput, ModelArch, GPUType
from hpcsim import Cluster, CLUSTER_CONFIGS, WorkloadGenerator, WorkloadConfig
from hpcsim import SimulationEngine, MetricsCollector, create_scheduler


@register
class GOGHInspiredScheduler(BaseScheduler):
    """
    GOGH-inspired: route each job to the GPU type yielding
    highest throughput. Among same-type candidates prefer
    consolidated placement for NVLink bandwidth.
    """
    name = "gogh"

    def _best_type_for_job(self, job):
        arch = getattr(job, "arch", ModelArch.RESNET50)
        bs   = getattr(job, "batch_size", 64)
        n    = getattr(job, "num_gpus_requested", 1)
        free = self.cluster.free_gpus_by_type()
        best_tp, best_type = 0.0, None
        for gtype, gpus in free.items():
            if len(gpus) < n:
                continue
            tp = solo_throughput(arch, gtype, bs)
            if tp > best_tp:
                best_tp, best_type = tp, gtype
        return best_type

    def schedule(self, pending, running, current_time):
        decision = SchedulingDecision()
        for job in sorted(pending, key=lambda j: -solo_throughput(
            getattr(j, "arch", ModelArch.RESNET50),
            GPUType.V100,
            getattr(j, "batch_size", 64),
        )):
            best_type = self._best_type_for_job(job)
            gpu_ids = self._find_gpus(job, prefer_consolidated=True, gpu_type=best_type)
            if not gpu_ids:
                gpu_ids = self._find_gpus(job)
            if gpu_ids:
                decision.add(job, gpu_ids)
        return decision


def compare():
    print("GOGH-Inspired vs Standard Schedulers")
    print("=" * 60)

    wcfg = WorkloadConfig(duration=1800.0, rng_seed=42)
    jobs = WorkloadGenerator(wcfg).generate()

    for name in ["fifo", "sjf", "tiresias", "gavel", "pollux", "chronus", "gogh"]:
        cl    = Cluster(CLUSTER_CONFIGS["gogh_hetero"])
        sched = create_scheduler(name, cl)
        m     = MetricsCollector()
        eng   = SimulationEngine(cl, sched, list(jobs), m, max_sim_time=1800.0)
        eng.run()
        s = m.summary()
        print(f"  {name:10s}  completed={s['jobs_completed']:3d}  "
              f"avg_jct={s['avg_jct_s']:7.1f}s  "
              f"util={s['avg_gpu_util']:5.1%}  "
              f"fairness={s['jains_fairness']:.3f}")


if __name__ == "__main__":
    compare()