"""
Full example: single simulation + benchmark comparison.

Run:
  cd hpc-gpu-sim
  PYTHONPATH=src python examples/example_simulation.py
"""

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from hpcsim import (
    Cluster, CLUSTER_CONFIGS,
    WorkloadGenerator, WorkloadConfig,
    create_scheduler,
    SimulationEngine, MetricsCollector,
    BenchmarkRunner, BenchmarkConfig,
)


def run_single():
    print("=" * 60)
    print("  SINGLE SIMULATION DEMO")
    print("=" * 60)

    cluster = Cluster(CLUSTER_CONFIGS["medium_heterogeneous_gavel"])
    print(f"\nCluster: {cluster}")
    print(f"GPU types: {set(g.gpu_type.value for g in cluster.gpus.values())}")

    wcfg = WorkloadConfig(duration=1800.0, rng_seed=42)
    jobs = WorkloadGenerator(wcfg).generate()
    print(f"\nGenerated {len(jobs)} jobs (1800s window)")

    for sched_name in ["fifo", "tiresias", "gavel", "pollux"]:
        cl    = Cluster(CLUSTER_CONFIGS["medium_heterogeneous_gavel"])
        sched = create_scheduler(sched_name, cl)
        metrics = MetricsCollector()
        engine  = SimulationEngine(cl, sched, list(jobs), metrics, max_sim_time=1800.0)
        result  = engine.run()
        s = metrics.summary()
        print(f"  [{sched_name:16s}]  "
              f"completed={s['jobs_completed']:3d}  "
              f"avg_jct={s['avg_jct_s']:7.1f}s  "
              f"util={s['avg_gpu_util']:5.1%}  "
              f"fairness={s['jains_fairness']:.3f}")


def run_benchmark():
    print("\n" + "=" * 60)
    print("  BENCHMARK COMPARISON")
    print("=" * 60)

    cfg = BenchmarkConfig(
        schedulers=["fifo", "sjf", "tiresias", "gavel", "pollux", "chronus"],
        cluster_config="medium_heterogeneous_gavel",
        num_runs=2,
        sim_duration=1800.0,
        plot_file="./outputs/benchmark_results.png",
    )
    runner  = BenchmarkRunner(cfg)
    results = runner.run()
    runner.print_table(results)
    return results


if __name__ == "__main__":
    run_single()
    run_benchmark()
    print("\nDone!  See ./outputs/benchmark_results.png")