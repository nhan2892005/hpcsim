"""
hpcsim — CLI entry point.

Usage:
  hpcsim list [schedulers|clusters|traces|all]
  hpcsim simulate --scheduler SCHED --cluster CLUSTER [options]
  hpcsim benchmark --schedulers s1,s2,... [options]
  hpcsim generate --trace TRACE --output FILE
"""

from __future__ import annotations
import argparse
import sys

from .cluster.cluster import CLUSTER_CONFIGS
from .workload.generator import WorkloadGenerator, WorkloadConfig, TRACE_CONFIGS
from .scheduler.schedulers import create_scheduler, list_schedulers
from .simulator.engine import SimulationEngine
from .metrics.collector import MetricsCollector
from .cluster.cluster import Cluster
from .benchmark.runner import BenchmarkRunner, BenchmarkConfig


def cmd_list(args):
    target = args.target if hasattr(args, "target") and args.target else "all"
    if target in ("schedulers", "all"):
        print("\nAvailable Schedulers:")
        for s in list_schedulers():
            print(f"  {s}")
    if target in ("clusters", "all"):
        print("\nAvailable Cluster Configs:")
        for c in CLUSTER_CONFIGS:
            cfg = CLUSTER_CONFIGS[c]
            total = sum(n * g for _, n, g in cfg.nodes)
            print(f"  {c:<35}  {total} GPUs")
    if target in ("traces", "all"):
        print("\nAvailable Trace Presets:")
        for t in TRACE_CONFIGS:
            print(f"  {t}")
    print()


def cmd_simulate(args):
    from tabulate import tabulate

    cluster  = Cluster(CLUSTER_CONFIGS[args.cluster])
    wcfg     = WorkloadConfig(duration=args.duration)
    jobs     = WorkloadGenerator(wcfg).generate()
    sched    = create_scheduler(args.scheduler, cluster)
    metrics  = MetricsCollector()
    engine   = SimulationEngine(
        cluster, sched, jobs, metrics,
        max_sim_time=args.duration,
        verbose=args.verbose,
    )
    result = engine.run()
    summary = metrics.summary()

    print(f"\nSimulation complete: {args.scheduler} on {args.cluster}")
    rows = [[k, f"{v:.4g}" if isinstance(v, float) else str(v)] for k, v in summary.items()]
    print(tabulate(rows, headers=["Metric", "Value"], tablefmt="github"))

    if args.plot:
        from .benchmark.runner import BenchmarkRunner
        runner = BenchmarkRunner()
        runner.plot_utilization(result, args.plot)
    print()


def cmd_benchmark(args):
    scheds = [s.strip() for s in args.schedulers.split(",")]
    cfg = BenchmarkConfig(
        schedulers=scheds,
        cluster_config=args.cluster,
        num_runs=args.num_runs,
        sim_duration=args.duration,
        output_csv=args.output_csv,
        plot_file=args.plot,
        verbose=args.verbose,
    )
    runner = BenchmarkRunner(cfg)
    print(f"Running benchmark: {scheds}  ×  {args.num_runs} runs ...")
    results = runner.run()
    runner.print_table(results)


def cmd_generate(args):
    import json
    if args.trace in TRACE_CONFIGS:
        wcfg = TRACE_CONFIGS[args.trace]
    else:
        wcfg = WorkloadConfig()
    gen  = WorkloadGenerator(wcfg)
    jobs = gen.generate()
    # Simple serialisation to JSON
    data = []
    for j in jobs:
        d = {k: (v.value if hasattr(v, "value") else v)
             for k, v in j.__dict__.items()
             if not k.startswith("_")}
        d["job_type"] = j.job_type.value if hasattr(j.job_type, "value") else str(j.job_type)
        data.append(d)
    with open(args.output, "w") as f:
        json.dump(data, f, indent=2, default=str)
    print(f"Generated {len(jobs)} jobs → {args.output}")


def main():
    parser = argparse.ArgumentParser(
        prog="hpcsim",
        description="HPC GPU Cluster Simulator for ML/DL/LLM Scheduling Research",
    )
    sub = parser.add_subparsers(dest="command")

    # list
    p_list = sub.add_parser("list", help="List available schedulers/clusters/traces")
    p_list.add_argument("target", nargs="?", choices=["schedulers", "clusters", "traces", "all"], default="all")

    # simulate
    p_sim = sub.add_parser("simulate", help="Run a single simulation")
    p_sim.add_argument("--scheduler", "-s", default="fifo")
    p_sim.add_argument("--cluster",   "-c", default="medium_heterogeneous_gavel")
    p_sim.add_argument("--duration",  "-d", type=float, default=3600.0)
    p_sim.add_argument("--plot",      help="Save utilisation plot to this file")
    p_sim.add_argument("--verbose",   action="store_true")

    # benchmark
    p_bench = sub.add_parser("benchmark", help="Compare multiple schedulers")
    p_bench.add_argument("--schedulers", default="fifo,tiresias,gavel,pollux")
    p_bench.add_argument("--cluster",    default="medium_heterogeneous_gavel")
    p_bench.add_argument("--num-runs",   type=int, default=3)
    p_bench.add_argument("--duration",   type=float, default=3600.0)
    p_bench.add_argument("--output-csv", help="Save CSV to this path")
    p_bench.add_argument("--plot",       help="Save bar chart to this path")
    p_bench.add_argument("--verbose",    action="store_true")

    # generate
    p_gen = sub.add_parser("generate", help="Generate a workload trace as JSON")
    p_gen.add_argument("--trace",  default="synthetic")
    p_gen.add_argument("--output", default="workload.json")

    args = parser.parse_args()
    if args.command == "list":
        cmd_list(args)
    elif args.command == "simulate":
        cmd_simulate(args)
    elif args.command == "benchmark":
        cmd_benchmark(args)
    elif args.command == "generate":
        cmd_generate(args)
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
