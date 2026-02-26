"""
hpcsim — CLI entry point.

Install & run with uv:
  uv run hpcsim <command> [options]

Commands:
  info          Show system info, installed packages, GPU availability
  list          List available schedulers / clusters / traces
  test          Run smoke tests to verify installation
  simulate      Run a single simulation and print metrics
  benchmark     Compare multiple schedulers (table + plots)
  generate      Generate and export a workload trace (JSON / CSV)
  replay        Re-run a saved workload trace for reproducibility
  train         Train RL scheduler (MaskablePPO / GAS-MARL)
  eval          Evaluate a trained RL model
  compare       Full comparison: RL vs classical schedulers
  plot          Plot saved results CSV (learning curve / benchmark)

Examples:
  uv run hpcsim info
  uv run hpcsim list
  uv run hpcsim test
  uv run hpcsim simulate --scheduler gavel --duration 3600
  uv run hpcsim benchmark --schedulers fifo,tiresias,gavel,pollux --runs 5 --plot bench.png
  uv run hpcsim generate --duration 7200 --output workload.json
  uv run hpcsim replay --workload workload.json --scheduler tiresias
  uv run hpcsim train --algo all --epochs 300 --ckpt-interval 50
  uv run hpcsim eval --model-dir models/ --episodes 10
  uv run hpcsim compare --classical fifo,gavel,pollux --rl maskable_ppo,gas_marl
  uv run hpcsim plot --type learning-curve --input models/maskable_ppo/train_log.csv
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import sys
import time
from pathlib import Path
from textwrap import dedent
from typing import Optional

# ─── Color helpers ────────────────────────────────────────────────────────────
BOLD   = "\033[1m"
GREEN  = "\033[32m"
CYAN   = "\033[36m"
YELLOW = "\033[33m"
RED    = "\033[31m"
RESET  = "\033[0m"

def _c(text, color):
    if sys.stdout.isatty():
        return f"{color}{text}{RESET}"
    return str(text)

def _header(title: str):
    w = 58
    print(f"\n{_c('═' * w, CYAN)}")
    print(f"{_c(f'  {title}', BOLD)}")
    print(f"{_c('═' * w, CYAN)}")

def _ok(msg):   print(f"  {_c('✓', GREEN)} {msg}")
def _warn(msg): print(f"  {_c('⚠', YELLOW)} {msg}")
def _err(msg):  print(f"  {_c('✗', RED)} {msg}", file=sys.stderr)

# ─── Shared helpers ───────────────────────────────────────────────────────────

def _load_cluster_and_jobs(cluster_name, duration,
                            arrival_rate=None, gpu_dist=None,
                            seed=42, workload_file=None):
    from .cluster.cluster import Cluster, CLUSTER_CONFIGS
    from .workload.generator import WorkloadGenerator, WorkloadConfig

    if cluster_name not in CLUSTER_CONFIGS:
        _err(f"Unknown cluster '{cluster_name}'. Run 'hpcsim list clusters'.")
        sys.exit(1)

    cluster = Cluster(CLUSTER_CONFIGS[cluster_name])

    if workload_file:
        jobs = _load_workload_file(workload_file)
    else:
        kw = dict(duration=duration, rng_seed=seed)
        if arrival_rate:
            kw["mean_arrival_interval"] = arrival_rate
        if gpu_dist:
            try:
                kw["gpu_dist"] = {int(k): float(v) for k, v in
                                   (x.split(":") for x in gpu_dist.split(","))}
            except Exception:
                _warn(f"Could not parse --gpu-dist '{gpu_dist}', using defaults.")
        jobs = WorkloadGenerator(WorkloadConfig(**kw)).generate()

    return cluster, jobs


def _load_workload_file(path: str) -> list:
    """Load a workload JSON saved by cmd_generate, reconstruct job objects."""
    from .workload.job import (
        TrainingJob, InferenceJob, LLMJob, HPOJob,
        CPUJob, MIGJob, HybridJob,
        JobType, JobStatus, SchedulingMode, ModelArch, ResourceType,
    )
    from .cluster.hardware import GPUType, CPUType
    from .cluster.hardware import MIGProfile

    _JOB_CLASSES = {
        "training":       TrainingJob,
        "inference":      InferenceJob,
        "llm_train":      LLMJob,
        "llm_infer":      LLMJob,
        "hpo":            HPOJob,
        "cpu":            CPUJob,
        "mig":            MIGJob,
        "hybrid":         HybridJob,
        # by class name (from __job_class__ field)
        "TrainingJob":    TrainingJob,
        "InferenceJob":   InferenceJob,
        "LLMJob":         LLMJob,
        "HPOJob":         HPOJob,
        "CPUJob":         CPUJob,
        "MIGJob":         MIGJob,
        "HybridJob":      HybridJob,
    }
    _ENUM_FIELDS = {
        "job_type":        JobType,
        "status":          JobStatus,
        "scheduling_mode": SchedulingMode,
        "arch":            ModelArch,
        "gpu_type_preference": GPUType,
        "cpu_type_preference": CPUType,
        "mig_profile":     MIGProfile,
        "resource_type":   ResourceType,
    }

    with open(path) as f:
        data = json.load(f)

    jobs = []
    for d in data:
        # Determine job class: prefer __job_class__, fall back to job_type
        cls = _JOB_CLASSES.get(d.get("__job_class__", ""),
              _JOB_CLASSES.get(d.get("job_type", "training"), TrainingJob))

        # Convert enum string values back to enum members
        d2 = {}
        for k, v in d.items():
            if k in _ENUM_FIELDS and isinstance(v, str):
                try:
                    d2[k] = _ENUM_FIELDS[k](v)
                except (ValueError, KeyError):
                    pass   # skip unknown enum values
            else:
                d2[k] = v

        # Only pass fields that exist in the dataclass
        import dataclasses
        valid = {f.name for f in dataclasses.fields(cls)}
        kwargs = {k: v for k, v in d2.items() if k in valid}
        try:
            jobs.append(cls(**kwargs))
        except Exception:
            jobs.append(TrainingJob(submit_time=d.get("submit_time", 0.0)))

    return jobs


def _print_metrics(summary: dict, title: str = "Results"):
    from tabulate import tabulate
    _header(title)

    sections = [
        ("── Job Completion ──", [
            ["Jobs completed",  summary.get("jobs_completed", 0)],
            ["Avg JCT",         f"{summary.get('avg_jct_s', 0):.1f} s"],
            ["Median JCT",      f"{summary.get('median_jct_s', 0):.1f} s"],
            ["p90 JCT",         f"{summary.get('p90_jct_s', 0):.1f} s"],
            ["Avg queue time",  f"{summary.get('avg_queue_s', 0):.1f} s"],
            ["Avg BSLD",        f"{summary.get('avg_bsld', 0):.4f}"],
        ]),
        ("── Utilization & Fairness ──", [
            ["GPU utilization",   f"{summary.get('avg_gpu_util', 0):.1%}"],
            ["Jain's fairness",   f"{summary.get('jains_fairness', 0):.4f}"],
            ["Deadline miss",     f"{summary.get('deadline_miss_pct', 0):.1f}%"],
            ["Preemptions",       summary.get("preemptions", 0)],
        ]),
        ("── Energy ──", [
            ["ReUtil",            f"{summary.get('renewable_energy_utilization', 0):.1%}"],
            ["Total energy",      f"{summary.get('total_energy_kwh', 0):.3f} kWh"],
            ["Renewable energy",  f"{summary.get('renewable_energy_wh', 0):.0f} Wh"],
            ["Brown energy",      f"{summary.get('brown_energy_wh', 0):.0f} Wh"],
        ]),
    ]

    for sec_title, rows in sections:
        print(f"\n  {_c(sec_title, CYAN)}")
        print(tabulate(rows, tablefmt="simple", colalign=("left", "right")))
    print()


def _add_common(p, cluster=True, duration=True, seed=True):
    if cluster:
        p.add_argument("--cluster", "-c", default="medium_heterogeneous_gavel",
                       metavar="NAME", help="Cluster preset (default: medium_heterogeneous_gavel)")
    if duration:
        p.add_argument("--duration", "-d", type=float, default=3600.0,
                       metavar="SECS", help="Simulation duration in seconds (default: 3600)")
    if seed:
        p.add_argument("--seed", type=int, default=42,
                       help="Random seed (default: 42)")


# ─── Command implementations ──────────────────────────────────────────────────

def cmd_info(args):
    import platform
    _header("HPCSim System Info")

    print(f"\n  {_c('── System ──', CYAN)}")
    print(f"  Python      : {sys.version.split()[0]}")
    print(f"  Platform    : {platform.platform()}")
    print(f"  HPCSim dir  : {Path(__file__).parent}")

    print(f"\n  {_c('── Core packages ──', CYAN)}")
    for pkg in ["numpy", "pandas", "scipy", "matplotlib", "tabulate", "seaborn"]:
        try:
            v = __import__(pkg).__version__
            _ok(f"{pkg:<18} {v}")
        except ImportError:
            _warn(f"{pkg:<18} NOT installed")

    print(f"\n  {_c('── RL packages ──', CYAN)}")
    try:
        import torch
        cuda = (f"  CUDA {torch.version.cuda}  ·  {torch.cuda.get_device_name(0)}"
                if torch.cuda.is_available() else "  CPU only")
        _ok(f"torch              {torch.__version__}{cuda}")
    except ImportError:
        _warn("torch              NOT installed  →  run: uv sync --extra rl")

    print(f"\n  {_c('── Schedulers ──', CYAN)}")
    from .scheduler.schedulers import list_schedulers
    scheds = list_schedulers()
    for i in range(0, len(scheds), 3):
        print("  " + "  ".join(f"{s:<22}" for s in scheds[i:i+3]))

    print(f"\n  {_c('── Clusters ──', CYAN)}")
    from .cluster.cluster import CLUSTER_CONFIGS
    for name, cfg in CLUSTER_CONFIGS.items():
        total = sum(n.gpu_count for n in cfg.nodes)
        types = ", ".join(sorted({t for n in cfg.nodes for t in n.gpu_types}))
        print(f"  {name:<40} {total:3d} GPUs  [{types}]")
    print()


def cmd_list(args):
    from .cluster.cluster import CLUSTER_CONFIGS
    from .scheduler.schedulers import list_schedulers
    from .workload.generator import TRACE_CONFIGS
    from tabulate import tabulate

    target = getattr(args, "target", "all") or "all"

    if target in ("schedulers", "all"):
        _header("Schedulers")
        descs = {
            "fifo": "First-In-First-Out (baseline)",
            "sjf": "Shortest Job First",
            "backfill": "EASY Backfilling",
            "sjf_backfill": "SJF + Backfilling",
            "tiresias": "GPU-time LAS (Tiresias)",
            "elas": "Elastic scheduling",
            "mlfq": "Multi-Level Feedback Queue",
            "gavel": "Heterogeneity-aware (Gavel)",
            "pollux": "Adaptive allocation (Pollux)",
            "themis": "Fairness-aware (Themis)",
            "chronus": "Deadline-aware (Chronus)",
            "elastic_flow": "Elastic Flow",
            "max_min_fairness": "Max-Min Fairness",
            "maskable_ppo": "RL: MaskablePPO (needs trained model)",
            "gas_marl": "RL: GAS-MARL, green-aware (needs trained model)",
        }
        rows = [[s, descs.get(s, "")] for s in list_schedulers()]
        print(tabulate(rows, headers=["Name", "Description"], tablefmt="simple"))

    if target in ("clusters", "all"):
        _header("Clusters")
        rows = []
        for name, cfg in CLUSTER_CONFIGS.items():
            specs = cfg._normalise_nodes()
            total_gpu  = sum(s.total_gpus() for s in specs)
            total_cpu  = sum(s.total_cpu_cores() for s in specs)
            num_nodes  = sum(s.num_nodes for s in specs)
            gpu_types  = ", ".join(sorted({s.gpu_type.value for s in specs
                                           if s.gpu_type is not None})) or "—"
            has_mig    = any(s.mig_profile.value != "none" for s in specs)
            mig_tag    = " [MIG]" if has_mig else ""
            rows.append([name, num_nodes, total_gpu, gpu_types + mig_tag, total_cpu])
        print(tabulate(rows,
                       headers=["Name", "Nodes", "GPUs", "GPU Types", "CPU Cores"],
                       tablefmt="simple"))

    if target in ("traces", "all"):
        _header("Trace Presets")
        rows = [[name, getattr(cfg, "description", "")]
                for name, cfg in TRACE_CONFIGS.items()]
        print(tabulate(rows, headers=["Name", "Description"], tablefmt="simple"))
    print()


def cmd_test(args):
    _header("Smoke Tests")
    passed = 0
    failed = 0

    def run_test(name, fn):
        nonlocal passed, failed
        try:
            fn()
            _ok(name)
            passed += 1
        except Exception as e:
            _warn(f"{name}  →  {e}")
            failed += 1

    run_test("Core imports", lambda: [
        __import__("hpcsim.cluster.cluster", fromlist=["Cluster"]),
        __import__("hpcsim.scheduler.schedulers", fromlist=["create_scheduler"]),
        __import__("hpcsim.simulator.engine", fromlist=["SimulationEngine"]),
        __import__("hpcsim.metrics.collector", fromlist=["MetricsCollector"]),
    ])

    def t_cluster():
        from .cluster.cluster import Cluster, CLUSTER_CONFIGS
        c = Cluster(CLUSTER_CONFIGS["tiny_test"])
        assert c.total_gpu_count() > 0

    run_test("Build cluster (tiny_test)", t_cluster)

    def t_workload():
        from .workload.generator import WorkloadGenerator, WorkloadConfig
        jobs = WorkloadGenerator(WorkloadConfig(duration=300, rng_seed=0)).generate()
        assert len(jobs) > 0

    run_test("Generate workload (300s)", t_workload)

    def t_simulate():
        from .cluster.cluster import Cluster, CLUSTER_CONFIGS
        from .workload.generator import WorkloadGenerator, WorkloadConfig
        from .scheduler.schedulers import create_scheduler
        from .simulator.engine import SimulationEngine
        from .metrics.collector import MetricsCollector
        cluster = Cluster(CLUSTER_CONFIGS["tiny_test"])
        jobs    = WorkloadGenerator(WorkloadConfig(duration=300, rng_seed=1)).generate()
        sched   = create_scheduler("fifo", cluster)
        metrics = MetricsCollector()
        SimulationEngine(cluster, sched, jobs, metrics, max_sim_time=300).run()
        assert metrics.summary()["jobs_completed"] >= 0

    run_test("Simulate FIFO 300s", t_simulate)

    def t_gavel():
        from .cluster.cluster import Cluster, CLUSTER_CONFIGS
        from .workload.generator import WorkloadGenerator, WorkloadConfig
        from .scheduler.schedulers import create_scheduler
        from .simulator.engine import SimulationEngine
        from .metrics.collector import MetricsCollector
        cluster = Cluster(CLUSTER_CONFIGS["tiny_test"])
        jobs    = WorkloadGenerator(WorkloadConfig(duration=300, rng_seed=2)).generate()
        sched   = create_scheduler("gavel", cluster)
        metrics = MetricsCollector()
        SimulationEngine(cluster, sched, jobs, metrics, max_sim_time=300).run()
        assert metrics.summary()["jobs_completed"] >= 0

    run_test("Simulate Gavel 300s", t_gavel)

    def t_energy():
        from .energy.renewable import RenewableEnergyModule
        re = RenewableEnergyModule(total_gpus=8, sim_duration=3600)
        assert re.available_power_watts(36000) >= 0

    run_test("Renewable energy module", t_energy)

    def t_benchmark():
        from .benchmark.runner import BenchmarkRunner, BenchmarkConfig
        cfg = BenchmarkConfig(
            schedulers=["fifo", "sjf"],
            cluster_config="tiny_test",
            num_runs=1,
            sim_duration=300,
            verbose=False,
        )
        results = BenchmarkRunner(cfg).run()
        assert "fifo" in results

    run_test("Benchmark (fifo vs sjf, 1 run)", t_benchmark)

    if not getattr(args, "no_rl", False):
        def t_rl_env():
            from .rl.env import HPCGreenEnv, EnvConfig
            from .workload.generator import WorkloadConfig
            env = HPCGreenEnv(EnvConfig(
                workload_config=WorkloadConfig(duration=300, rng_seed=0),
                cluster_config="tiny_test",
                sim_duration_sec=300,
                seq_len=16,
            ))
            obs = env.reset()
            assert obs is not None and len(obs) > 0

        run_test("RL environment (HPCGreenEnv)", t_rl_env)

    print(f"\n  {'─'*40}")
    if failed == 0:
        print(f"  {_c(f'All {passed} tests passed ✓', GREEN)}")
    else:
        print(f"  {_c(f'{failed} test(s) failed', RED)} · {passed} passed")
    print()
    sys.exit(0 if failed == 0 else 1)


def cmd_simulate(args):
    from .simulator.engine import SimulationEngine
    from .metrics.collector import MetricsCollector
    from .scheduler.schedulers import create_scheduler, list_schedulers

    if args.scheduler not in list_schedulers():
        _err(f"Unknown scheduler '{args.scheduler}'.")
        _err(f"Available: {', '.join(list_schedulers())}")
        sys.exit(1)

    cluster, jobs = _load_cluster_and_jobs(
        args.cluster, args.duration,
        arrival_rate=getattr(args, "arrival_rate", None),
        gpu_dist=getattr(args, "gpu_dist", None),
        seed=args.seed,
        workload_file=getattr(args, "workload", None),
    )

    _header(f"Simulate: {args.scheduler}  on  {args.cluster}")
    print(f"  Duration : {args.duration:.0f}s  ({args.duration/3600:.2f}h)")
    print(f"  Jobs     : {len(jobs)} submitted")
    print(f"  GPUs     : {cluster.total_gpu_count()} total")

    sched   = create_scheduler(args.scheduler, cluster)
    metrics = MetricsCollector()
    rconf   = None  # engine builds RenewableEnergyModule internally with correct cluster size

    t0 = time.time()
    SimulationEngine(cluster, sched, jobs, metrics,
                     max_sim_time=args.duration,
                     renewable_config=rconf).run()
    elapsed = time.time() - t0

    summary = metrics.summary()
    _print_metrics(summary, title=f"Results: {args.scheduler}")
    print(f"  Wall-clock: {elapsed:.2f}s\n")

    if getattr(args, "output_json", None):
        Path(args.output_json).write_text(json.dumps(summary, indent=2, default=str))
        _ok(f"JSON → {args.output_json}")

    if getattr(args, "output_csv", None):
        with open(args.output_csv, "w", newline="") as f:
            w = csv.writer(f)
            w.writerow(["metric", "value"])
            for k, v in summary.items():
                w.writerow([k, v])
        _ok(f"CSV  → {args.output_csv}")

    if getattr(args, "plot", None):
        _plot_simulation(metrics, args.plot,
                         title=f"{args.scheduler} on {args.cluster}")
        _ok(f"Plot → {args.plot}")
    print()


def _plot_simulation(metrics, path: str, title: str = ""):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    try:
        # utilization_time_series() → (times: list, utils: list)
        times, utils = metrics.utilization_time_series()
        # energy data from snapshots
        snap_times  = [s.time         for s in metrics.snapshots]
        snap_power  = [s.power_watts  for s in metrics.snapshots]
        snap_green  = [s.renewable_power_watts for s in metrics.snapshots]

        fig, axes = plt.subplots(2, 1, figsize=(12, 6))
        fig.suptitle(title, fontsize=13, fontweight="bold")

        # ── GPU Utilization ─────────────────────────────────────────────────
        axes[0].fill_between(times, utils, alpha=0.6, color="#2196F3", label="GPU Util")
        axes[0].set_ylabel("GPU Utilization")
        axes[0].set_ylim(0, 1)
        if utils:
            avg = sum(utils) / len(utils)
            axes[0].axhline(avg, color="red", linestyle="--", label=f"Avg {avg:.1%}")
        axes[0].legend(fontsize=9)
        axes[0].grid(True, alpha=0.3)

        # ── Power / Energy ───────────────────────────────────────────────────
        axes[1].fill_between(snap_times, snap_green, alpha=0.5,
                             color="#4CAF50", label="Renewable (W)")
        axes[1].fill_between(snap_times, snap_power, alpha=0.3,
                             color="#FF5722", label="Consumed (W)")
        axes[1].set_xlabel("Time (s)")
        axes[1].set_ylabel("Power (W)")
        axes[1].legend(fontsize=9)
        axes[1].grid(True, alpha=0.3)

        plt.tight_layout()
        plt.savefig(path, dpi=150, bbox_inches="tight")
        plt.close()
        _ok(f"Plot saved → {path}")
    except Exception as e:
        _warn(f"Plot failed: {e}")


def cmd_benchmark(args):
    from .benchmark.runner import BenchmarkRunner, BenchmarkConfig
    from .scheduler.schedulers import list_schedulers

    scheds = [s.strip() for s in args.schedulers.split(",")]
    bad = [s for s in scheds if s not in list_schedulers()]
    if bad:
        _err(f"Unknown schedulers: {bad}")
        sys.exit(1)

    _header("Benchmark")
    print(f"  Schedulers : {', '.join(scheds)}")
    print(f"  Cluster    : {args.cluster}")
    print(f"  Runs       : {args.runs}  ·  Duration: {args.duration:.0f}s")

    cfg = BenchmarkConfig(
        schedulers=scheds,
        cluster_config=args.cluster,
        num_runs=args.runs,
        sim_duration=args.duration,
        output_csv=getattr(args, "output_csv", None),
        plot_file=getattr(args, "plot", None),
        verbose=getattr(args, "verbose", False),
        rng_seed=args.seed,
    )
    runner  = BenchmarkRunner(cfg)
    results = runner.run()
    runner.print_table(results)

    if getattr(args, "output_csv", None):
        _ok(f"CSV  → {args.output_csv}")
    if getattr(args, "plot", None):
        _ok(f"Plot → {args.plot}")
    print()


def cmd_generate(args):
    from .workload.generator import WorkloadGenerator, WorkloadConfig, TRACE_CONFIGS

    _header("Generate Workload")

    if getattr(args, "trace", None) and args.trace in TRACE_CONFIGS:
        wcfg = TRACE_CONFIGS[args.trace]
        print(f"  Preset   : {args.trace}")
    else:
        kw = dict(duration=args.duration, rng_seed=args.seed)
        if getattr(args, "arrival_rate", None):
            kw["mean_arrival_interval"] = args.arrival_rate
        if getattr(args, "gpu_dist", None):
            try:
                kw["gpu_dist"] = {int(k): float(v) for k, v in
                                   (x.split(":") for x in args.gpu_dist.split(","))}
            except Exception:
                _warn("Could not parse --gpu-dist, using defaults.")
        wcfg = WorkloadConfig(**kw)
        print(f"  Duration : {args.duration:.0f}s  ·  Seed: {args.seed}")

    t0   = time.time()
    jobs = WorkloadGenerator(wcfg).generate()
    elapsed = time.time() - t0

    output = getattr(args, "output", "workload.json")
    if output.endswith(".csv"):
        import pandas as pd
        rows = [{**{k: (v.value if hasattr(v, "value") else v)
                  for k, v in j.__dict__.items() if not k.startswith("_")},
                  "__job_class__": type(j).__name__}
                 for j in jobs]
        pd.DataFrame(rows).to_csv(output, index=False)
    else:
        def _job_to_dict(j):
            d = {k: (v.value if hasattr(v, "value") else v)
                 for k, v in j.__dict__.items() if not k.startswith("_")}
            d["__job_class__"] = type(j).__name__
            return d
        data = [_job_to_dict(j) for j in jobs]
        Path(output).write_text(json.dumps(data, indent=2, default=str))

    size = Path(output).stat().st_size
    _ok(f"Saved {len(jobs)} jobs → {output}  ({size/1024:.1f} KB, {elapsed:.2f}s)")
    print()


def cmd_replay(args):
    from .simulator.engine import SimulationEngine
    from .metrics.collector import MetricsCollector
    from .scheduler.schedulers import create_scheduler
    from .cluster.cluster import Cluster, CLUSTER_CONFIGS

    if not Path(args.workload).exists():
        _err(f"Workload file not found: {args.workload}")
        sys.exit(1)

    jobs    = _load_workload_file(args.workload)
    cluster = Cluster(CLUSTER_CONFIGS[args.cluster])
    sched   = create_scheduler(args.scheduler, cluster)
    metrics = MetricsCollector()
    dur     = args.duration or (max(j.submit_time for j in jobs) + 3600)
    rconf   = None  # engine builds RenewableEnergyModule internally with correct cluster size

    _header(f"Replay: {Path(args.workload).name}")
    print(f"  Scheduler : {args.scheduler}  ·  Cluster: {args.cluster}")
    print(f"  Jobs      : {len(jobs)}  ·  Duration: {dur:.0f}s")

    t0 = time.time()
    SimulationEngine(cluster, sched, jobs, metrics,
                     max_sim_time=dur, renewable_config=rconf).run()
    elapsed = time.time() - t0

    _print_metrics(metrics.summary(), title=f"Replay: {args.scheduler}")
    print(f"  Wall-clock: {elapsed:.2f}s\n")

    if getattr(args, "output_json", None):
        Path(args.output_json).write_text(json.dumps(metrics.summary(), indent=2, default=str))
        _ok(f"Results → {args.output_json}")
    if getattr(args, "plot", None):
        _plot_simulation(metrics, args.plot)
        _ok(f"Plot    → {args.plot}")
    print()


def cmd_train(args):
    _header("Train RL Scheduler")
    print(f"  Algorithm  : {args.algo}")
    print(f"  Epochs     : {args.epochs}  ·  Traj/epoch: {args.traj}")
    print(f"  Cluster    : {args.cluster}")
    print(f"  η (eta)    : {args.eta}  ·  Device: {args.device}")
    print(f"  Save dir   : {args.save_dir}")
    print(f"  Checkpoint : every {args.ckpt_interval} epochs  ·  Log every {args.log_interval}")
    if args.resume:
        print(f"  Resume     : {args.resume}")
    print()

    try:
        from .rl.train import run_training
    except ImportError as e:
        _err(f"RL dependencies missing: {e}")
        _err("Install with: uv sync --extra rl")
        sys.exit(1)

    resume = args.resume
    if resume == "auto":
        base  = Path(args.save_dir)
        algos = ["maskable_ppo", "gas_marl"] if args.algo == "all" else [args.algo]
        for alg in algos:
            ckpts = sorted((base / alg / "checkpoints").glob("epoch_*"))
            if ckpts:
                resume = str(ckpts[-1])
                print(f"  Auto-resume → {resume}")
                break

    run_training(
        algo=args.algo,
        epochs=args.epochs,
        traj_num=args.traj,
        cluster_config=args.cluster,
        sim_duration=args.duration,
        eta=args.eta,
        save_dir=args.save_dir,
        device=args.device,
        seed=args.seed,
        checkpoint_interval=args.ckpt_interval,
        resume_from=resume if resume != "auto" else None,
        log_interval=args.log_interval,
        save_best=not args.no_save_best,
        verbose=True,
    )
    print()


def cmd_eval(args):
    _header("Evaluate RL Model")
    print(f"  Model dir : {args.model_dir}")
    print(f"  Algorithm : {args.algo}  ·  Episodes: {args.episodes}")
    print()

    try:
        from .rl.train import run_evaluation
    except ImportError as e:
        _err(f"RL dependencies missing: {e}")
        _err("Install with: uv sync --extra rl")
        sys.exit(1)

    if not Path(args.model_dir).exists():
        _err(f"Model directory not found: {args.model_dir}")
        sys.exit(1)

    results = run_evaluation(
        model_dir=args.model_dir,
        algo=args.algo,
        n_episodes=args.episodes,
        cluster_config=args.cluster,
        sim_duration=args.duration,
        seed=args.seed,
        output_csv=getattr(args, "output_csv", None),
        verbose=True,
    )

    _header("Evaluation Summary")
    from tabulate import tabulate
    rows = [
        [alg,
         f"{r['mean_green']:.4f} ± {r['std_green']:.4f}",
         f"{r['mean_bsld']:.4f} ± {r['std_bsld']:.4f}",
         f"{r['mean_reward']:.4f} ± {r.get('std_reward', 0):.4f}"]
        for alg, r in results.items()
    ]
    print(tabulate(rows,
                   headers=["Algorithm", "ReUtil (mean±std)",
                             "AvgBSLD (mean±std)", "Reward (mean±std)"],
                   tablefmt="simple"))
    if getattr(args, "output_csv", None):
        _ok(f"Results → {args.output_csv}")
    print()


def cmd_compare(args):
    classical = [s.strip() for s in args.classical.split(",") if s.strip()]
    rl        = [s.strip() for s in args.rl.split(",")        if s.strip()]

    _header("Compare: RL vs Classical")
    print(f"  Classical  : {', '.join(classical)}")
    print(f"  RL         : {', '.join(rl)}")
    print(f"  Runs/sched : {args.runs}  ·  Duration: {args.duration:.0f}s")
    print()

    try:
        from .rl.train import run_comparison
    except ImportError as e:
        _err(f"RL dependencies missing: {e}")
        _err("Install with: uv sync --extra rl")
        sys.exit(1)

    run_comparison(
        model_dir=args.model_dir,
        cluster_config=args.cluster,
        classical_schedulers=classical,
        rl_schedulers=rl,
        num_runs=args.runs,
        sim_duration=args.duration,
        output_csv=getattr(args, "output_csv", None),
        plot_file=getattr(args, "plot", None),
        verbose=True,
    )

    if getattr(args, "output_csv", None):
        _ok(f"CSV  → {args.output_csv}")
    if getattr(args, "plot", None):
        _ok(f"Plot → {args.plot}")
    print()


def cmd_plot(args):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import pandas as pd

    if not Path(args.input).exists():
        _err(f"File not found: {args.input}")
        sys.exit(1)

    df  = pd.read_csv(args.input)
    out = args.output or (Path(args.input).stem + ".png")
    _header(f"Plot: {args.type}")

    if args.type == "learning-curve":
        required = {"epoch", "avg_reward", "avg_green", "avg_bsld"}
        if not required.issubset(df.columns):
            _err(f"Expected columns: {required}")
            sys.exit(1)

        w = max(1, len(df) // 30)
        fig, axes = plt.subplots(1, 3, figsize=(15, 5))
        fig.suptitle("RL Training Progress", fontsize=13, fontweight="bold")

        for ax, raw, color, label in [
            (axes[0], "avg_reward", "#2196F3", "Reward"),
            (axes[1], "avg_green",  "#4CAF50", "ReUtil"),
            (axes[2], "avg_bsld",   "#FF5722", "AvgBSLD"),
        ]:
            smooth = df[raw].rolling(w, center=True).mean()
            ax.plot(df["epoch"], df[raw], alpha=0.2, color=color, linewidth=0.8)
            ax.plot(df["epoch"], smooth, color=color, linewidth=2)
            last = df[raw].iloc[-1]
            ax.axhline(last, color="grey", linestyle=":", linewidth=1)
            ax.annotate(f"{last:.4f}", xy=(df["epoch"].iloc[-1], last),
                        fontsize=8, color="grey",
                        xytext=(-55, 6), textcoords="offset points")
            ax.set_xlabel("Epoch")
            ax.set_ylabel(label)
            ax.set_title(label)
            ax.grid(True, alpha=0.3)

    elif args.type == "benchmark":
        if "scheduler" not in df.columns:
            _err("Expected 'scheduler' column in benchmark CSV.")
            sys.exit(1)
        grouped = df.groupby("scheduler").mean(numeric_only=True).reset_index()
        metrics_to_plot = [
            ("avg_jct_s",                    "Avg JCT (s)",   "#2196F3"),
            ("avg_gpu_util",                 "GPU Util",      "#4CAF50"),
            ("renewable_energy_utilization", "ReUtil",        "#8BC34A"),
            ("avg_bsld",                     "Avg BSLD",      "#FF5722"),
        ]
        metrics_to_plot = [(c, l, co) for c, l, co in metrics_to_plot if c in grouped.columns]
        n = len(metrics_to_plot)
        fig, axes = plt.subplots(1, n, figsize=(5 * n, 5))
        if n == 1:
            axes = [axes]
        fig.suptitle("Benchmark Comparison", fontsize=13, fontweight="bold")
        for ax, (col, label, color) in zip(axes, metrics_to_plot):
            ax.bar(grouped["scheduler"], grouped[col], color=color, alpha=0.85, edgecolor="white")
            ax.set_title(label)
            ax.set_ylabel(label)
            ax.set_xticklabels(grouped["scheduler"], rotation=35, ha="right", fontsize=8)
            ax.grid(True, axis="y", alpha=0.3)

    else:
        _err(f"Unknown --type '{args.type}'. Use: learning-curve | benchmark")
        sys.exit(1)

    plt.tight_layout()
    plt.savefig(out, dpi=150, bbox_inches="tight")
    plt.close()
    _ok(f"Saved → {out}")
    print()


# ─── Main ─────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        prog="hpcsim",
        description=dedent("""\
            HPCSim — HPC GPU Cluster Scheduler Simulator

            Run with uv (recommended):
              uv sync                   install core dependencies
              uv sync --extra rl        install with PyTorch for RL training
              uv run hpcsim <command>   run any command

            Or after `pip install -e .`:
              hpcsim <command>
        """),
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--version", action="version", version="hpcsim 0.2.0")

    sub = parser.add_subparsers(dest="command", metavar="<command>",
                                title="Available commands")

    # info
    sub.add_parser("info",
                   help="System info, packages, GPU status")

    # list
    p_list = sub.add_parser("list", help="List schedulers / clusters / trace presets")
    p_list.add_argument("target", nargs="?",
                        choices=["schedulers", "clusters", "traces", "all"],
                        default="all")

    # test
    p_test = sub.add_parser("test", help="Smoke tests to verify installation")
    p_test.add_argument("--no-rl", action="store_true",
                        help="Skip RL env test (if torch not installed)")

    # simulate
    p_sim = sub.add_parser(
        "simulate", help="Run one simulation and show metrics",
        description="Run a single simulation. Use --workload to replay a saved trace.\n\n"
                    "Examples:\n"
                    "  hpcsim simulate --scheduler gavel\n"
                    "  hpcsim simulate --scheduler tiresias --duration 7200 --plot sim.png\n"
                    "  hpcsim simulate --workload trace.json --scheduler pollux",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p_sim.add_argument("--scheduler", "-s", default="fifo", metavar="NAME")
    _add_common(p_sim)
    p_sim.add_argument("--arrival-rate", type=float, default=None, metavar="RATE")
    p_sim.add_argument("--gpu-dist", default=None, metavar="1:0.3,2:0.3,4:0.3,8:0.1")
    p_sim.add_argument("--workload", default=None, metavar="FILE",
                       help="Replay from saved workload JSON")
    p_sim.add_argument("--no-green", action="store_true",
                       help="Disable renewable energy model")
    p_sim.add_argument("--plot",        default=None, metavar="FILE")
    p_sim.add_argument("--output-json", default=None, metavar="FILE")
    p_sim.add_argument("--output-csv",  default=None, metavar="FILE")

    # benchmark
    p_bench = sub.add_parser(
        "benchmark", help="Compare multiple schedulers (table + plots)",
        description="Run multiple schedulers on the same workloads and compare.\n\n"
                    "Examples:\n"
                    "  hpcsim benchmark --schedulers fifo,tiresias,gavel,pollux\n"
                    "  hpcsim benchmark --schedulers fifo,gavel --runs 10 --plot b.png",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p_bench.add_argument("--schedulers", "-s",
                         default="fifo,sjf,tiresias,gavel,pollux")
    _add_common(p_bench)
    p_bench.add_argument("--runs", "-n", type=int, default=3, dest="runs")
    p_bench.add_argument("--arrival-rate", type=float, default=None, metavar="RATE")
    p_bench.add_argument("--output-csv", default=None, metavar="FILE")
    p_bench.add_argument("--plot",       default=None, metavar="FILE")
    p_bench.add_argument("--verbose",    action="store_true")

    # generate
    p_gen = sub.add_parser(
        "generate", help="Generate a synthetic workload trace (JSON / CSV)",
        description="Synthesize jobs and save for reproducible replay.\n\n"
                    "Examples:\n"
                    "  hpcsim generate --output workload.json\n"
                    "  hpcsim generate --duration 86400 --arrival-rate 0.02 --output trace.csv",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p_gen.add_argument("--duration", "-d", type=float, default=86400.0)
    p_gen.add_argument("--arrival-rate", type=float, default=None, metavar="RATE")
    p_gen.add_argument("--gpu-dist", default=None, metavar="1:0.3,2:0.3,4:0.3,8:0.1")
    p_gen.add_argument("--trace",    default=None, metavar="PRESET")
    p_gen.add_argument("--output", "-o", default="workload.json", metavar="FILE")
    p_gen.add_argument("--seed", type=int, default=42)

    # replay
    p_replay = sub.add_parser(
        "replay", help="Replay a saved workload trace with any scheduler",
        description="Re-run a previously generated workload for reproducibility.\n\n"
                    "Examples:\n"
                    "  hpcsim replay --workload workload.json --scheduler gavel\n"
                    "  hpcsim replay -w trace.json -s tiresias --output-json results.json",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p_replay.add_argument("--workload",  "-w", required=True, metavar="FILE")
    p_replay.add_argument("--scheduler", "-s", default="fifo", metavar="NAME")
    _add_common(p_replay)
    p_replay.add_argument("--no-green",    action="store_true")
    p_replay.add_argument("--output-json", default=None, metavar="FILE")
    p_replay.add_argument("--plot",        default=None, metavar="FILE")

    # train
    p_train = sub.add_parser(
        "train", help="Train RL scheduler (MaskablePPO / GAS-MARL) — requires PyTorch",
        description="Train RL-based schedulers. Requires: uv sync --extra rl\n\n"
                    "Examples:\n"
                    "  hpcsim train --algo all --epochs 300\n"
                    "  hpcsim train --algo maskable_ppo --epochs 150 --ckpt-interval 25\n"
                    "  hpcsim train --algo gas_marl --resume auto\n"
                    "  hpcsim train --algo maskable_ppo --epochs 300 \\\n"
                    "    --resume models/maskable_ppo/checkpoints/epoch_0150",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p_train.add_argument("--algo",     default="all",
                         choices=["maskable_ppo", "gas_marl", "all"])
    p_train.add_argument("--epochs",   type=int,   default=300)
    p_train.add_argument("--traj",     type=int,   default=100,
                         help="Trajectories per epoch (default: 100)")
    _add_common(p_train, duration=False)
    p_train.add_argument("--duration", "-d", type=float, default=86400.0,
                         help="Episode duration in seconds (default: 86400)")
    p_train.add_argument("--eta",      type=float, default=0.005,
                         help="BSLD penalty factor η (default: 0.005)")
    p_train.add_argument("--device",   default="auto",
                         choices=["auto", "cpu", "cuda"])
    p_train.add_argument("--save-dir", default="models", metavar="DIR")
    p_train.add_argument("--ckpt-interval", type=int, default=50,
                         dest="ckpt_interval",
                         help="Save checkpoint every N epochs (0=off, default: 50)")
    p_train.add_argument("--resume",   default=None,
                         metavar="PATH|auto",
                         help="Resume from checkpoint dir, or 'auto' for latest")
    p_train.add_argument("--log-interval", type=int, default=10,
                         dest="log_interval",
                         help="Print progress every N epochs (default: 10)")
    p_train.add_argument("--no-save-best", action="store_true",
                         help="Disable saving best-reward model")

    # eval
    p_eval = sub.add_parser(
        "eval", help="Evaluate a trained RL model",
        description="Run N evaluation episodes and report metrics.\n\n"
                    "Examples:\n"
                    "  hpcsim eval --model-dir models/ --episodes 20\n"
                    "  hpcsim eval --model-dir models/maskable_ppo/checkpoints/best \\\n"
                    "    --algo maskable_ppo --output-csv eval.csv",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p_eval.add_argument("--model-dir", default="models", metavar="DIR")
    p_eval.add_argument("--algo",      default="all",
                        choices=["maskable_ppo", "gas_marl", "all"])
    p_eval.add_argument("--episodes",  type=int, default=10)
    _add_common(p_eval, duration=False)
    p_eval.add_argument("--duration",  "-d", type=float, default=86400.0)
    p_eval.add_argument("--output-csv", default=None, metavar="FILE")

    # compare
    p_cmp = sub.add_parser(
        "compare", help="RL vs classical: full comparison table + plot",
        description="Side-by-side benchmark of RL and classical schedulers.\n"
                    "Requires trained models (run 'hpcsim train' first).\n\n"
                    "Examples:\n"
                    "  hpcsim compare --classical fifo,tiresias,gavel,pollux \\\n"
                    "    --rl maskable_ppo,gas_marl\n"
                    "  hpcsim compare --classical fifo,gavel --rl gas_marl \\\n"
                    "    --runs 5 --plot comparison.png --output-csv comparison.csv",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p_cmp.add_argument("--classical",  default="fifo,tiresias,gavel,pollux")
    p_cmp.add_argument("--rl",         default="maskable_ppo,gas_marl")
    p_cmp.add_argument("--model-dir",  default="models", metavar="DIR")
    _add_common(p_cmp, duration=False)
    p_cmp.add_argument("--duration",  "-d", type=float, default=3600.0)
    p_cmp.add_argument("--runs", "-n",  type=int, default=3, dest="runs")
    p_cmp.add_argument("--output-csv", default=None, metavar="FILE")
    p_cmp.add_argument("--plot",       default=None, metavar="FILE")

    # plot
    p_plot = sub.add_parser(
        "plot", help="Plot training curves or benchmark results from CSV",
        description="Generate publication-quality figures from saved CSV data.\n\n"
                    "Examples:\n"
                    "  hpcsim plot --type learning-curve \\\n"
                    "    --input models/maskable_ppo/train_log.csv\n"
                    "  hpcsim plot --type benchmark --input results.csv --output fig.png",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p_plot.add_argument("--type",   required=True,
                        choices=["learning-curve", "benchmark"])
    p_plot.add_argument("--input",  "-i", required=True, metavar="FILE")
    p_plot.add_argument("--output", "-o", default=None,  metavar="FILE")

    # ── Dispatch ──────────────────────────────────────────────────────────────
    args = parser.parse_args()

    commands = {
        "info":      cmd_info,
        "list":      cmd_list,
        "test":      cmd_test,
        "simulate":  cmd_simulate,
        "benchmark": cmd_benchmark,
        "generate":  cmd_generate,
        "replay":    cmd_replay,
        "train":     cmd_train,
        "eval":      cmd_eval,
        "compare":   cmd_compare,
        "plot":      cmd_plot,
    }

    if args.command in commands:
        commands[args.command](args)
    else:
        parser.print_help()
        print(f"\n  Quick start:")
        print(f"  {_c('hpcsim info', CYAN)}                   system info & available schedulers")
        print(f"  {_c('hpcsim test', CYAN)}                   verify installation")
        print(f"  {_c('hpcsim simulate --scheduler gavel', CYAN)}  run a simulation")
        print(f"  {_c('hpcsim benchmark', CYAN)}               compare all classical schedulers")
        print(f"  {_c('hpcsim train --algo all', CYAN)}        train RL schedulers")
        print()


if __name__ == "__main__":
    main()