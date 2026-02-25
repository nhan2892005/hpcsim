"""
Benchmark Runner — compare multiple schedulers over multiple runs.

Features:
- Multi-scheduler comparison with statistical analysis
- Multiple independent runs per scheduler (variance analysis)
- Normalised metrics relative to FIFO baseline
- Tabulate tables with results
- Matplotlib plots: comparison bar chart, utilisation timeline
- CSV export
"""

from __future__ import annotations
from dataclasses import dataclass, field
from typing import Optional
import statistics
import random
import csv
import io

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from tabulate import tabulate

from ..cluster.cluster import Cluster, CLUSTER_CONFIGS
from ..workload.generator import WorkloadGenerator, WorkloadConfig, TRACE_CONFIGS
from ..scheduler.schedulers import create_scheduler, list_schedulers
from ..simulator.engine import SimulationEngine, SimulationResult
from ..metrics.collector import MetricsCollector


@dataclass
class BenchmarkConfig:
    schedulers:    list[str] = field(default_factory=lambda: ["fifo", "tiresias", "gavel", "pollux"])
    cluster_config: str = "medium_heterogeneous_gavel"
    trace:         str = ""
    workload_config: Optional[WorkloadConfig] = None
    num_runs:      int = 3
    sim_duration:  float = 3600.0
    rng_seed:      int = 42
    output_csv:    Optional[str] = None
    plot_file:     Optional[str] = None
    verbose:       bool = False


@dataclass
class RunResult:
    scheduler_name: str
    run_idx:        int
    summary:        dict
    result:         SimulationResult


@dataclass
class AggregateResult:
    scheduler_name: str
    runs:           list[RunResult]
    mean:           dict
    std:            dict


def _make_workload(cfg: BenchmarkConfig, seed: int) -> tuple[Cluster, list]:
    cluster = Cluster(CLUSTER_CONFIGS[cfg.cluster_config])
    if cfg.trace and cfg.trace in TRACE_CONFIGS:
        wcfg = TRACE_CONFIGS[cfg.trace]
    else:
        wcfg = cfg.workload_config or WorkloadConfig(duration=cfg.sim_duration)
    wcfg_seeded = WorkloadConfig(
        **{**wcfg.__dict__, "rng_seed": seed, "duration": cfg.sim_duration}
    )
    jobs = WorkloadGenerator(wcfg_seeded).generate()
    return cluster, jobs


class BenchmarkRunner:
    """Benchmark multiple schedulers with statistical significance."""

    def __init__(self, config: Optional[BenchmarkConfig] = None):
        self.config = config or BenchmarkConfig()

    def run(self) -> dict[str, AggregateResult]:
        cfg = self.config
        schedulers = cfg.schedulers or list_schedulers()
        results: dict[str, list[RunResult]] = {s: [] for s in schedulers}
        base_seeds = [42 + i * 100 for i in range(cfg.num_runs)]

        for run_idx, seed in enumerate(base_seeds):
            for sched_name in schedulers:
                cluster, jobs = _make_workload(cfg, seed)
                scheduler = create_scheduler(sched_name, cluster)
                metrics = MetricsCollector()
                engine = SimulationEngine(
                    cluster, scheduler, jobs, metrics,
                    max_sim_time=cfg.sim_duration,
                    verbose=cfg.verbose,
                )
                sim_result = engine.run()
                summary = metrics.summary()
                results[sched_name].append(RunResult(
                    scheduler_name=sched_name,
                    run_idx=run_idx,
                    summary=summary,
                    result=sim_result,
                ))
                if cfg.verbose:
                    print(f"  [{sched_name}] run {run_idx+1}/{cfg.num_runs} "
                          f"→ avg_jct={summary['avg_jct_s']:.1f}s  "
                          f"util={summary['avg_gpu_util']:.1%}")

        # Aggregate
        aggregated: dict[str, AggregateResult] = {}
        keys = list(results[schedulers[0]][0].summary.keys())
        for sched_name, runs in results.items():
            mean_d, std_d = {}, {}
            for k in keys:
                vals = [r.summary[k] for r in runs if r.summary.get(k, 0) is not None]
                mean_d[k] = statistics.mean(vals) if vals else 0.0
                std_d[k]  = statistics.stdev(vals) if len(vals) > 1 else 0.0
            aggregated[sched_name] = AggregateResult(
                scheduler_name=sched_name, runs=runs, mean=mean_d, std=std_d
            )

        if cfg.output_csv:
            self._write_csv(aggregated, cfg.output_csv)
        if cfg.plot_file:
            self.plot_comparison(aggregated, cfg.plot_file)

        return aggregated

    # ── Display ───────────────────────────────────────────────────────────────

    def print_table(self, results: dict[str, AggregateResult]):
        headers = [
            "Scheduler", "Jobs", "Avg JCT (s)", "Median JCT (s)",
            "p90 JCT (s)", "Avg Queue (s)", "AvgBSLD",
            "GPU Util %", "ReUtil %",
            "Fairness", "DL Miss %", "Energy (kWh)", "Preemptions",
        ]
        rows = []
        for name, agg in results.items():
            m = agg.mean
            rows.append([
                name,
                f"{m.get('jobs_completed', 0):.0f}",
                f"{m.get('avg_jct_s', 0):.1f} ±{agg.std.get('avg_jct_s', 0):.1f}",
                f"{m.get('median_jct_s', 0):.1f}",
                f"{m.get('p90_jct_s', 0):.1f}",
                f"{m.get('avg_queue_s', 0):.1f}",
                f"{m.get('avg_bsld', 0):.2f}",
                f"{m.get('avg_gpu_util', 0)*100:.1f}%",
                f"{m.get('renewable_energy_utilization', 0)*100:.1f}%",   # ← NEW
                f"{m.get('jains_fairness', 0):.3f}",
                f"{m.get('deadline_miss_pct', 0):.1f}%",
                f"{m.get('total_energy_kwh', 0):.2f}",
                f"{m.get('preemptions', 0):.0f}",
            ])
        print()
        print("=" * 120)
        print(f"  BENCHMARK RESULTS  —  cluster: {self.config.cluster_config}  "
              f"runs: {self.config.num_runs}  duration: {self.config.sim_duration}s")
        print("=" * 120)
        print(tabulate(rows, headers=headers, tablefmt="github"))
        print()

        # Normalised relative to FIFO
        if "fifo" in results:
            fifo_jct = results["fifo"].mean.get("avg_jct_s", 1.0)
            fifo_re  = results["fifo"].mean.get("renewable_energy_utilization", 0.01)
            print("  Normalised Avg JCT (FIFO = 1.00)  |  ReUtil relative to FIFO:")
            norm_rows = []
            for name, agg in results.items():
                norm_jct = agg.mean.get("avg_jct_s", 0) / max(fifo_jct, 1e-9)
                re_util  = agg.mean.get("renewable_energy_utilization", 0)
                norm_re  = re_util / max(fifo_re, 1e-9)
                norm_rows.append([name, f"{norm_jct:.3f}", f"{re_util:.1%}", f"{norm_re:.2f}x"])
            print(tabulate(norm_rows, headers=["Scheduler", "Norm JCT", "ReUtil", "vs FIFO"], tablefmt="github"))
            print()

    def plot_comparison(
        self,
        results: dict[str, AggregateResult],
        output_path: str = "benchmark_results.png",
    ):
        sched_names = list(results.keys())
        n = len(sched_names)
        metrics_to_plot = [
            ("avg_jct_s",                    "Avg JCT (s)",          "lower"),
            ("avg_gpu_util",                  "GPU Utilisation",      "higher"),
            ("renewable_energy_utilization",  "ReUtil (Green %)",     "higher"),
            ("avg_bsld",                      "Avg Bounded Slowdown", "lower"),
        ]

        fig, axes = plt.subplots(2, 2, figsize=(14, 9))
        fig.suptitle("HPC GPU Cluster Scheduler Benchmark", fontsize=14, fontweight="bold")
        axes = axes.flatten()

        colours = plt.cm.tab10(np.linspace(0, 1, n))

        for ax, (metric, label, direction) in zip(axes, metrics_to_plot):
            means = [results[s].mean.get(metric, 0) for s in sched_names]
            stds  = [results[s].std.get(metric, 0)  for s in sched_names]
            bars = ax.bar(range(n), means, color=colours, yerr=stds, capsize=5,
                          edgecolor="black", linewidth=0.5, alpha=0.85)
            ax.set_xticks(range(n))
            ax.set_xticklabels(sched_names, rotation=35, ha="right", fontsize=8)
            ax.set_title(f"{label} ({'↓' if direction == 'lower' else '↑'} better)")
            ax.set_ylabel(label)
            ax.grid(axis="y", alpha=0.3)
            # Annotate bars
            for bar, val in zip(bars, means):
                h = bar.get_height()
                ax.text(bar.get_x() + bar.get_width() / 2.0, h * 1.01,
                        f"{val:.2g}", ha="center", va="bottom", fontsize=7)

        plt.tight_layout()
        plt.savefig(output_path, dpi=150, bbox_inches="tight")
        plt.close()
        print(f"  Plot saved → {output_path}")

    def plot_utilization(
        self,
        result: SimulationResult,
        output_path: str = "utilization.png",
    ):
        times, utils = result.metrics.utilization_time_series()
        if not times:
            return
        fig, ax = plt.subplots(figsize=(12, 4))
        ax.plot(times, [u * 100 for u in utils], lw=1.2, color="#2563EB")
        ax.fill_between(times, [u * 100 for u in utils], alpha=0.2, color="#2563EB")
        ax.set_xlabel("Simulation Time (s)")
        ax.set_ylabel("GPU Utilisation (%)")
        ax.set_title(f"GPU Utilisation — {result.scheduler_name} on {result.cluster_name}")
        ax.set_ylim(0, 105)
        ax.axhline(y=sum(u * 100 for u in utils) / len(utils), color="red",
                   linestyle="--", lw=1, label="Mean")
        ax.legend()
        ax.grid(alpha=0.3)
        plt.tight_layout()
        plt.savefig(output_path, dpi=150, bbox_inches="tight")
        plt.close()
        print(f"  Utilisation plot saved → {output_path}")

    def _write_csv(self, results: dict[str, AggregateResult], path: str):
        if not results:
            return
        keys = list(next(iter(results.values())).mean.keys())
        with open(path, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["scheduler"] + keys + [f"{k}_std" for k in keys])
            for name, agg in results.items():
                row = [name] + [agg.mean.get(k, 0) for k in keys] + [agg.std.get(k, 0) for k in keys]
                writer.writerow(row)
        print(f"  CSV saved → {path}")
