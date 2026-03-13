"""
Green-Aware HPC Scheduling — RL Integration Example.

Demonstrates:
  1. Renewable energy module (solar + wind time series)
  2. Simulation with green energy tracking + ReUtil metric
  3. Training MaskablePPO and GAS-MARL agents
  4. Full comparison: classical + RL schedulers with ReUtil metric

Run:
    cd hpc-gpu-sim
    PYTHONPATH=src python examples/green_rl_example.py
"""

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

# 1. Demonstrate renewable energy module 
def demo_renewable():
    print("=" * 60)
    print("  RENEWABLE ENERGY MODULE DEMO")
    print("=" * 60)
    from hpcsim.energy.renewable import RenewableEnergyModule
    re = RenewableEnergyModule(total_gpus=112, sim_duration=86_400.0)
    print(f"\n  Cluster: 112 GPUs | 24-hour simulation")
    print(f"\n  Hourly renewable power profile (first 24 hours):")
    print(f"  {'Hour':>5}  {'Solar+Wind (kW)':>18}  {'vs Cluster Idle':>18}")
    idle_kw = 112 * 25 / 1000
    for h in range(0, 24, 2):
        pw = re.available_power_watts(h * 3600) / 1000
        bar = "█" * int(pw / 3)
        print(f"  {h:02d}:00  {pw:>14.1f} kW  {pw/idle_kw:>14.1f}×  {bar}")
    print()


# 2. Classical simulation with ReUtil 
def demo_simulation_with_green():
    print("=" * 60)
    print("  SIMULATION WITH GREEN ENERGY METRICS")
    print("=" * 60)
    from hpcsim import (
        Cluster, CLUSTER_CONFIGS,
        WorkloadGenerator, WorkloadConfig,
        create_scheduler, SimulationEngine, MetricsCollector,
    )

    wcfg = WorkloadConfig(duration=3600.0, rng_seed=42)
    jobs = WorkloadGenerator(wcfg).generate()
    print(f"\n  Generated {len(jobs)} jobs")

    header = f"  {'Scheduler':<16}  {'Jobs':>5}  {'AvgJCT':>10}  {'AvgBSLD':>9}  {'GPUUtil':>8}  {'ReUtil':>8}"
    print(f"\n{header}")
    print("  " + "-" * 72)

    for sched_name in ["fifo", "sjf", "tiresias", "gavel", "pollux", "chronus"]:
        cl    = Cluster(CLUSTER_CONFIGS["medium_heterogeneous_gavel"])
        sched = create_scheduler(sched_name, cl)
        m     = MetricsCollector()
        eng   = SimulationEngine(cl, sched, list(jobs), m, max_sim_time=3600.0)
        eng.run()
        s = m.summary()
        print(
            f"  {sched_name:<16}  {s['jobs_completed']:>5}  "
            f"{s['avg_jct_s']:>10.1f}  {s['avg_bsld']:>9.2f}  "
            f"{s['avg_gpu_util']:>7.1%}  {s['renewable_energy_utilization']:>7.1%}"
        )
    print()


# 3. RL Environment quick test 
def demo_rl_env():
    print("=" * 60)
    print("  RL ENVIRONMENT DEMO")
    print("=" * 60)

    try:
        import torch
    except ImportError:
        print("\n  [SKIP] PyTorch not installed.")
        print("  Install: pip install torch scipy")
        return

    from hpcsim.rl.env import HPCGreenEnv, EnvConfig
    from hpcsim.workload.generator import WorkloadConfig

    env_cfg = EnvConfig(
        workload_config=WorkloadConfig(duration=3600.0, rng_seed=0),
        sim_duration_sec=3600.0,
        seq_len=50,
    )
    env = HPCGreenEnv(env_cfg)
    obs = env.reset()
    print(f"\n  Env obs shape: {obs.shape}")
    print(f"  Cluster GPUs:  {env._total_gpus}")
    print(f"  Pending jobs:  {len(env._pending)}")

    total_r = 0.0
    steps   = 0
    while True:
        mask   = env.action_mask1()
        action = int(mask.argmax())  # always pick first valid job
        obs, r, done, bsld_r, *_, green_r = env.step(action, 0)
        total_r += r
        steps   += 1
        if done:
            print(f"  Completed {steps} steps | ReUtil={green_r:.3f} | AvgBSLD={abs(bsld_r):.2f}")
            break
    print()


# 4. Training example (few epochs for demo) 
def demo_training():
    print("=" * 60)
    print("  RL TRAINING DEMO (10 epochs — for demo only)")
    print("  For production: epochs=300, traj_num=100")
    print("=" * 60)

    try:
        import torch
    except ImportError:
        print("\n  [SKIP] PyTorch not installed. pip install torch scipy")
        return

    from hpcsim.rl.train import run_training

    agents = run_training(
        algo="all",
        epochs=10,           # use 300 for real training
        traj_num=5,          # use 100 for real training
        cluster_config="medium_heterogeneous_gavel",
        sim_duration=3600.0,
        eta=0.002,
        save_dir="./models_demo",
        device="auto",
        seed=42,
        verbose=True,
    )
    print(f"\n  Trained: {list(agents.keys())}")
    print()


# 5. Full benchmark comparison 
def demo_comparison():
    print("=" * 60)
    print("  FULL BENCHMARK COMPARISON (classical only)")
    print("  (Add RL schedulers after training)")
    print("=" * 60)

    from hpcsim import BenchmarkRunner, BenchmarkConfig

    cfg = BenchmarkConfig(
        schedulers=["fifo", "sjf", "tiresias", "gavel", "pollux", "chronus"],
        cluster_config="medium_heterogeneous_gavel",
        num_runs=2,
        sim_duration=1800.0,
        plot_file="./outputs/green_benchmark.png",
    )
    runner  = BenchmarkRunner(cfg)
    results = runner.run()
    runner.print_table(results)

    # ReUtil ranking
    sorted_re = sorted(
        results.items(),
        key=lambda x: x[1].mean.get("renewable_energy_utilization", 0),
        reverse=True,
    )
    print("  ReUtil ranking:")
    for rank, (name, agg) in enumerate(sorted_re, 1):
        util = agg.mean.get("renewable_energy_utilization", 0)
        bsld = agg.mean.get("avg_bsld", 0)
        print(f"    {rank}. {name:<16}  ReUtil={util:.1%}  AvgBSLD={bsld:.2f}")
    print()


# Run all demos 
if __name__ == "__main__":
    import os
    os.makedirs("./outputs", exist_ok=True)

    demo_renewable()
    demo_simulation_with_green()
    demo_rl_env()
    demo_training()
    demo_comparison()

    print("Done!")
    print("  benchmark plot → ./outputs/green_benchmark.png")
    print("  trained models → ./models_demo/")
