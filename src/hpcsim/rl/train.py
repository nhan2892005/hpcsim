"""
Training & Evaluation Entry Points for RL-based HPC Schedulers.

Quick Start
-----------
Train both models:
    python -m hpcsim.rl.train --algo all --epochs 300

Train specific:
    python -m hpcsim.rl.train --algo gas_marl --epochs 300 --cluster gogh_hetero

Evaluate vs classical schedulers:
    python -m hpcsim.rl.train --mode eval --model-dir models/

API:
    from hpcsim.rl.train import run_training, run_evaluation, run_comparison
"""

from __future__ import annotations
import argparse
import json
import csv
import time
from pathlib import Path
from typing import Optional
import os

# ── Lazy imports to avoid torch dependency at import time ─────────────────────

def _import_torch_deps():
    from .env import HPCGreenEnv, EnvConfig
    from .maskable_ppo import MaskablePPOAgent, train_maskable_ppo
    from .gas_marl import GASMARLAgent, train_gas_marl
    return HPCGreenEnv, EnvConfig, MaskablePPOAgent, train_maskable_ppo, GASMARLAgent, train_gas_marl


# ─── Training ─────────────────────────────────────────────────────────────────

def run_training(
    algo: str = "all",
    epochs: int = 300,
    traj_num: int = 100,
    cluster_config: str = "medium_heterogeneous_gavel",
    sim_duration: float = 86_400.0,
    eta: float = 0.005,
    save_dir: str = "models",
    device: str = "auto",
    seed: int = 42,
    verbose: bool = True,
    checkpoint_interval: int = 50,
    resume_from: Optional[str] = None,
    log_interval: int = 10,
    save_best: bool = True,
) -> dict:
    """
    Train RL scheduler(s).

    Args:
        algo:                  'maskable_ppo' | 'gas_marl' | 'all'
        epochs:                number of training epochs (default: 300)
        traj_num:              trajectories per epoch (default: 100)
        cluster_config:        cluster preset name
        sim_duration:          simulation duration per episode in seconds (default: 24h)
        eta:                   penalty factor eta for AvgBSLD in reward
        save_dir:              directory to save trained models
        device:                'cpu', 'cuda', or 'auto'
        seed:                  random seed for reproducibility
        verbose:               print training progress
        checkpoint_interval:   save checkpoint every N epochs (0 = disabled)
        resume_from:           path to checkpoint to resume from
        log_interval:          print progress every N epochs
        save_best:             save a copy of the best model

    Returns:
        dict with trained agents keyed by algorithm name

    Example:
        >>> from hpcsim.rl.train import run_training
        >>> agents = run_training(
        ...     algo='all',
        ...     epochs=300,
        ...     checkpoint_interval=50,
        ...     save_best=True,
        ... )
    """
    HPCGreenEnv, EnvConfig, MaskablePPOAgent, train_maskable_ppo, \
        GASMARLAgent, train_gas_marl = _import_torch_deps()

    from ..workload.generator import WorkloadConfig

    env_cfg = EnvConfig(
        workload_config=WorkloadConfig(
            duration=sim_duration,
            rng_seed=seed,
        ),
        cluster_config=cluster_config,
        sim_duration_sec=sim_duration,
        eta=eta,
        seed=seed,
    )

    trained = {}
    algos   = ["maskable_ppo", "gas_marl"] if algo == "all" else [algo]

    for alg in algos:
        t0  = time.time()
        out = Path(save_dir) / alg
        log = str(out / "train_log.csv")
        out.mkdir(parents=True, exist_ok=True)

        if verbose:
            print(f"\n{'='*60}")
            print(f"  Training {alg.upper()}")
            print(f"  Cluster: {cluster_config} | Epochs: {epochs} | TrajNum: {traj_num}")
            print(f"  η={eta}  SimDuration={sim_duration/3600:.1f}h  Device: {device}")
            print(f"{'='*60}")

        # Resolve resume path per-algorithm
        _resume = None
        if resume_from:
            _candidate = Path(resume_from)
            # Allow either "models/maskable_ppo/checkpoints/epoch_0100"
            # or just "models/" to auto-find latest checkpoint for this algo
            if _candidate.exists():
                _resume = str(_candidate)
            else:
                # Try save_dir/<alg>/checkpoints/<latest epoch_XXXX>
                _auto = out / "checkpoints"
                if _auto.exists():
                    ckpts = sorted(_auto.glob("epoch_*"))
                    if ckpts:
                        _resume = str(ckpts[-1])
                        print(f"  Auto-resuming {alg} from {_resume}")

        if alg == "maskable_ppo":
            agent = train_maskable_ppo(
                env_config=env_cfg,
                save_dir=str(out),
                epochs=epochs,
                traj_num=traj_num,
                device=device,
                csv_log=log,
                verbose=verbose,
                checkpoint_interval=checkpoint_interval,
                resume_from=_resume,
                log_interval=log_interval,
                save_best=save_best,
            )
        else:
            agent = train_gas_marl(
                env_config=env_cfg,
                save_dir=str(out),
                epochs=epochs,
                traj_num=traj_num,
                device=device,
                csv_log=log,
                verbose=verbose,
                checkpoint_interval=checkpoint_interval,
                resume_from=_resume,
                log_interval=log_interval,
                save_best=save_best,
            )

        elapsed = time.time() - t0
        trained[alg] = agent
        if verbose:
            print(f"  Training time: {elapsed/60:.1f} min")
            print(f"  Model saved → {out}")
            print(f"  Training log → {log}")

    # Save config for reference
    config_path = Path(save_dir) / "train_config.json"
    with open(config_path, "w") as f:
        json.dump({
            "algo": algo, "epochs": epochs, "traj_num": traj_num,
            "cluster_config": cluster_config, "sim_duration": sim_duration,
            "eta": eta, "seed": seed, "device": device,
        }, f, indent=2)

    return trained


# ─── Evaluation ───────────────────────────────────────────────────────────────

def run_evaluation(
    model_dir: str,
    algo: str = "all",
    n_episodes: int = 10,
    seq_len: int = 512,
    cluster_config: str = "medium_heterogeneous_gavel",
    sim_duration: float = 86_400.0,
    seed: int = 0,
    output_csv: Optional[str] = None,
    verbose: bool = True,
) -> dict:
    """
    Evaluate trained RL model(s) on the HPCGreenEnv.

    Returns per-episode metrics: ReUtil, AvgBSLD, reward.

    Args:
        model_dir:      directory containing saved model files
        algo:           'maskable_ppo' | 'gas_marl' | 'all'
        n_episodes:     number of evaluation episodes
        seq_len:        decisions per episode
        cluster_config: cluster preset
        sim_duration:   episode duration (seconds)
        seed:           base random seed
        output_csv:     optional path to write results CSV
        verbose:        print per-episode summary

    Returns:
        dict[algo_name] → dict with 'mean_green', 'mean_bsld', 'mean_reward'

    Example:
        >>> from hpcsim.rl.train import run_evaluation
        >>> results = run_evaluation("models/", n_episodes=10)
    """
    HPCGreenEnv, EnvConfig, MaskablePPOAgent, _, GASMARLAgent, _ = _import_torch_deps()
    from ..workload.generator import WorkloadConfig
    from .maskable_ppo import MaskablePPOAgent as MPPO
    from .gas_marl import GASMARLAgent as MARL

    algos = ["maskable_ppo", "gas_marl"] if algo == "all" else [algo]
    results = {}

    for alg in algos:
        agent_dir = Path(model_dir) / alg
        if not agent_dir.exists():
            print(f"  [WARN] No model found at {agent_dir}, skipping.")
            continue

        env_cfg = EnvConfig(
            workload_config=WorkloadConfig(duration=sim_duration, rng_seed=seed),
            cluster_config=cluster_config,
            sim_duration_sec=sim_duration,
            seq_len=seq_len,
        )
        env = HPCGreenEnv(env_cfg)

        if alg == "maskable_ppo":
            agent = MPPO(device="cpu")
            agent.load(str(agent_dir))
        else:
            agent = MARL(device="cpu")
            agent.load(str(agent_dir))

        episode_greens, episode_bslds, episode_rewards = [], [], []

        for ep in range(n_episodes):
            env.seed(seed + ep)
            obs = env.reset()
            total_reward = 0.0
            ep_green = 0.0
            ep_bsld  = 0.0
            done = False

            while not done:
                if alg == "maskable_ppo":
                    mask = env.action_mask1()
                    action = agent.eval_act(obs, mask)
                    delay  = 0
                else:
                    inv1 = 1.0 - env.action_mask1()
                    inv2 = env.action_mask2()
                    action, delay = agent.eval_action(obs, inv1, inv2)

                obs, r, done, bsld_r, *_, green_r = env.step(action, delay)
                total_reward += r
                if done:
                    ep_green = green_r
                    ep_bsld  = abs(bsld_r)

            episode_greens.append(ep_green)
            episode_bslds.append(ep_bsld)
            episode_rewards.append(total_reward)

            if verbose:
                print(f"  [{alg}] ep {ep+1:3d}/{n_episodes}  "
                      f"ReUtil={ep_green:.4f}  AvgBSLD={ep_bsld:.2f}  "
                      f"reward={total_reward:.4f}")

        import statistics
        results[alg] = {
            "mean_green":   statistics.mean(episode_greens),
            "std_green":    statistics.stdev(episode_greens) if n_episodes > 1 else 0.0,
            "mean_bsld":    statistics.mean(episode_bslds),
            "std_bsld":     statistics.stdev(episode_bslds) if n_episodes > 1 else 0.0,
            "mean_reward":  statistics.mean(episode_rewards),
            "episodes":     n_episodes,
        }
        if verbose:
            r = results[alg]
            print(f"\n  [{alg}] Summary:")
            print(f"    ReUtil:  {r['mean_green']:.4f} ± {r['std_green']:.4f}")
            print(f"    AvgBSLD: {r['mean_bsld']:.2f}  ± {r['std_bsld']:.2f}")
            print(f"    Reward:  {r['mean_reward']:.4f}")

    if output_csv:
        with open(output_csv, "w", newline="") as f:
            w = csv.writer(f)
            w.writerow(["algo", "mean_green", "std_green", "mean_bsld", "std_bsld", "mean_reward"])
            for alg, r in results.items():
                w.writerow([alg, r["mean_green"], r["std_green"],
                            r["mean_bsld"], r["std_bsld"], r["mean_reward"]])
        print(f"\n  Results saved → {output_csv}")

    return results


# ─── Full Comparison (RL + Classical) ────────────────────────────────────────

def run_comparison(
    model_dir: Optional[str] = None,
    cluster_config: str = "medium_heterogeneous_gavel",
    classical_schedulers: Optional[list] = None,
    rl_schedulers: Optional[list] = None,
    num_runs: int = 3,
    sim_duration: float = 3600.0,
    output_csv: Optional[str] = None,
    plot_file: Optional[str] = None,
    verbose: bool = True,
) -> None:
    """
    Compare RL schedulers vs classical schedulers in BenchmarkRunner.

    Renewable energy utilization is included as a metric.
    RL schedulers use pre-trained models if model_dir is provided.

    Args:
        model_dir:            directory with trained models (optional)
        cluster_config:       cluster preset name
        classical_schedulers: list of classical scheduler names
        rl_schedulers:        list of RL scheduler names (from model_dir)
        num_runs:             runs per scheduler
        sim_duration:         simulation duration per run (seconds)
        output_csv:           optional CSV output path
        plot_file:            optional plot output path

    Example:
        >>> from hpcsim.rl.train import run_comparison
        >>> run_comparison(
        ...     model_dir="models/",
        ...     classical_schedulers=["fifo", "tiresias", "gavel"],
        ...     rl_schedulers=["maskable_ppo", "gas_marl"],
        ...     num_runs=3,
        ... )
    """
    from ..benchmark.runner import BenchmarkRunner, BenchmarkConfig
    from ..cluster.cluster import Cluster, CLUSTER_CONFIGS

    if classical_schedulers is None:
        classical_schedulers = ["fifo", "sjf", "tiresias", "gavel", "pollux"]
    if rl_schedulers is None:
        rl_schedulers = []

    # Register RL schedulers in the scheduler registry
    if rl_schedulers and model_dir:
        _register_rl_schedulers(model_dir, rl_schedulers, cluster_config)

    all_schedulers = classical_schedulers + rl_schedulers

    cfg = BenchmarkConfig(
        schedulers=all_schedulers,
        cluster_config=cluster_config,
        num_runs=num_runs,
        sim_duration=sim_duration,
        output_csv=output_csv,
        plot_file=plot_file,
        verbose=verbose,
    )

    runner  = BenchmarkRunner(cfg)
    print(f"\nRunning comparison: {all_schedulers}  ×  {num_runs} runs")
    results = runner.run()
    runner.print_table(results)

    # Print renewable energy utilization summary
    _print_green_summary(results, verbose)


def _register_rl_schedulers(model_dir: str, names: list[str], cluster_cfg: str):
    """Register RL schedulers so create_scheduler can find them."""
    from ..scheduler.schedulers import _REGISTRY
    from ..cluster.cluster import Cluster, CLUSTER_CONFIGS

    for name in names:
        if name not in _REGISTRY:
            # Create factory that loads the model
            _model_path = str(Path(model_dir) / name)

            def make_factory(mpath, alg_name):
                def factory(cluster):
                    if alg_name == "maskable_ppo":
                        from .maskable_ppo import MaskablePPOScheduler
                        return MaskablePPOScheduler(cluster, model_dir=mpath)
                    elif alg_name == "gas_marl":
                        from .gas_marl import GASMARLScheduler
                        return GASMARLScheduler(cluster, model_dir=mpath)
                return factory

            _REGISTRY[name] = make_factory(_model_path, name)


def _print_green_summary(results, verbose: bool):
    """Print renewable energy utilization from benchmark results."""
    if not verbose:
        return
    print("\n  Renewable Energy Utilization:")
    print(f"  {'Scheduler':<20} {'ReUtil (avg)':>14}")
    print("  " + "-" * 36)
    for name, agg in results.items():
        util = agg.mean.get("renewable_energy_utilization", 0.0)
        print(f"  {name:<20} {util:>13.1%}")
    print()


# ─── CLI ─────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        prog="python -m hpcsim.rl.train",
        description="Train and evaluate RL-based green-aware HPC schedulers",
    )
    sub = parser.add_subparsers(dest="mode", help="Mode: train | eval | compare")

    # ── train ──
    p_train = sub.add_parser("train", help="Train RL scheduler")
    p_train.add_argument("--algo",    default="all",
                         choices=["maskable_ppo", "gas_marl", "all"])
    p_train.add_argument("--epochs",  type=int,   default=300)
    p_train.add_argument("--traj",    type=int,   default=100,
                         help="Trajectories per epoch")
    p_train.add_argument("--cluster", default="medium_heterogeneous_gavel")
    p_train.add_argument("--duration",type=float, default=86400.0,
                         help="Episode duration in seconds (default 24h)")
    p_train.add_argument("--eta",     type=float, default=0.005,
                         help="AvgBSLD penalty factor η")
    p_train.add_argument("--save-dir",default="models")
    p_train.add_argument("--device",   default="auto")
    p_train.add_argument("--seed",     type=int,   default=42)
    p_train.add_argument("--ckpt-interval", type=int, default=50,
                         dest="ckpt_interval",
                         help="Save checkpoint every N epochs (0=off, default 50)")
    p_train.add_argument("--resume",   default=None,
                         help="Resume from checkpoint dir or 'auto' for latest")
    p_train.add_argument("--log-interval", type=int, default=10,
                         dest="log_interval",
                         help="Print progress every N epochs (default 10)")
    p_train.add_argument("--no-save-best", action="store_false", dest="save_best",
                         help="Disable saving best-reward model")

    # ── eval ──
    p_eval = sub.add_parser("eval", help="Evaluate trained model")
    p_eval.add_argument("--model-dir", default="models")
    p_eval.add_argument("--algo",      default="all",
                        choices=["maskable_ppo", "gas_marl", "all"])
    p_eval.add_argument("--episodes",  type=int,   default=10)
    p_eval.add_argument("--cluster",   default="medium_heterogeneous_gavel")
    p_eval.add_argument("--duration",  type=float, default=86400.0)
    p_eval.add_argument("--output-csv",default=None)

    # ── compare ──
    p_cmp = sub.add_parser("compare", help="Compare RL vs classical schedulers")
    p_cmp.add_argument("--model-dir",   default="models")
    p_cmp.add_argument("--classical",   default="fifo,tiresias,gavel,pollux,chronus",
                       help="Comma-separated classical schedulers")
    p_cmp.add_argument("--rl",          default="maskable_ppo,gas_marl",
                       help="Comma-separated RL schedulers")
    p_cmp.add_argument("--cluster",     default="medium_heterogeneous_gavel")
    p_cmp.add_argument("--num-runs",    type=int,   default=3)
    p_cmp.add_argument("--duration",    type=float, default=3600.0)
    p_cmp.add_argument("--output-csv",  default=None)
    p_cmp.add_argument("--plot",        default=None)

    args = parser.parse_args()

    if args.mode == "train":
        resume_val = getattr(args, "resume", None)
        if resume_val == "auto":
            resume_val = None  # run_training will auto-find latest ckpt
        run_training(
            algo=args.algo, epochs=args.epochs, traj_num=args.traj,
            cluster_config=args.cluster, sim_duration=args.duration,
            eta=args.eta, save_dir=args.save_dir, device=args.device,
            seed=args.seed,
            checkpoint_interval=getattr(args, "ckpt_interval", 50),
            resume_from=resume_val,
            log_interval=getattr(args, "log_interval", 10),
            save_best=getattr(args, "save_best", True),
        )

    elif args.mode == "eval":
        run_evaluation(
            model_dir=args.model_dir, algo=args.algo,
            n_episodes=args.episodes, cluster_config=args.cluster,
            sim_duration=args.duration, output_csv=args.output_csv,
        )

    elif args.mode == "compare":
        run_comparison(
            model_dir=args.model_dir,
            classical_schedulers=args.classical.split(","),
            rl_schedulers=args.rl.split(","),
            cluster_config=args.cluster,
            num_runs=args.num_runs,
            sim_duration=args.duration,
            output_csv=args.output_csv,
            plot_file=args.plot,
        )
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
