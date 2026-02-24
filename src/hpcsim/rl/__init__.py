"""
hpcsim.rl — Reinforcement Learning Schedulers for Green-Aware HPC Scheduling.

Schedulers:
  MaskablePPOScheduler  — Single-action PPO with job-selection masking
  GASMARLScheduler      — Multi-action PPO (job selection + delay decision)

Environment:
  HPCGreenEnv           — Step-based RL training environment with renewable energy

Training:
  train_maskable_ppo()  — Train MaskablePPO agent
  train_gas_marl()      — Train GAS-MARL agent
  run_training()        — Train via CLI-style API
  run_evaluation()      — Evaluate trained models
  run_comparison()      — Compare RL + classical schedulers

Quick start:
    # Train
    from hpcsim.rl.train import run_training
    agents = run_training(algo='all', epochs=300, traj_num=100)

    # Use in simulation
    from hpcsim import create_scheduler, Cluster, CLUSTER_CONFIGS
    from hpcsim.rl.maskable_ppo import MaskablePPOScheduler
    cluster   = Cluster(CLUSTER_CONFIGS['medium_heterogeneous_gavel'])
    scheduler = MaskablePPOScheduler(cluster, model_dir='models/maskable_ppo')

    # CLI
    python -m hpcsim.rl.train train --algo all --epochs 300
    python -m hpcsim.rl.train eval  --model-dir models/
    python -m hpcsim.rl.train compare --model-dir models/ --classical fifo,gavel,pollux

Note:
    PyTorch is required for RL agent training/inference.
    The env module (HPCGreenEnv) works without PyTorch.
    Install: pip install torch scipy
"""

from .env import HPCGreenEnv, EnvConfig
from .train import run_training, run_evaluation, run_comparison

# Lazy imports — require PyTorch
def __getattr__(name):
    if name in ("MaskablePPOAgent", "MaskablePPOScheduler", "train_maskable_ppo"):
        from .maskable_ppo import MaskablePPOAgent, MaskablePPOScheduler, train_maskable_ppo
        return locals()[name]
    if name in ("GASMARLAgent", "GASMARLScheduler", "train_gas_marl"):
        from .gas_marl import GASMARLAgent, GASMARLScheduler, train_gas_marl
        return locals()[name]
    raise AttributeError(f"module 'hpcsim.rl' has no attribute {name!r}")

__all__ = [
    "HPCGreenEnv", "EnvConfig",
    "MaskablePPOAgent", "MaskablePPOScheduler", "train_maskable_ppo",
    "GASMARLAgent", "GASMARLScheduler", "train_gas_marl",
    "run_training", "run_evaluation", "run_comparison",
]
