"""
MaskablePPO — Single-action PPO with action masking for HPC Job Scheduling.

Adapted from uploaded MaskablePPO.py for the hpcsim heterogeneous cluster.
Key changes:
  - State includes GPU-type-aware job features + renewable energy forecast
  - Integrates with HPCGreenEnv and existing BaseScheduler interface
  - Saves/loads model for deployment in BenchmarkRunner

Training reward: ReUtil - η × AvgBSLD   (sparse, at episode end)
"""

from __future__ import annotations
import os
import csv
from pathlib import Path
from typing import Optional
import numpy as np
import scipy.signal
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.distributions import Categorical
from torch.utils.data.sampler import BatchSampler, SubsetRandomSampler

from .env import (
    HPCGreenEnv, EnvConfig,
    MAX_QUEUE_SIZE, RUN_WIN, GREEN_WIN, CLUSTER_WIN, TOTAL_ROWS, JOB_FEATURES,
)
from .networks import MaskablePPOActor, MaskablePPOCritic, CategoricalMasked
from ..cluster.cluster import Cluster
from ..scheduler.schedulers import BaseScheduler, SchedulingDecision, register

def _to_scalar(x) -> float:
    """Safely convert any scalar-like (tensor, ndarray, float) to Python float."""
    # torch.Tensor
    if hasattr(x, "reshape") and hasattr(x, "item"):
        return float(x.reshape(-1)[0].item())
    # numpy scalar / 0-d array
    if hasattr(x, "item"):
        return float(x.item())
    # numpy ndarray with .flat
    if hasattr(x, "flat"):
        return float(x.flat[0])
    # any iterable
    if hasattr(x, "__iter__"):
        return float(next(iter(x)))
    return float(x)



# ─── Experience Buffer ────────────────────────────────────────────────────────

class RolloutBuffer:
    def __init__(self):
        self.states:     list[torch.Tensor] = []
        self.masks:      list[torch.Tensor] = []
        self.actions:    list[torch.Tensor] = []
        self.log_probs:  list[torch.Tensor] = []
        self.returns:    list[float]         = []
        self.advantages: list[float]         = []

    def clear(self):
        self.__init__()

    def store(self, states, masks, actions, log_probs, returns, advantages):
        self.states.extend(states)
        self.masks.extend(masks)
        self.actions.extend(actions)
        self.log_probs.extend(log_probs)
        self.returns.extend(returns)
        self.advantages.extend(advantages)

    def __len__(self):
        return len(self.states)


# ─── MaskablePPO Agent ────────────────────────────────────────────────────────

class MaskablePPOAgent:
    """
    PPO agent with masked categorical distribution.

    Args:
        device:          'cpu' | 'cuda'
        d_model:         hidden dimension for networks
        lr_actor:        learning rate for actor
        lr_critic:       learning rate for critic
        gamma:           discount factor (1.0 recommended for HPC)
        lam:             GAE lambda
        clip_param:      PPO clipping epsilon
        ppo_epochs:      gradient update iterations per batch
        batch_size:      mini-batch size
        entropy_coef:    entropy bonus coefficient
    """

    def __init__(
        self,
        device: str = "cpu",
        d_model: int = 128,
        lr_actor: float = 1e-4,
        lr_critic: float = 5e-4,
        gamma: float = 1.0,
        lam: float = 0.97,
        clip_param: float = 0.2,
        ppo_epochs: int = 8,
        batch_size: int = 256,
        entropy_coef: float = 0.0,
        max_grad_norm: float = 0.5,
    ):
        self.device = torch.device(device)
        self.gamma = gamma
        self.lam   = lam
        self.clip_param  = clip_param
        self.ppo_epochs  = ppo_epochs
        self.batch_size  = batch_size
        self.entropy_coef = entropy_coef
        self.max_grad_norm = max_grad_norm

        self.actor  = MaskablePPOActor(d_model).to(self.device)
        self.critic = MaskablePPOCritic(d_model).to(self.device)
        self.actor_opt  = optim.Adam(self.actor.parameters(),  lr=lr_actor,  eps=1e-6)
        self.critic_opt = optim.Adam(self.critic.parameters(), lr=lr_critic, eps=1e-6)

        # Per-trajectory memory
        self._states:    list = []
        self._masks:     list = []
        self._actions:   list = []
        self._log_probs: list = []
        self._values:    list = []
        self._rewards:   list = []

        self.buffer = RolloutBuffer()

    # ── Inference ─────────────────────────────────────────────────────────────

    def _parse_obs(self, obs_flat: np.ndarray) -> torch.Tensor:
        """Convert flat obs to [1, total_rows, JOB_FEATURES] tensor."""
        obs = obs_flat.reshape(1, TOTAL_ROWS, JOB_FEATURES)
        return torch.FloatTensor(obs).to(self.device)

    def act(
        self, obs_flat: np.ndarray, mask_valid: np.ndarray
    ) -> tuple[int, torch.Tensor, torch.Tensor]:
        """
        Sample an action.
        mask_valid: [MAX_QUEUE_SIZE] — 1.0 where job is valid choice
        Returns: action_idx, log_prob, value
        """
        state = self._parse_obs(obs_flat)
        mask_t = torch.FloatTensor(
            mask_valid.reshape(1, MAX_QUEUE_SIZE)
        ).to(self.device)

        with torch.no_grad():
            logits = self.actor(state)               # [1, Q]
            value  = self.critic(state)
        dist = CategoricalMasked(logits=logits, masks=mask_t, device=self.device)
        action = dist.sample()
        log_prob = dist.log_prob(action)
        return action.item(), log_prob, value

    @torch.no_grad()
    def eval_act(self, obs_flat: np.ndarray, mask_valid: np.ndarray) -> int:
        """Greedy action for evaluation (no gradient)."""
        state  = self._parse_obs(obs_flat)
        mask_t = torch.FloatTensor(mask_valid.reshape(1, MAX_QUEUE_SIZE)).to(self.device)
        logits = self.actor(state)
        logits = torch.where(mask_t.bool(), logits, torch.tensor(-1e8, device=self.device))
        return int(logits.argmax(dim=-1).item())

    def remember(self, state_t, value, log_prob, action, reward, mask_t):
        self._rewards.append(_to_scalar(reward))
        self._states.append(state_t.to("cpu"))
        self._log_probs.append(log_prob.to("cpu"))
        self._values.append(value.to("cpu"))
        self._actions.append(torch.tensor([action]))
        self._masks.append(mask_t.to("cpu"))

    def clear_memory(self):
        self._states = []; self._masks = []; self._actions = []
        self._log_probs = []; self._values = []; self._rewards = []

    # ── GAE + Returns ─────────────────────────────────────────────────────────

    @staticmethod
    def _discount_cumsum(x: np.ndarray, discount: float) -> np.ndarray:
        return scipy.signal.lfilter([1], [1, -discount], x[::-1])[::-1]

    def finish_path(self, last_val: float = 0.0):
        rews = np.append(np.array([_to_scalar(x) for x in self._rewards], dtype=np.float32), _to_scalar(last_val))
        values = torch.cat(self._values).squeeze(-1).numpy()
        vals   = np.append(values, last_val)
        deltas = rews[:-1] + self.gamma * vals[1:] - vals[:-1]
        adv = self._discount_cumsum(deltas, self.gamma * self.lam)
        ret = self._discount_cumsum(rews, self.gamma)[:-1]
        return adv.tolist(), ret.tolist()

    def commit_trajectory(self, last_reward: float):
        adv, ret = self.finish_path(last_reward)
        self.buffer.store(
            self._states, self._masks, self._actions,
            self._log_probs, ret, adv,
        )
        self.clear_memory()

    # ── PPO Update ────────────────────────────────────────────────────────────

    def train(self):
        if not self.buffer.states:
            return
        states     = torch.cat(self.buffer.states)
        masks      = torch.cat(self.buffer.masks)
        actions    = torch.cat(self.buffer.actions)
        old_lps    = torch.cat(self.buffer.log_probs)
        returns    = torch.tensor(self.buffer.returns,    dtype=torch.float32)
        advantages = torch.tensor(self.buffer.advantages, dtype=torch.float32)
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-9)

        n = len(self.buffer.states)
        for _ in range(self.ppo_epochs):
            for idx in BatchSampler(SubsetRandomSampler(range(n)), self.batch_size, False):
                idx_t = torch.tensor(idx)
                s   = states[idx_t].to(self.device)
                m   = masks[idx_t].to(self.device)
                a   = actions[idx_t].to(self.device)
                olp = old_lps[idx_t].to(self.device)
                ret = returns[idx_t].to(self.device)
                adv = advantages[idx_t].to(self.device)

                # Actor loss
                logits = self.actor(s)
                dist   = CategoricalMasked(logits=logits, masks=m[:, :MAX_QUEUE_SIZE],
                                           device=self.device)
                new_lp  = dist.log_prob(a.squeeze(-1))
                entropy = dist.entropy()
                ratio   = torch.exp(new_lp - olp)
                surr1   = ratio * adv
                surr2   = torch.clamp(ratio, 1 - self.clip_param, 1 + self.clip_param) * adv
                actor_loss = -torch.min(surr1, surr2).mean() \
                             - self.entropy_coef * entropy.mean()

                self.actor_opt.zero_grad()
                actor_loss.backward()
                nn.utils.clip_grad_norm_(self.actor.parameters(), self.max_grad_norm)
                self.actor_opt.step()

                # Critic loss
                val  = self.critic(s).squeeze(-1)
                closs = F.mse_loss(val, ret)
                self.critic_opt.zero_grad()
                closs.backward()
                nn.utils.clip_grad_norm_(self.critic.parameters(), self.max_grad_norm)
                self.critic_opt.step()

    # ── Persistence ───────────────────────────────────────────────────────────

    def save(self, directory: str):
        Path(directory).mkdir(parents=True, exist_ok=True)
        torch.save(self.actor.state_dict(),  os.path.join(directory, "actor.pt"))
        torch.save(self.critic.state_dict(), os.path.join(directory, "critic.pt"))

    def load(self, directory: str, map_location: str = "cpu"):
        self.actor.load_state_dict(
            torch.load(os.path.join(directory, "actor.pt"), map_location=map_location))
        self.critic.load_state_dict(
            torch.load(os.path.join(directory, "critic.pt"), map_location=map_location))
        self.actor.eval(); self.critic.eval()


# ─── Training Entry Point ─────────────────────────────────────────────────────

def train_maskable_ppo(
    env_config: Optional[EnvConfig] = None,
    save_dir: str = "models/maskable_ppo",
    epochs: int = 300,
    traj_num: int = 100,
    device: str = "auto",
    csv_log: Optional[str] = None,
    verbose: bool = True,
    checkpoint_interval: int = 50,
    resume_from: Optional[str] = None,
    log_interval: int = 10,
    save_best: bool = True,
) -> MaskablePPOAgent:
    """
    Train a MaskablePPO agent on HPCGreenEnv.

    Args:
        env_config:            environment config (workload, cluster, renewable settings)
        save_dir:              directory to save final model
        epochs:                number of training epochs (each epoch = traj_num episodes)
        traj_num:              trajectories per epoch before gradient update
        device:                'cpu', 'cuda', or 'auto'
        csv_log:               optional CSV path (default: save_dir/train_log.csv)
        verbose:               print training progress bar and epoch summaries
        checkpoint_interval:   save a checkpoint every N epochs (0 = disabled)
        resume_from:           checkpoint dir to resume training from
        log_interval:          print progress every N epochs
        save_best:             keep a copy of the best model by avg_reward

    Checkpoint layout::

        save_dir/
          actor.pt              <- final model
          critic.pt
          train_log.csv         <- full training history
          checkpoints/
            epoch_0050/         <- periodic checkpoint
              actor.pt
              critic.pt
              meta.json         <- {epoch, avg_reward, avg_green, avg_bsld}
            best/               <- best-reward checkpoint
              actor.pt
              critic.pt
              meta.json

    Example::

        from hpcsim.rl.maskable_ppo import train_maskable_ppo

        # Fresh training with checkpoints every 50 epochs
        agent = train_maskable_ppo(
            epochs=300,
            checkpoint_interval=50,
            save_best=True,
            log_interval=5,
        )

        # Resume from a checkpoint
        agent = train_maskable_ppo(
            epochs=300,
            resume_from="models/maskable_ppo/checkpoints/epoch_0100",
        )
    """
    import json as _json
    import time as _time

    if device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"

    save_path = Path(save_dir)
    save_path.mkdir(parents=True, exist_ok=True)
    ckpt_dir  = save_path / "checkpoints"

    env   = HPCGreenEnv(env_config)
    agent = MaskablePPOAgent(device=device)

    # ── Resume from checkpoint ───────────────────────────────────────────────
    start_epoch = 0
    if resume_from and Path(resume_from).exists():
        agent.load(resume_from, map_location=device)
        meta_file = Path(resume_from) / "meta.json"
        if meta_file.exists():
            meta        = _json.loads(meta_file.read_text())
            start_epoch = meta.get("epoch", 0) + 1
        if verbose:
            print(f"  [MaskablePPO] Resumed from {resume_from}  "
                  f"(continuing from epoch {start_epoch})")

    # ── CSV log (append if resuming) ─────────────────────────────────────────
    _csv_path    = csv_log or str(save_path / "train_log.csv")
    _append_mode = start_epoch > 0 and Path(_csv_path).exists()
    _f = open(_csv_path, "a" if _append_mode else "w", newline="")
    _w = csv.writer(_f)
    if not _append_mode:
        _w.writerow(["epoch", "avg_reward", "avg_green", "avg_bsld",
                     "elapsed_sec", "device"])

    best_reward = float("-inf")
    t_start     = _time.time()

    if verbose:
        print(f"\n  {'='*56}")
        print(f"  [MaskablePPO] Device={device.upper()}  "
              f"epochs={epochs}  traj/epoch={traj_num}")
        print(f"  checkpoint_interval={checkpoint_interval}  "
              f"save_best={save_best}  log_interval={log_interval}")
        print(f"  {'='*56}")
        print(f"  {'Epoch':>6}  {'Reward':>10}  {'ReUtil':>9}  "
              f"{'AvgBSLD':>9}  {'ETA':>8}")
        print(f"  {'─'*54}")

    for epoch in range(start_epoch, epochs):
        epoch_rewards, epoch_green, epoch_bsld = 0.0, 0.0, 0.0
        t = 0
        obs       = env.reset()
        r         = 0.0
        green_rwd = 0.0

        while True:
            mask = env.action_mask1()
            action, log_prob, value = agent.act(obs, mask)

            state_t = torch.FloatTensor(obs.reshape(1, TOTAL_ROWS, JOB_FEATURES))
            mask_t  = torch.FloatTensor(
                np.pad(mask, (0, TOTAL_ROWS - MAX_QUEUE_SIZE)).reshape(1, TOTAL_ROWS)
            )
            agent.remember(state_t, value, log_prob, action, green_rwd, mask_t)

            obs, r, done, bsld_r, _, _, _, green_rwd = env.step(action, 0)
            epoch_rewards += r
            epoch_green   += green_rwd

            if done:
                t += 1
                agent.commit_trajectory(r)
                epoch_bsld += abs(bsld_r)
                obs    = env.reset()
                r      = 0.0
                green_rwd = 0.0
                if t >= traj_num:
                    break

        agent.train()
        agent.buffer.clear()

        avg_rew   = epoch_rewards / traj_num
        avg_green = epoch_green   / traj_num
        avg_bsld  = epoch_bsld    / traj_num
        elapsed   = _time.time() - t_start

        # ── Console log with progress bar ─────────────────────────────────────
        if verbose and ((epoch + 1) % log_interval == 0 or epoch == epochs - 1
                        or epoch == start_epoch):
            done_epochs = epoch - start_epoch + 1
            total_span  = epochs - start_epoch
            pct         = done_epochs / total_span
            bar_len     = 16
            filled      = int(bar_len * pct)
            bar         = "█" * filled + "░" * (bar_len - filled)
            eta_sec     = (elapsed / done_epochs) * (total_span - done_epochs)
            eta_str     = (f"{eta_sec/3600:.1f}h" if eta_sec > 3600
                           else f"{eta_sec/60:.1f}m" if eta_sec > 60
                           else f"{eta_sec:.0f}s")
            print(f"  [{bar}] ep={epoch+1:4d}  "
                  f"reward={avg_rew:+.4f}  "
                  f"green={avg_green:.4f}  "
                  f"bsld={avg_bsld:.4f}  "
                  f"ETA={eta_str}")

        # ── CSV row ───────────────────────────────────────────────────────────
        _w.writerow([epoch, f"{avg_rew:.6f}", f"{avg_green:.6f}",
                     f"{avg_bsld:.6f}", f"{elapsed:.1f}", device])
        _f.flush()

        # ── Periodic checkpoint ───────────────────────────────────────────────
        if checkpoint_interval > 0 and (epoch + 1) % checkpoint_interval == 0:
            ckpt_path = ckpt_dir / f"epoch_{epoch+1:04d}"
            agent.save(str(ckpt_path))
            meta = {"epoch": epoch, "avg_reward": avg_rew,
                    "avg_green": avg_green, "avg_bsld": avg_bsld,
                    "elapsed_sec": round(elapsed, 1)}
            (ckpt_path / "meta.json").write_text(_json.dumps(meta, indent=2))
            if verbose:
                print(f"  ✓ Checkpoint → {ckpt_path}")

        # ── Best-model tracking ───────────────────────────────────────────────
        if save_best and avg_rew > best_reward:
            best_reward = avg_rew
            best_path   = ckpt_dir / "best"
            agent.save(str(best_path))
            meta = {"epoch": epoch, "avg_reward": avg_rew,
                    "avg_green": avg_green, "avg_bsld": avg_bsld}
            (best_path / "meta.json").write_text(_json.dumps(meta, indent=2))

    _f.close()
    agent.save(str(save_path))

    total_time = _time.time() - t_start
    if verbose:
        print(f"\n  [MaskablePPO] Done  ({total_time/60:.1f} min)")
        print(f"  Final model  → {save_path}")
        print(f"  Training log → {_csv_path}")
        if save_best:
            print(f"  Best model   → {ckpt_dir / 'best'}  "
                  f"(reward={best_reward:.4f})")
        if checkpoint_interval > 0:
            n = sum(1 for p in ckpt_dir.iterdir() if p.is_dir() and p.name.startswith("epoch_"))
            print(f"  Checkpoints  → {ckpt_dir}  ({n} periodic + 1 best)")

    return agent


# ─── Scheduler Wrapper ────────────────────────────────────────────────────────

@register
class MaskablePPOScheduler(BaseScheduler):
    """
    MaskablePPO-based scheduler compatible with BaseScheduler interface.

    Load a pretrained model or provide an agent directly.
    Renewable energy info is passed via the cluster snapshot.

    Usage (inference):
        scheduler = MaskablePPOScheduler(cluster, model_dir="models/maskable_ppo")
        # Then use normally in SimulationEngine / BenchmarkRunner

    Usage (fresh agent — will act randomly until trained):
        scheduler = MaskablePPOScheduler(cluster)
    """
    name = "maskable_ppo"

    def __init__(
        self,
        cluster: Cluster,
        model_dir: Optional[str] = None,
        device: str = "cpu",
        env_config: Optional[EnvConfig] = None,
    ):
        super().__init__("MaskablePPO", cluster)
        self._env_cfg = env_config or EnvConfig(
            cluster_config=cluster.config.name
            if cluster.config.name in ("medium_heterogeneous_gavel",
                                       "gogh_hetero", "tiny_test",
                                       "small_v100", "large_mixed")
            else "medium_heterogeneous_gavel"
        )
        # Create a lightweight env just for obs/feature computation
        self._env = HPCGreenEnv(self._env_cfg)
        self._agent = MaskablePPOAgent(device=device)
        if model_dir and Path(model_dir).exists():
            self._agent.load(model_dir, map_location=device)
            print(f"[MaskablePPO] Loaded model from {model_dir}")

    def schedule(self, pending, running, current_time):
        decision = SchedulingDecision()
        if not pending:
            return decision

        # Sync env state to current simulation state
        self._sync_env(pending, running, current_time)

        obs  = self._env._get_obs()
        mask = self._env.action_mask1()

        action = self._agent.eval_act(obs, mask)
        action = min(action, len(pending) - 1)

        # Try to schedule the selected job
        job = pending[action]
        n   = self._gpu_count_for_job(job)
        gpu_ids = self._find_gpus(job, prefer_consolidated=True)
        if gpu_ids:
            decision.add(job, gpu_ids)

        # Greedy fill remaining capacity
        for j in pending:
            if j is job:
                continue
            gids = self._find_gpus(j, prefer_consolidated=True)
            if gids:
                decision.add(j, gids)

        return decision

    def _sync_env(self, pending, running, current_time: float):
        """Approximate env state from simulation state for obs generation."""
        self._env._current_time = current_time
        self._env._pending = list(pending)[:MAX_QUEUE_SIZE]
        self._env._avail_gpus = self.cluster.free_gpu_count()
        # Build lightweight running job records
        self._env._running = []
        for job in running:
            from .env import _RunningJob
            rj = _RunningJob(
                job_id=job.job_id,
                start_time=getattr(job, "start_time", current_time) or current_time,
                finish_time=current_time + 600.0,  # estimate
                num_gpus=len(getattr(job, "allocated_gpus", [])) or 1,
                power_w=self._env._estimate_job_power(job),
                req_runtime=600.0,
                wait_time=0.0,
            )
            self._env._running.append(rj)
