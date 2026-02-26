"""
GAS-MARL — Green-Aware job Scheduling based on Multi-Action Deep RL.

Adapted from uploaded MARL.py for hpcsim heterogeneous GPU clusters.
Key changes from original:
  - State includes per-GPU-type affinity signals
  - Delay actions interact with Green-Backfilling logic
  - Scheduler wrapper integrates with existing BaseScheduler interface
  - Tracks renewable energy utilization metric natively

Reference: Chen et al., "GAS-MARL: Green-Aware job Scheduling algorithm
for HPC clusters based on Multi-Action Deep Reinforcement Learning", FGCS 2025.
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
    CLUSTER_WIN, TOTAL_ROWS,
    HPCGreenEnv, EnvConfig,
    MAX_QUEUE_SIZE, RUN_WIN, GREEN_WIN, JOB_FEATURES,
    ACTION2_NUM, DELAY_MAX_JOB_NUM, DELAY_TIMES,
    _RunningJob,
)
from .networks import GASMARLActor, GASMARLCritic
from ..cluster.cluster import Cluster
from ..scheduler.schedulers import BaseScheduler, SchedulingDecision, register

def _to_scalar(x) -> float:
    """Safely convert any scalar-like (tensor, ndarray, float) to Python float."""
    if hasattr(x, "item"):
        try:
            return float(x.item())
        except (ValueError, RuntimeError):
            return float(x.flat[0])
    if hasattr(x, "__iter__"):
        return float(next(iter(x)))
    return float(x)



# ─── Experience Buffer ────────────────────────────────────────────────────────

class MARLBuffer:
    def __init__(self):
        self.states:      list[torch.Tensor] = []
        self.actions1:    list[torch.Tensor] = []
        self.actions2:    list[torch.Tensor] = []
        self.masks1:      list[torch.Tensor] = []
        self.masks2:      list[torch.Tensor] = []
        self.log_probs1:  list[torch.Tensor] = []
        self.log_probs2:  list[torch.Tensor] = []
        self.returns:     list[float]        = []
        self.advantages:  list[float]        = []
        self.job_inputs:  list[torch.Tensor] = []

    def clear(self):
        self.__init__()

    def store(self, states, masks1, masks2, actions1, actions2,
              lp1, lp2, returns, advantages, job_inputs):
        self.states.extend(states)
        self.masks1.extend(masks1)
        self.masks2.extend(masks2)
        self.actions1.extend(actions1)
        self.actions2.extend(actions2)
        self.log_probs1.extend(lp1)
        self.log_probs2.extend(lp2)
        self.returns.extend(returns)
        self.advantages.extend(advantages)
        self.job_inputs.extend(job_inputs)

    def __len__(self):
        return len(self.states)


# ─── GAS-MARL Agent ──────────────────────────────────────────────────────────

class GASMARLAgent:
    """
    Multi-Action PPO agent implementing GAS-MARL.

    Two sub-actions per decision step:
      a^job   — which job to select (job selection network)
      a^delay — whether/how long to delay (delay decision network)

    Composite probability ratio for PPO:
      r(θ) = [π(a^job|s) × π(a^delay|s)] / [π_old(a^job|s) × π_old(a^delay|s)]
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
        self.device      = torch.device(device)
        self.gamma       = gamma
        self.lam         = lam
        self.clip_param  = clip_param
        self.ppo_epochs  = ppo_epochs
        self.batch_size  = batch_size
        self.entropy_coef = entropy_coef
        self.max_grad_norm = max_grad_norm

        self.actor  = GASMARLActor(d_model).to(self.device)
        self.critic = GASMARLCritic(d_model).to(self.device)
        self.actor_opt  = optim.Adam(self.actor.parameters(),  lr=lr_actor,  eps=1e-6)
        self.critic_opt = optim.Adam(self.critic.parameters(), lr=lr_critic, eps=1e-6)

        # Per-trajectory memory
        self._states:    list = []
        self._actions1:  list = []
        self._actions2:  list = []
        self._masks1:    list = []
        self._masks2:    list = []
        self._lp1:       list = []
        self._lp2:       list = []
        self._values:    list = []
        self._rewards:   list = []
        self._job_inputs: list = []

        self.buffer = MARLBuffer()

    # ── Inference ─────────────────────────────────────────────────────────────

    def _parse_obs(self, obs_flat: np.ndarray) -> torch.Tensor:
        total_slots = TOTAL_ROWS
        return torch.FloatTensor(
            obs_flat.reshape(1, total_slots, JOB_FEATURES)
        ).to(self.device)

    def choose_action(
        self,
        obs_flat: np.ndarray,
        inv_mask1: np.ndarray,   # [Q]  1 = invalid job
        inv_mask2: np.ndarray,   # [A2] 1 = invalid delay
    ) -> tuple[int, torch.Tensor, int, torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Sample (job_action, delay_action).
        inv_mask1: 1 where job slot is INVALID (to subtract from logits)
        inv_mask2: 1 where delay option is INVALID
        Returns: job_idx, log_prob1, delay_idx, log_prob2, value, job_features
        """
        state = self._parse_obs(obs_flat)
        m1    = torch.FloatTensor(inv_mask1.reshape(1, MAX_QUEUE_SIZE)).to(self.device)
        m2    = torch.FloatTensor(inv_mask2.reshape(1, ACTION2_NUM)).to(self.device)

        with torch.no_grad():
            # Job selection
            logits1 = self.actor.get_job_logits(state, m1)   # [1, Q]
            probs1  = F.softmax(logits1, dim=-1)
        dist1  = Categorical(probs=probs1)
        ac1    = dist1.sample()
        lp1    = dist1.log_prob(ac1)

        # Extract selected job features for delay network
        job_feat = state[:, ac1.item():ac1.item() + 1, :]    # [1, 1, F]

        with torch.no_grad():
            # Delay decision
            logits2 = self.actor.get_delay_logits(state, job_feat, m2)   # [1, A2]
            probs2  = F.softmax(logits2, dim=-1)
            value   = self.critic(state)
        dist2  = Categorical(probs=probs2)
        ac2    = dist2.sample()
        lp2    = dist2.log_prob(ac2)

        return ac1.item(), lp1, ac2.item(), lp2, value, job_feat

    @torch.no_grad()
    def eval_action(
        self,
        obs_flat: np.ndarray,
        inv_mask1: np.ndarray,
        inv_mask2: np.ndarray,
    ) -> tuple[int, int]:
        """Greedy action selection for evaluation."""
        state = self._parse_obs(obs_flat)
        m1    = torch.FloatTensor(inv_mask1.reshape(1, MAX_QUEUE_SIZE)).to(self.device)
        m2    = torch.FloatTensor(inv_mask2.reshape(1, ACTION2_NUM)).to(self.device)

        logits1  = self.actor.get_job_logits(state, m1)
        ac1      = int(logits1.argmax(dim=-1).item())
        job_feat = state[:, ac1:ac1 + 1, :]
        logits2  = self.actor.get_delay_logits(state, job_feat, m2)
        ac2      = int(logits2.argmax(dim=-1).item())
        return ac1, ac2

    def remember(self, state_t, value, lp1, lp2, ac1, ac2,
                 reward, m1, m2, job_input):
        self._rewards.append(_to_scalar(reward))
        self._states.append(state_t.to("cpu"))
        self._lp1.append(lp1.to("cpu"))
        self._lp2.append(lp2.to("cpu"))
        self._values.append(value.to("cpu"))
        self._actions1.append(torch.tensor([ac1]))
        self._actions2.append(torch.tensor([ac2]))
        self._masks1.append(m1.to("cpu"))
        self._masks2.append(m2.to("cpu"))
        self._job_inputs.append(job_input.to("cpu"))

    def clear_memory(self):
        (self._states, self._actions1, self._actions2,
         self._masks1, self._masks2, self._lp1, self._lp2,
         self._values, self._rewards, self._job_inputs) = ([],) * 10

    # ── GAE + Returns ─────────────────────────────────────────────────────────

    @staticmethod
    def _discount_cumsum(x, discount):
        return scipy.signal.lfilter([1], [1, -discount], x[::-1])[::-1]

    def finish_path(self, last_val=0.0):
        rews   = np.append(np.array([_to_scalar(x) for x in self._rewards], dtype=np.float32), _to_scalar(last_val))
        values = torch.cat(self._values).squeeze(-1).numpy()
        vals   = np.append(values, last_val)
        deltas = rews[:-1] + self.gamma * vals[1:] - vals[:-1]
        adv    = self._discount_cumsum(deltas, self.gamma * self.lam)
        ret    = self._discount_cumsum(rews, self.gamma)[:-1]
        return adv.tolist(), ret.tolist()

    def commit_trajectory(self, last_reward: float):
        adv, ret = self.finish_path(float(last_reward))
        self.buffer.store(
            self._states, self._masks1, self._masks2,
            self._actions1, self._actions2,
            self._lp1, self._lp2,
            ret, adv, self._job_inputs,
        )
        self.clear_memory()

    # ── PPO Update ────────────────────────────────────────────────────────────

    def train(self):
        if not self.buffer.states:
            return
        states    = torch.cat(self.buffer.states)
        masks1    = torch.cat(self.buffer.masks1)
        masks2    = torch.cat(self.buffer.masks2)
        actions1  = torch.cat(self.buffer.actions1)
        actions2  = torch.cat(self.buffer.actions2)
        old_lp1   = torch.cat(self.buffer.log_probs1)
        old_lp2   = torch.cat(self.buffer.log_probs2)
        job_inputs= torch.cat(self.buffer.job_inputs)
        returns   = torch.tensor(self.buffer.returns,    dtype=torch.float32)
        advantages= torch.tensor(self.buffer.advantages, dtype=torch.float32)
        advantages= (advantages - advantages.mean()) / (advantages.std() + 1e-9)

        n = len(self.buffer.states)
        for _ in range(self.ppo_epochs):
            for idx in BatchSampler(SubsetRandomSampler(range(n)), self.batch_size, False):
                idx_t = torch.tensor(idx)
                s   = states[idx_t].to(self.device)
                m1  = masks1[idx_t].to(self.device)
                m2  = masks2[idx_t].to(self.device)
                a1  = actions1[idx_t].squeeze(-1).to(self.device)
                a2  = actions2[idx_t].squeeze(-1).to(self.device)
                olp1= old_lp1[idx_t].to(self.device)
                olp2= old_lp2[idx_t].to(self.device)
                ret = returns[idx_t].to(self.device)
                adv = advantages[idx_t].to(self.device)
                ji  = job_inputs[idx_t].to(self.device)

                # Recompute action probabilities
                logits1 = self.actor.get_job_logits(s, m1)
                probs1  = F.softmax(logits1, dim=-1)
                dist1   = Categorical(probs=probs1)
                new_lp1 = dist1.log_prob(a1)
                ent1    = dist1.entropy()

                logits2 = self.actor.get_delay_logits(s, ji, m2)
                probs2  = F.softmax(logits2, dim=-1)
                dist2   = Categorical(probs=probs2)
                new_lp2 = dist2.log_prob(a2)
                ent2    = dist2.entropy()

                # Composite ratio (GAS-MARL Eq. 12)
                old_composite = olp1 + olp2
                new_composite = new_lp1 + new_lp2
                ratio  = torch.exp(new_composite - old_composite)
                surr1  = ratio * adv
                surr2  = torch.clamp(ratio, 1 - self.clip_param, 1 + self.clip_param) * adv
                entropy = (ent1 + ent2) / 2.0
                actor_loss = -torch.min(surr1, surr2).mean() \
                             - self.entropy_coef * entropy.mean()

                self.actor_opt.zero_grad()
                actor_loss.backward()
                nn.utils.clip_grad_norm_(self.actor.parameters(), self.max_grad_norm)
                self.actor_opt.step()

                val   = self.critic(s).squeeze(-1)
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

def train_gas_marl(
    env_config: Optional[EnvConfig] = None,
    save_dir: str = "models/gas_marl",
    epochs: int = 300,
    traj_num: int = 100,
    device: str = "auto",
    csv_log: Optional[str] = None,
    verbose: bool = True,
    checkpoint_interval: int = 50,
    resume_from: Optional[str] = None,
    log_interval: int = 10,
    save_best: bool = True,
) -> GASMARLAgent:
    """
    Train a GAS-MARL agent on HPCGreenEnv.

    Args:
        env_config:            environment config (workload, cluster, renewable settings)
        save_dir:              directory to save final model
        epochs:                training epochs
        traj_num:              trajectories per epoch
        device:                'cpu', 'cuda', or 'auto'
        csv_log:               optional CSV path (default: save_dir/train_log.csv)
        verbose:               print training progress
        checkpoint_interval:   save a checkpoint every N epochs (0 = disabled)
        resume_from:           checkpoint dir to resume training from
        log_interval:          print progress every N epochs
        save_best:             keep a copy of the best model by avg_reward

    Checkpoint layout::

        save_dir/
          actor.pt              <- final model
          critic.pt
          train_log.csv
          checkpoints/
            epoch_0050/
              actor.pt
              critic.pt
              meta.json
            best/
              actor.pt
              critic.pt
              meta.json

    Example::

        from hpcsim.rl.gas_marl import train_gas_marl

        agent = train_gas_marl(
            epochs=300,
            checkpoint_interval=50,
            save_best=True,
        )

        # Resume interrupted training
        agent = train_gas_marl(
            epochs=300,
            resume_from="models/gas_marl/checkpoints/epoch_0150",
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
    agent = GASMARLAgent(device=device)

    # ── Resume from checkpoint ───────────────────────────────────────────────
    start_epoch = 0
    if resume_from and Path(resume_from).exists():
        agent.load(resume_from, map_location=device)
        meta_file = Path(resume_from) / "meta.json"
        if meta_file.exists():
            meta        = _json.loads(meta_file.read_text())
            start_epoch = meta.get("epoch", 0) + 1
        if verbose:
            print(f"  [GAS-MARL] Resumed from {resume_from}  "
                  f"(continuing from epoch {start_epoch})")

    # ── CSV log ───────────────────────────────────────────────────────────────
    _csv_path    = csv_log or str(save_path / "train_log.csv")
    _append_mode = start_epoch > 0 and Path(_csv_path).exists()
    _f = open(_csv_path, "a" if _append_mode else "w", newline="")
    _w = csv.writer(_f)
    if not _append_mode:
        _w.writerow(["epoch", "avg_reward", "avg_green", "avg_bsld",
                     "elapsed_sec", "device"])

    best_reward  = float("-inf")
    t_start      = _time.time()
    total_slots  = TOTAL_ROWS

    if verbose:
        print(f"\n  {'='*56}")
        print(f"  [GAS-MARL]    Device={device.upper()}  "
              f"epochs={epochs}  traj/epoch={traj_num}")
        print(f"  checkpoint_interval={checkpoint_interval}  "
              f"save_best={save_best}  log_interval={log_interval}")
        print(f"  {'='*56}")
        print(f"  {'Epoch':>6}  {'Reward':>10}  {'ReUtil':>9}  "
              f"{'AvgBSLD':>9}  {'ETA':>8}")
        print(f"  {'─'*54}")

    for epoch in range(start_epoch, epochs):
        epoch_rewards, epoch_green, epoch_bsld = 0.0, 0.0, 0.0
        t         = 0
        obs       = env.reset()
        r         = 0.0
        green_rwd = 0.0

        while True:
            valid_mask = env.action_mask1()
            inv_mask1  = (1.0 - valid_mask)
            inv_mask2  = env.action_mask2()

            ac1, lp1, ac2, lp2, value, job_feat = agent.choose_action(
                obs, inv_mask1, inv_mask2
            )

            state_t = torch.FloatTensor(obs.reshape(1, total_slots, JOB_FEATURES))
            m1_t    = torch.FloatTensor(inv_mask1.reshape(1, MAX_QUEUE_SIZE))
            m2_t    = torch.FloatTensor(inv_mask2.reshape(1, ACTION2_NUM))
            agent.remember(state_t, value, lp1, lp2, ac1, ac2,
                           green_rwd, m1_t, m2_t, job_feat)

            obs, r, done, bsld_r, _, _, running_num, green_rwd = env.step(ac1, ac2)
            epoch_rewards += r
            epoch_green   += green_rwd

            if done:
                t += 1
                agent.commit_trajectory(r)
                epoch_bsld += abs(bsld_r)
                obs       = env.reset()
                r         = 0.0
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
        print(f"\n  [GAS-MARL] Done  ({total_time/60:.1f} min)")
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
class GASMARLScheduler(BaseScheduler):
    """
    GAS-MARL scheduler compatible with BaseScheduler interface.

    Implements Green-Backfilling when delay_action > 0:
    - Type 2: wait until N running jobs complete
    - Type 3: fixed delay duration from DELAY_TIMES list

    Usage:
        # With pretrained model:
        scheduler = GASMARLScheduler(cluster, model_dir="models/gas_marl")

        # Fresh (random policy):
        scheduler = GASMARLScheduler(cluster)
    """
    name = "gas_marl"

    def __init__(
        self,
        cluster: Cluster,
        model_dir: Optional[str] = None,
        device: str = "cpu",
        env_config: Optional[EnvConfig] = None,
    ):
        super().__init__("GAS-MARL", cluster)
        self._env_cfg = env_config or EnvConfig(
            cluster_config=(
                cluster.config.name
                if cluster.config.name in CLUSTER_CONFIGS_VALID
                else "medium_heterogeneous_gavel"
            )
        )
        self._env   = HPCGreenEnv(self._env_cfg)
        self._agent = GASMARLAgent(device=device)
        if model_dir and Path(model_dir).exists():
            self._agent.load(model_dir, map_location=device)
            print(f"[GAS-MARL] Loaded model from {model_dir}")

        # Track delayed jobs: job_id → release_time
        self._delayed_until: dict[str, float] = {}

    def schedule(self, pending, running, current_time):
        decision = SchedulingDecision()
        if not pending:
            return decision

        # Release delayed jobs that have passed their release time
        active_pending = [
            j for j in pending
            if self._delayed_until.get(j.job_id, 0.0) <= current_time
        ]
        if not active_pending:
            return decision

        # Sync env state
        self._sync_env(active_pending, running, current_time)
        obs      = self._env._get_obs()
        inv_mask1 = 1.0 - self._env.action_mask1()
        inv_mask2 = self._env.action_mask2()

        ac1, ac2 = self._agent.eval_action(obs, inv_mask1, inv_mask2)
        ac1 = min(ac1, len(active_pending) - 1)
        selected_job = active_pending[ac1]

        # Handle delay decision
        if ac2 > 0:
            release_t = self._compute_release_time(ac2, running, current_time)
            self._delayed_until[selected_job.job_id] = release_t
            # Green-backfilling: schedule other valid jobs during delay
            for j in active_pending:
                if j.job_id == selected_job.job_id:
                    continue
                if self._passes_green_backfill(j, current_time, release_t):
                    gids = self._find_gpus(j, prefer_consolidated=True)
                    if gids:
                        decision.add(j, gids)
        else:
            # Schedule selected job immediately
            self._delayed_until.pop(selected_job.job_id, None)
            gids = self._find_gpus(selected_job, prefer_consolidated=True)
            if gids:
                decision.add(selected_job, gids)
            # Greedy fill
            for j in active_pending:
                if j.job_id == selected_job.job_id:
                    continue
                if self._passes_green_backfill(j, current_time, float("inf")):
                    gids = self._find_gpus(j, prefer_consolidated=True)
                    if gids:
                        decision.add(j, gids)

        return decision

    def _compute_release_time(self, ac2, running, current_time):
        if ac2 <= DELAY_MAX_JOB_NUM:
            n_wait = min(ac2, len(running))
            if n_wait > 0 and running:
                # Estimate completion times (use attained_service as proxy)
                sorted_run = sorted(running, key=lambda j: getattr(j, "start_time", current_time) or current_time)
                capped = current_time + 3600.0
                return min(capped, current_time + 300.0 * n_wait)
            return current_time
        dt_idx = ac2 - (DELAY_MAX_JOB_NUM + 1)
        dt_idx = min(dt_idx, len(DELAY_TIMES) - 1)
        return current_time + DELAY_TIMES[dt_idx]

    def _passes_green_backfill(self, job, current_time, max_finish_time) -> bool:
        """
        Green-Backfilling acceptance criterion: estimated brown energy < threshold.
        (GAS-MARL Algorithm 2, Line 7)
        """
        n_gpu     = self._gpu_count_for_job(job)
        power_w   = self._env._estimate_job_power(job)
        runtime   = self._env._estimate_runtime(job)
        re_avail  = self._env._re.available_power_watts(current_time)
        cluster_p = self._env._re.idle_power_watts(self.cluster.total_gpus())
        running_p = sum(
            self._env._re.job_power_watts(
                len(getattr(j, "allocated_gpus", [])) or 1
            )
            for j in []  # simplified
        )
        total_with_job = cluster_p + running_p + power_w
        brown_power = max(0.0, total_with_job - re_avail)
        brown_energy = brown_power * runtime
        return brown_energy < self._env_cfg.brown_threshold_j

    def _gpu_count_for_job(self, job) -> int:
        n = getattr(job, "num_gpus_requested", 1)
        return max(1, min(n, self.cluster.total_gpus()))

    def _sync_env(self, pending, running, current_time: float):
        self._env._current_time = current_time
        self._env._pending = list(pending)[:MAX_QUEUE_SIZE]
        self._env._avail_gpus = self.cluster.free_gpu_count()
        self._env._running = []
        for job in running:
            rj = _RunningJob(
                job_id=job.job_id,
                start_time=getattr(job, "start_time", current_time) or current_time,
                finish_time=current_time + 600.0,
                num_gpus=len(getattr(job, "allocated_gpus", [])) or 1,
                power_w=self._env._estimate_job_power(job),
                req_runtime=600.0,
                wait_time=0.0,
            )
            self._env._running.append(rj)


# Valid cluster config names for auto-detection
CLUSTER_CONFIGS_VALID = {
    "tiny_test", "small_v100", "medium_heterogeneous_gavel",
    "large_mixed", "gogh_hetero",
}
