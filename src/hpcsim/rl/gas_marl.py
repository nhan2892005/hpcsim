"""
GAS-MARL — Green-Aware job Scheduling based on Multi-Action Deep RL.

Adapted from uploaded MARL.py for hpcsim heterogeneous GPU clusters.
Key changes from original:
  - State includes per-GPU-type affinity signals
  - Delay actions interact with Green-Backfilling logic
  - Scheduler wrapper integrates with existing BaseScheduler interface
  - Tracks renewable energy utilization metric natively

Performance optimisations (v2):
  
  1. Pre-allocated numpy MARLBuffer
     All experience (states, masks1, masks2, actions1, actions2, log_probs,
     values, rewards, job_inputs, returns, advantages) is stored in numpy
     arrays allocated once in __init__.  No per-step list growth or tensor
     allocation.

  2. Zero GPU↔CPU transfers in the training hot loop
     choose_action() performs its two forward passes on device, then extracts
     Python scalars via .item() and job features via .cpu().numpy() before
     returning.  No GPU tensor persists between steps.

  3. Single batch GPU transfer per PPO update
     buffer.get_tensors(device) converts the full epoch of numpy data in ONE
     torch.as_tensor() call per array, then mini-batches index the resident
     GPU tensors — no per-batch .to() calls.

  4. GAE entirely on numpy
     finish_path() runs scipy.signal.lfilter on numpy slices — no gradient
     tape, no device transfer, O(T) time.

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
    if hasattr(x, "reshape") and hasattr(x, "item"):
        return float(x.reshape(-1)[0].item())
    if hasattr(x, "item"):
        return float(x.item())
    if hasattr(x, "flat"):
        return float(x.flat[0])
    if hasattr(x, "__iter__"):
        return float(next(iter(x)))
    return float(x)


# Pre-allocated MARL Buffer 

class MARLBuffer:
    """
    Pre-allocated numpy buffer for GAS-MARL two-action PPO experience.

    Design principles
    
    Identical philosophy to RolloutBuffer in maskable_ppo.py:
    • All arrays allocated once; clear() resets only the integer pointers.
    • No torch.Tensor stored — all data is numpy dtype-matched arrays.
    • GAE in finish_path() uses scipy.signal.lfilter on numpy slices.
    • get_tensors(device) performs ONE torch.as_tensor() call per array.

    Extra arrays vs RolloutBuffer:
        masks1     [max_size, MAX_QUEUE_SIZE]   float32  — invalid job mask
        masks2     [max_size, ACTION2_NUM]       float32  — invalid delay mask
        actions1   [max_size]                    int64    — job selection
        actions2   [max_size]                    int64    — delay selection
        log_probs1 [max_size]                    float32
        log_probs2 [max_size]                    float32
        job_inputs [max_size, JOB_FEATURES]      float32  — selected job features
    """

    def __init__(self, max_size: int = 32_768) -> None:
        self.max_size = max_size
        # Pre-allocate all arrays 
        self.states     = np.empty((max_size, TOTAL_ROWS * JOB_FEATURES), dtype=np.float32)
        self.masks1     = np.empty((max_size, MAX_QUEUE_SIZE),             dtype=np.float32)
        self.masks2     = np.empty((max_size, ACTION2_NUM),                dtype=np.float32)
        self.actions1   = np.empty((max_size,),                             dtype=np.int64)
        self.actions2   = np.empty((max_size,),                             dtype=np.int64)
        self.log_probs1 = np.empty((max_size,),                             dtype=np.float32)
        self.log_probs2 = np.empty((max_size,),                             dtype=np.float32)
        self.values     = np.empty((max_size,),                             dtype=np.float32)
        self.rewards    = np.zeros((max_size,),                             dtype=np.float32)
        self.job_inputs = np.empty((max_size, JOB_FEATURES),               dtype=np.float32)
        # Computed by finish_path
        self.returns    = np.empty((max_size,),                             dtype=np.float32)
        self.advantages = np.empty((max_size,),                             dtype=np.float32)
        # Pointer management
        self.ptr         = 0
        self.size        = 0
        self._traj_start = 0

    # Write 

    def add(
        self,
        obs_flat:  np.ndarray,   # [TOTAL_ROWS * JOB_FEATURES]
        mask1:     np.ndarray,   # [MAX_QUEUE_SIZE] — invalid mask (1 = invalid)
        mask2:     np.ndarray,   # [ACTION2_NUM]    — invalid mask (1 = invalid)
        action1:   int,
        action2:   int,
        log_prob1: float,
        log_prob2: float,
        value:     float,
        reward:    float,
        job_feat:  np.ndarray,   # [JOB_FEATURES]  — features of selected job
    ) -> None:
        """
        Write one step into the buffer (O(1), no allocation).

        All scalar arguments must be Python floats/ints (not tensors).
        obs_flat, mask1, mask2, job_feat must be contiguous numpy float32 arrays.
        """
        i = self.ptr
        if i >= self.max_size:
            return
        self.states[i]     = obs_flat
        self.masks1[i]     = mask1
        self.masks2[i]     = mask2
        self.actions1[i]   = action1
        self.actions2[i]   = action2
        self.log_probs1[i] = log_prob1
        self.log_probs2[i] = log_prob2
        self.values[i]     = value
        self.rewards[i]    = reward
        self.job_inputs[i] = job_feat
        self.ptr += 1

    # GAE computation 

    def finish_path(self, last_val: float, gamma: float, lam: float) -> None:
        """
        Compute GAE advantages and discounted returns for the current trajectory.

        Identical algorithm to RolloutBuffer.finish_path() — entirely on numpy,
        no tensors, no device transfers.  Supports multiple trajectories per
        epoch via the _traj_start pointer.
        """
        start = self._traj_start
        end   = self.ptr
        if end <= start:
            return

        rews   = np.append(self.rewards[start:end], last_val).astype(np.float32)
        vals   = np.append(self.values[start:end],  last_val).astype(np.float32)
        deltas = rews[:-1] + gamma * vals[1:] - vals[:-1]

        adv = scipy.signal.lfilter(
            [1], [1, -(gamma * lam)], deltas[::-1]
        )[::-1].copy().astype(np.float32)

        ret = scipy.signal.lfilter(
            [1], [1, -gamma], rews[::-1]
        )[::-1][:-1].copy().astype(np.float32)

        self.advantages[start:end] = adv
        self.returns[start:end]    = ret
        self._traj_start = end
        self.size        = end

    # Batch GPU transfer 

    def get_tensors(self, device: torch.device) -> dict[str, torch.Tensor]:
        """
        Convert the full epoch of experience to GPU tensors in ONE transfer.

        job_inputs is returned as [N, 1, JOB_FEATURES] to match the shape
        expected by GASMARLActor.get_delay_logits().
        """
        n = self.size
        return {
            "states":     torch.as_tensor(
                              self.states[:n], device=device
                          ).view(n, TOTAL_ROWS, JOB_FEATURES),
            "masks1":     torch.as_tensor(self.masks1[:n],     device=device),
            "masks2":     torch.as_tensor(self.masks2[:n],     device=device),
            "actions1":   torch.as_tensor(self.actions1[:n],   device=device),
            "actions2":   torch.as_tensor(self.actions2[:n],   device=device),
            "log_probs1": torch.as_tensor(self.log_probs1[:n], device=device),
            "log_probs2": torch.as_tensor(self.log_probs2[:n], device=device),
            "returns":    torch.as_tensor(self.returns[:n],    device=device),
            "advantages": torch.as_tensor(self.advantages[:n], device=device),
            # [N, 1, JOB_FEATURES] — shape required by get_delay_logits
            "job_inputs": torch.as_tensor(
                              self.job_inputs[:n], device=device
                          ).unsqueeze(1),
        }

    # Lifecycle 

    def clear(self) -> None:
        """Reset pointers without deallocating arrays."""
        self.ptr         = 0
        self.size        = 0
        self._traj_start = 0

    def __len__(self) -> int:
        return self.size


# GAS-MARL Agent 

class GASMARLAgent:
    """
    Multi-Action PPO agent implementing GAS-MARL.

    Two sub-actions per decision step:
      a^job   — which job to select (job selection network)
      a^delay — whether/how long to delay (delay decision network)

    Composite probability ratio for PPO:
      r(θ) = [π(a^job|s) × π(a^delay|s)] / [π_old(a^job|s) × π_old(a^delay|s)]

    Performance notes (v2)
    
    • choose_action() returns (int, float, int, float, float, np.ndarray) —
      all Python scalars + one numpy array.  No GPU tensor survives the call.

    • remember() writes all values directly into the pre-allocated MARLBuffer.
      Zero tensor allocation, zero .to() call per step.

    • commit_trajectory() → buffer.finish_path() — entirely on numpy.

    • train() does ONE batch GPU upload via buffer.get_tensors(device), then
      iterates mini-batches with integer index tensors on the resident data.
    """

    def __init__(
        self,
        device:        str   = "cpu",
        d_model:       int   = 128,
        lr_actor:      float = 1e-4,
        lr_critic:     float = 5e-4,
        gamma:         float = 1.0,
        lam:           float = 0.97,
        clip_param:    float = 0.2,
        ppo_epochs:    int   = 8,
        batch_size:    int   = 256,
        entropy_coef:  float = 0.0,
        max_grad_norm: float = 0.5,
        buffer_size:   int   = 32_768,
    ):
        self.device        = torch.device(device)
        self.gamma         = gamma
        self.lam           = lam
        self.clip_param    = clip_param
        self.ppo_epochs    = ppo_epochs
        self.batch_size    = batch_size
        self.entropy_coef  = entropy_coef
        self.max_grad_norm = max_grad_norm

        self.actor  = GASMARLActor(d_model).to(self.device)
        self.critic = GASMARLCritic(d_model).to(self.device)
        self.actor_opt  = optim.Adam(self.actor.parameters(),  lr=lr_actor,  eps=1e-6)
        self.critic_opt = optim.Adam(self.critic.parameters(), lr=lr_critic, eps=1e-6)

        # Single pre-allocated buffer — persists across epochs
        self.buffer = MARLBuffer(max_size=buffer_size)

    # Inference 

    def choose_action(
        self,
        obs_flat:   np.ndarray,   # [TOTAL_ROWS * JOB_FEATURES]
        inv_mask1:  np.ndarray,   # [MAX_QUEUE_SIZE] — 1 = INVALID job slot
        inv_mask2:  np.ndarray,   # [ACTION2_NUM]    — 1 = INVALID delay option
    ) -> tuple[int, float, int, float, float, np.ndarray]:
        """
        Sample (job_action, delay_action).

        Both forward passes run on self.device.  All results are extracted as
        Python scalars or numpy arrays before returning — no GPU tensor persists
        between environment steps.

        Args:
            obs_flat:  flat observation from env._get_obs()
            inv_mask1: invalid mask for job selection   (1 = forbidden)
            inv_mask2: invalid mask for delay selection (1 = forbidden)

        Returns:
            ac1:          job index   (int)
            lp1:          log π(a1|s) (float)
            ac2:          delay index (int)
            lp2:          log π(a2|s, a1) (float)
            value:        V(s)         (float)
            job_feat_np:  features of selected job  (np.ndarray [JOB_FEATURES])
        """
        state = torch.as_tensor(
            obs_flat.reshape(1, TOTAL_ROWS, JOB_FEATURES),
            device=self.device,
        )
        m1 = torch.as_tensor(
            inv_mask1.reshape(1, MAX_QUEUE_SIZE),
            device=self.device,
        )
        m2 = torch.as_tensor(
            inv_mask2.reshape(1, ACTION2_NUM),
            device=self.device,
        )

        with torch.no_grad():
            # Job selection 
            logits1 = self.actor.get_job_logits(state, m1)    # [1, Q]
            probs1  = F.softmax(logits1, dim=-1)
            dist1   = Categorical(probs=probs1)
            ac1_t   = dist1.sample()                           # [1]
            lp1_t   = dist1.log_prob(ac1_t)                   # [1]

            ac1_int = int(ac1_t.item())

            # Extract selected job feature row from obs (zero-copy on CPU,
            # device gather on CUDA — either way just one row = 12 floats)
            job_feat_t = state[:, ac1_int:ac1_int + 1, :]     # [1, 1, JOB_FEATURES]

            # Delay decision 
            logits2 = self.actor.get_delay_logits(state, job_feat_t, m2)  # [1, A2]
            probs2  = F.softmax(logits2, dim=-1)
            dist2   = Categorical(probs=probs2)
            ac2_t   = dist2.sample()                           # [1]
            lp2_t   = dist2.log_prob(ac2_t)                   # [1]

            value_t = self.critic(state)                       # [1, 1]

        # Extract Python scalars and numpy array before GPU tensors go out of scope
        lp1       = float(lp1_t.item())
        lp2       = float(lp2_t.item())
        ac2_int   = int(ac2_t.item())
        value_f   = float(value_t.item())
        # job_feat_np: move to CPU (already there if device="cpu"), then numpy
        job_feat_np = job_feat_t.squeeze(0).squeeze(0).cpu().numpy()   # [JOB_FEATURES]

        return ac1_int, lp1, ac2_int, lp2, value_f, job_feat_np

    @torch.no_grad()
    def eval_action(
        self,
        obs_flat:  np.ndarray,
        inv_mask1: np.ndarray,
        inv_mask2: np.ndarray,
    ) -> tuple[int, int]:
        """Greedy action selection for evaluation (no storage, no gradient)."""
        state = torch.as_tensor(
            obs_flat.reshape(1, TOTAL_ROWS, JOB_FEATURES),
            device=self.device,
        )
        m1 = torch.as_tensor(
            inv_mask1.reshape(1, MAX_QUEUE_SIZE),
            device=self.device,
        )
        m2 = torch.as_tensor(
            inv_mask2.reshape(1, ACTION2_NUM),
            device=self.device,
        )
        logits1  = self.actor.get_job_logits(state, m1)
        ac1      = int(logits1.argmax(dim=-1).item())
        job_feat = state[:, ac1:ac1 + 1, :]
        logits2  = self.actor.get_delay_logits(state, job_feat, m2)
        ac2      = int(logits2.argmax(dim=-1).item())
        return ac1, ac2

    # Experience recording 

    def remember(
        self,
        obs_flat:  np.ndarray,
        inv_mask1: np.ndarray,
        inv_mask2: np.ndarray,
        action1:   int,
        action2:   int,
        log_prob1: float,
        log_prob2: float,
        value:     float,
        reward:    float,
        job_feat:  np.ndarray,   # [JOB_FEATURES]
    ) -> None:
        """
        Write one step into the pre-allocated buffer (O(1), zero allocation).

        All arguments must be numpy arrays or Python scalars.
        This method intentionally accepts no torch.Tensor to prevent accidental
        GPU↔CPU transfers in the hot loop.
        """
        self.buffer.add(
            obs_flat, inv_mask1, inv_mask2,
            action1, action2,
            log_prob1, log_prob2,
            value, reward,
            job_feat,
        )

    def commit_trajectory(self, last_reward: float) -> None:
        """
        Finalise GAE for the current trajectory.

        Delegates to buffer.finish_path() — numpy only, no device transfer.
        """
        self.buffer.finish_path(float(last_reward), self.gamma, self.lam)

    # PPO Update 

    def train(self) -> None:
        """
        Multi-action PPO gradient update (GAS-MARL Eq. 12).

        Composite ratio: r(θ) = exp[(log π(a1|s) + log π(a2|s,a1))
                                  − (log π_old(a1|s) + log π_old(a2|s,a1))]

        Optimisation flow:
          1. buffer.get_tensors(device) — ONE batch GPU upload.
          2. Normalise advantages in-place on device.
          3. Mini-batch PPO epochs with integer indexing.
        """
        if len(self.buffer) == 0:
            return

        # Single batch GPU transfer 
        batch      = self.buffer.get_tensors(self.device)
        states     = batch["states"]       # [N, TOTAL_ROWS, JOB_FEATURES]
        masks1     = batch["masks1"]       # [N, MAX_QUEUE_SIZE] — invalid mask
        masks2     = batch["masks2"]       # [N, ACTION2_NUM]    — invalid mask
        actions1   = batch["actions1"]     # [N]
        actions2   = batch["actions2"]     # [N]
        old_lp1    = batch["log_probs1"]   # [N]
        old_lp2    = batch["log_probs2"]   # [N]
        returns    = batch["returns"]      # [N]
        advantages = batch["advantages"]   # [N]
        job_inputs = batch["job_inputs"]   # [N, 1, JOB_FEATURES]

        # Normalise advantages on device (single GPU pass)
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-9)

        n = len(self.buffer)
        for _ in range(self.ppo_epochs):
            for idx in BatchSampler(SubsetRandomSampler(range(n)), self.batch_size, False):
                idx_t = torch.tensor(idx, device=self.device)

                s   = states[idx_t]      # [B, TOTAL_ROWS, JOB_FEATURES]
                m1  = masks1[idx_t]      # [B, MAX_QUEUE_SIZE]
                m2  = masks2[idx_t]      # [B, ACTION2_NUM]
                a1  = actions1[idx_t]    # [B]
                a2  = actions2[idx_t]    # [B]
                olp1= old_lp1[idx_t]     # [B]
                olp2= old_lp2[idx_t]     # [B]
                ret = returns[idx_t]     # [B]
                adv = advantages[idx_t]  # [B]
                ji  = job_inputs[idx_t]  # [B, 1, JOB_FEATURES]

                # Recompute action probabilities 
                # Job selection
                logits1  = self.actor.get_job_logits(s, m1)
                probs1   = F.softmax(logits1, dim=-1)
                dist1    = Categorical(probs=probs1)
                new_lp1  = dist1.log_prob(a1)
                ent1     = dist1.entropy()

                # Delay decision (conditioned on stored job features)
                logits2  = self.actor.get_delay_logits(s, ji, m2)
                probs2   = F.softmax(logits2, dim=-1)
                dist2    = Categorical(probs=probs2)
                new_lp2  = dist2.log_prob(a2)
                ent2     = dist2.entropy()

                # Composite PPO ratio (GAS-MARL Eq. 12) 
                old_composite = olp1 + olp2
                new_composite = new_lp1 + new_lp2
                ratio   = torch.exp(new_composite - old_composite)
                surr1   = ratio * adv
                surr2   = torch.clamp(
                    ratio, 1 - self.clip_param, 1 + self.clip_param
                ) * adv
                entropy = (ent1 + ent2) / 2.0
                actor_loss = (
                    -torch.min(surr1, surr2).mean()
                    - self.entropy_coef * entropy.mean()
                )

                self.actor_opt.zero_grad()
                actor_loss.backward()
                nn.utils.clip_grad_norm_(
                    self.actor.parameters(), self.max_grad_norm
                )
                self.actor_opt.step()

                # Critic 
                val   = self.critic(s).squeeze(-1)
                closs = F.mse_loss(val, ret)
                self.critic_opt.zero_grad()
                closs.backward()
                nn.utils.clip_grad_norm_(
                    self.critic.parameters(), self.max_grad_norm
                )
                self.critic_opt.step()

    # Persistence 

    def save(self, directory: str) -> None:
        Path(directory).mkdir(parents=True, exist_ok=True)
        torch.save(self.actor.state_dict(),  os.path.join(directory, "actor.pt"))
        torch.save(self.critic.state_dict(), os.path.join(directory, "critic.pt"))

    def load(self, directory: str, map_location: str = "cpu") -> None:
        self.actor.load_state_dict(
            torch.load(os.path.join(directory, "actor.pt"),
                       map_location=map_location))
        self.critic.load_state_dict(
            torch.load(os.path.join(directory, "critic.pt"),
                       map_location=map_location))
        self.actor.eval()
        self.critic.eval()


# Training Entry Point 

def train_gas_marl(
    env_config:          Optional[EnvConfig] = None,
    save_dir:            str   = "models/gas_marl",
    epochs:              int   = 300,
    traj_num:            int   = 100,
    device:              str   = "auto",
    csv_log:             Optional[str] = None,
    verbose:             bool  = True,
    checkpoint_interval: int   = 50,
    resume_from:         Optional[str] = None,
    log_interval:        int   = 10,
    save_best:           bool  = True,
) -> GASMARLAgent:
    """
    Train a GAS-MARL agent on HPCGreenEnv.

    Performance notes
    
    The training hot loop no longer creates any torch.FloatTensor() objects or
    calls .to("cpu"/"cuda") per step:
      - choose_action() → returns Python scalars + numpy job_feat
      - remember()      → writes numpy obs directly into pre-allocated buffer
      - commit_trajectory() → GAE on numpy, no tensors
      - agent.train()       → single GPU batch upload, mini-batch indexing

    Args:
        env_config:            environment config (workload, cluster, renewable)
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

    env = HPCGreenEnv(env_config)

    # Size the buffer to fit one full epoch without overflow
    seq_len     = getattr(env_config, "seq_len", 256) if env_config else 256
    buffer_size = max(traj_num * (seq_len + 16), 32_768)

    agent = GASMARLAgent(device=device, buffer_size=buffer_size)

    # Resume from checkpoint 
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

    # CSV log 
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
        print(f"\n  {'='*60}")
        print(f"  [GAS-MARL]    Device={device.upper()}  "
              f"epochs={epochs}  traj/epoch={traj_num}")
        print(f"  buffer_size={buffer_size:,}  "
              f"checkpoint_interval={checkpoint_interval}  "
              f"save_best={save_best}  log_interval={log_interval}")
        print(f"  {'='*60}")
        print(f"  {'Epoch':>6}  {'Reward':>10}  {'ReUtil':>9}  "
              f"{'AvgBSLD':>9}  {'ETA':>8}")
        print(f"  {''*54}")

    for epoch in range(start_epoch, epochs):
        epoch_rewards, epoch_green, epoch_bsld = 0.0, 0.0, 0.0

        # Epoch loop 
        agent.buffer.clear()   # reset ptr, keep allocation

        t         = 0          # completed trajectories this epoch
        obs       = env.reset()
        green_rwd = 0.0

        while True:
            # Decision step 
            valid_mask = env.action_mask1()        # [MAX_QUEUE_SIZE] 1 = valid
            inv_mask1  = (1.0 - valid_mask).astype(np.float32)   # 1 = INVALID
            inv_mask2  = env.action_mask2().astype(np.float32)   # 1 = INVALID

            # choose_action() → all Python scalars + numpy job_feat_np
            ac1, lp1, ac2, lp2, value, job_feat_np = agent.choose_action(
                obs, inv_mask1, inv_mask2
            )

            # remember() writes numpy + scalars into pre-allocated buffer.
            # green_rwd is the reward from the PREVIOUS step (delayed reward).
            agent.remember(
                obs, inv_mask1, inv_mask2,
                ac1, ac2,
                lp1, lp2,
                value, green_rwd,
                job_feat_np,
            )

            obs, r, done, bsld_r, _, _, _, green_rwd = env.step(ac1, ac2)
            epoch_rewards += r
            epoch_green   += green_rwd

            if done:
                t += 1
                agent.commit_trajectory(r)
                epoch_bsld += abs(bsld_r)
                obs       = env.reset()
                green_rwd = 0.0
                if t >= traj_num:
                    break

        # PPO update: single GPU batch upload + mini-batch gradient steps 
        agent.train()
        agent.buffer.clear()   # reset ptr — arrays stay allocated

        avg_rew   = epoch_rewards / traj_num
        avg_green = epoch_green   / traj_num
        avg_bsld  = epoch_bsld    / traj_num
        elapsed   = _time.time() - t_start

        # Console log with progress bar 
        if verbose and ((epoch + 1) % log_interval == 0
                        or epoch == epochs - 1
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

        # CSV row 
        _w.writerow([epoch, f"{avg_rew:.6f}", f"{avg_green:.6f}",
                     f"{avg_bsld:.6f}", f"{elapsed:.1f}", device])
        _f.flush()

        # Periodic checkpoint 
        if checkpoint_interval > 0 and (epoch + 1) % checkpoint_interval == 0:
            ckpt_path = ckpt_dir / f"epoch_{epoch+1:04d}"
            agent.save(str(ckpt_path))
            meta = {
                "epoch": epoch, "avg_reward": avg_rew,
                "avg_green": avg_green, "avg_bsld": avg_bsld,
                "elapsed_sec": round(elapsed, 1),
            }
            (ckpt_path / "meta.json").write_text(_json.dumps(meta, indent=2))
            if verbose:
                print(f"  ✓ Checkpoint → {ckpt_path}")

        # Best-model tracking 
        if save_best and avg_rew > best_reward:
            best_reward = avg_rew
            best_path   = ckpt_dir / "best"
            agent.save(str(best_path))
            meta = {
                "epoch": epoch, "avg_reward": avg_rew,
                "avg_green": avg_green, "avg_bsld": avg_bsld,
            }
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
            n = sum(
                1 for p in ckpt_dir.iterdir()
                if p.is_dir() and p.name.startswith("epoch_")
            )
            print(f"  Checkpoints  → {ckpt_dir}  ({n} periodic + 1 best)")

    return agent


# Scheduler Wrapper 

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
        cluster:    Cluster,
        model_dir:  Optional[str]   = None,
        device:     str             = "cpu",
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
        """
        Core scheduling decision for GAS-MARL.

        Returns a SchedulingDecision that:
        - Schedules the selected job immediately (ac2 == 0), OR
        - Records a delay for the selected job (ac2 > 0) and sets
          decision.delay_info so that BackfillWrapper can compute the
          correct Green-Backfilling window (see scheduler/backfill.py §4.3).
        """
        decision = SchedulingDecision()
        if not pending:
            return decision

        # Only consider jobs whose delay window has expired
        active_pending = [
            j for j in pending
            if self._delayed_until.get(j.job_id, 0.0) <= current_time
        ]
        if not active_pending:
            return decision

        # Feed state into agent
        self._sync_env(active_pending, running, current_time)
        obs       = self._env._get_obs()
        inv_mask1 = (1.0 - self._env.action_mask1()).astype(np.float32)
        inv_mask2 = self._env.action_mask2().astype(np.float32)

        ac1, ac2 = self._agent.eval_action(obs, inv_mask1, inv_mask2)
        ac1 = min(ac1, len(active_pending) - 1)
        selected_job = active_pending[ac1]

        if ac2 > 0:
            # Delay decision 
            release_t = self._compute_release_time(ac2, running, current_time)
            self._delayed_until[selected_job.job_id] = release_t
            decision.delay_info = {
                "delay_type":    ac2,
                "release_time":  release_t,
                "head_job_id":   selected_job.job_id,
                "head_req_gpus": getattr(selected_job, "num_gpus_requested", 1),
            }
        else:
            # Immediate schedule 
            self._delayed_until.pop(selected_job.job_id, None)
            gids = self._find_gpus(selected_job, prefer_consolidated=True)
            if gids:
                decision.add(selected_job, gids)

        return decision

    def _compute_release_time(
        self, ac2: int, running: list, current_time: float
    ) -> float:
        """Translate action index → absolute release timestamp."""
        if ac2 <= DELAY_MAX_JOB_NUM:
            n_wait = min(ac2, len(running))
            if n_wait > 0 and running:
                return min(current_time + 3600.0,
                           current_time + 300.0 * n_wait)
            return current_time
        dt_idx = ac2 - (DELAY_MAX_JOB_NUM + 1)
        dt_idx = min(dt_idx, len(DELAY_TIMES) - 1)
        return current_time + DELAY_TIMES[dt_idx]

    def _gpu_count_for_job(self, job) -> int:
        n = getattr(job, "num_gpus_requested", 1)
        return max(1, min(n, self.cluster.total_gpus()))

    def _sync_env(self, pending, running, current_time: float) -> None:
        self._env._current_time = current_time
        self._env._pending      = list(pending)[:MAX_QUEUE_SIZE]
        self._env._avail_gpus   = self.cluster.free_gpu_count()
        self._env._running      = []
        for job in running:
            rj = _RunningJob(
                job_id=job.job_id,
                start_time=getattr(job, "start_time", current_time) or current_time,
                finish_time=current_time + 600.0,
                num_gpus=len(getattr(job, "allocated_gpus", [])) or 1,
                num_cpus=0,
                num_mig=0,
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
