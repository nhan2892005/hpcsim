"""
Neural Network Architectures for Green-Aware HPC Schedulers.

Adapted from uploaded MaskablePPO.py and MARL.py:
  - ActorNet / CriticNet shared backbone
  - GASMARLActor with two-head design (job selection + delay decision)
  - CategoricalMasked for action masking

Architecture extended for heterogeneous GPU clusters:
  - GPU-type affinity embedding in job encoder
  - Separate encoder for renewable energy forecast
  - Running-jobs encoder with completion-time awareness
"""

from __future__ import annotations
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Categorical

from .env import (
    MAX_QUEUE_SIZE, RUN_WIN, GREEN_WIN,
    JOB_FEATURES, RUN_FEATURES, GREEN_FEATURES,
    ACTION2_NUM,
)


# ─── Masked Categorical Distribution ─────────────────────────────────────────

class CategoricalMasked(Categorical):
    """Categorical distribution with boolean action masking."""

    def __init__(self, logits: torch.Tensor, masks: torch.Tensor, device):
        self.masks = masks.bool().to(device)
        # Set masked logits to -∞ so they get ~0 probability
        masked_logits = torch.where(self.masks, logits, torch.tensor(-1e8, device=device))
        super().__init__(logits=masked_logits)

    def entropy(self) -> torch.Tensor:
        p_log_p = self.logits * self.probs
        return -p_log_p.sum(dim=-1)


# ─── Shared Encoders ─────────────────────────────────────────────────────────

class _JobEncoder(nn.Module):
    def __init__(self, in_feat: int, d_model: int = 128):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_feat, 64), nn.ReLU(),
            nn.Linear(64, d_model), nn.ReLU(),
        )
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class _GreenEncoder(nn.Module):
    def __init__(self, in_feat: int = GREEN_FEATURES, d_model: int = 128):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_feat, 64), nn.ReLU(),
            nn.Linear(64, d_model), nn.ReLU(),
        )
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


# ─── MaskablePPO Networks ─────────────────────────────────────────────────────

class MaskablePPOActor(nn.Module):
    """
    Actor for MaskablePPO: selects one job from the waiting queue.
    Outputs logits of shape [batch, MAX_QUEUE_SIZE].
    """

    def __init__(self, d_model: int = 128):
        super().__init__()
        self.d_model = d_model
        self.job_encoder = _JobEncoder(JOB_FEATURES, d_model)
        self.decoder = nn.Sequential(
            nn.Linear(d_model, 64), nn.ReLU(),
            nn.Linear(64, 16),     nn.ReLU(),
            nn.Linear(16, 1),
        )

    def forward(self, obs: torch.Tensor) -> torch.Tensor:
        """
        obs: [B, MAX_QUEUE_SIZE + RUN_WIN + GREEN_WIN, JOB_FEATURES]
        returns logits: [B, MAX_QUEUE_SIZE]
        """
        jobs = obs[:, :MAX_QUEUE_SIZE, :]          # [B, Q, F]
        enc  = self.job_encoder(jobs)              # [B, Q, d_model]
        logits = self.decoder(enc).squeeze(-1)     # [B, Q]
        return logits


class MaskablePPOCritic(nn.Module):
    """
    Critic for MaskablePPO: estimates state value.
    Uses all three encoders (jobs + running + green).
    """

    def __init__(self, d_model: int = 128):
        super().__init__()
        self.d_model = d_model
        self.job_encoder     = _JobEncoder(JOB_FEATURES,  d_model)
        self.run_encoder     = _JobEncoder(JOB_FEATURES,  d_model)   # same feat dim
        self.green_encoder   = _GreenEncoder(JOB_FEATURES, d_model)

        total_slots = MAX_QUEUE_SIZE + RUN_WIN + GREEN_WIN
        self.hidden = nn.Sequential(
            nn.Linear(d_model, 32), nn.ReLU(),
            nn.Linear(32, 8),       nn.ReLU(),
        )
        self.out = nn.Sequential(
            nn.Flatten(),
            nn.Linear(total_slots * 8, 64), nn.ReLU(),
            nn.Linear(64, 8),               nn.ReLU(),
            nn.Linear(8, 1),
        )

    def forward(self, obs: torch.Tensor) -> torch.Tensor:
        """obs: [B, Q+R+G, F]"""
        jobs   = self.job_encoder(obs[:, :MAX_QUEUE_SIZE, :])
        runs   = self.run_encoder(obs[:, MAX_QUEUE_SIZE:MAX_QUEUE_SIZE + RUN_WIN, :])
        greens = self.green_encoder(obs[:, MAX_QUEUE_SIZE + RUN_WIN:, :])
        merged = torch.cat([jobs, runs, greens], dim=1)   # [B, total_slots, d]
        h      = self.hidden(merged)
        return self.out(h)


# ─── GAS-MARL Networks ────────────────────────────────────────────────────────

class GASMARLActor(nn.Module):
    """
    Two-head actor for GAS-MARL:
      Head 1 — job selection: logits [B, MAX_QUEUE_SIZE]
      Head 2 — delay decision: logits [B, ACTION2_NUM]
                (conditioned on the selected job embedding)
    """

    def __init__(self, d_model: int = 128):
        super().__init__()
        self.d_model = d_model

        # Shared encoders
        self.job_encoder   = _JobEncoder(JOB_FEATURES, d_model)
        self.run_encoder   = _JobEncoder(JOB_FEATURES, d_model)
        self.green_encoder = _GreenEncoder(JOB_FEATURES, d_model)

        # Job-selection head
        self.job_head = nn.Sequential(
            nn.Linear(d_model, 64), nn.ReLU(),
            nn.Linear(64, 16),     nn.ReLU(),
            nn.Linear(16, 1),
        )

        # Selected job projection (to combine with context for delay head)
        self.selected_proj = nn.Linear(JOB_FEATURES, d_model)

        total_slots = MAX_QUEUE_SIZE + RUN_WIN + GREEN_WIN + 1  # +1 for selected job
        self.delay_hidden = nn.Sequential(
            nn.Linear(d_model, 32), nn.ReLU(),
            nn.Linear(32, JOB_FEATURES), nn.ReLU(),
        )
        self.delay_head = nn.Sequential(
            nn.Flatten(),
            nn.Linear(total_slots * JOB_FEATURES, 64), nn.ReLU(),
            nn.Linear(64, ACTION2_NUM),
        )

    def encode(self, obs: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Encode all three streams. Returns job_enc, run_enc, green_enc."""
        job_enc   = self.job_encoder(obs[:, :MAX_QUEUE_SIZE, :])
        run_enc   = self.run_encoder(obs[:, MAX_QUEUE_SIZE:MAX_QUEUE_SIZE + RUN_WIN, :])
        green_enc = self.green_encoder(obs[:, MAX_QUEUE_SIZE + RUN_WIN:, :])
        return job_enc, run_enc, green_enc

    def get_job_logits(
        self, obs: torch.Tensor, inv_mask: torch.Tensor
    ) -> torch.Tensor:
        """
        Job selection logits (masked).
        inv_mask: [B, Q] — 1.0 where action is INVALID (subtract large number)
        """
        job_enc, _, _ = self.encode(obs)
        logits = self.job_head(job_enc).squeeze(-1)   # [B, Q]
        return logits - inv_mask * 1e9

    def get_delay_logits(
        self,
        obs: torch.Tensor,
        selected_job_features: torch.Tensor,
        inv_mask: torch.Tensor,
    ) -> torch.Tensor:
        """
        Delay decision logits conditioned on selected job.
        selected_job_features: [B, 1, JOB_FEATURES]
        inv_mask: [B, ACTION2_NUM] — 1.0 where INVALID
        """
        job_enc, run_enc, green_enc = self.encode(obs)
        sel_enc = self.selected_proj(selected_job_features)  # [B, 1, d_model]
        context = torch.cat([job_enc, run_enc, green_enc, sel_enc], dim=1)
        h = self.delay_hidden(context)
        logits = self.delay_head(h)                         # [B, ACTION2_NUM]
        return logits - inv_mask * 1e9


class GASMARLCritic(nn.Module):
    """Critic for GAS-MARL (same as MaskablePPO critic)."""

    def __init__(self, d_model: int = 128):
        super().__init__()
        self.d_model = d_model
        self.job_encoder   = _JobEncoder(JOB_FEATURES, d_model)
        self.run_encoder   = _JobEncoder(JOB_FEATURES, d_model)
        self.green_encoder = _GreenEncoder(JOB_FEATURES, d_model)

        total_slots = MAX_QUEUE_SIZE + RUN_WIN + GREEN_WIN
        self.hidden = nn.Sequential(
            nn.Linear(d_model, 32), nn.ReLU(),
            nn.Linear(32, JOB_FEATURES), nn.ReLU(),
        )
        self.out = nn.Sequential(
            nn.Flatten(),
            nn.Linear(total_slots * JOB_FEATURES, 64), nn.ReLU(),
            nn.Linear(64, 8),                           nn.ReLU(),
            nn.Linear(8, 1),
        )

    def forward(self, obs: torch.Tensor) -> torch.Tensor:
        jobs   = self.job_encoder(obs[:, :MAX_QUEUE_SIZE, :])
        runs   = self.run_encoder(obs[:, MAX_QUEUE_SIZE:MAX_QUEUE_SIZE + RUN_WIN, :])
        greens = self.green_encoder(obs[:, MAX_QUEUE_SIZE + RUN_WIN:, :])
        merged = torch.cat([jobs, runs, greens], dim=1)
        h = self.hidden(merged)
        return self.out(h)
