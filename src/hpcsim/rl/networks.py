"""
Neural Network Architectures for Green-Aware HPC Schedulers.

State space: (MAX_QUEUE_SIZE + RUN_WIN + GREEN_WIN + CLUSTER_WIN) × JOB_FEATURES
             = (64 + 32 + 24 + 1) × 12 = 121 × 12 = 1452 features

Architecture overview:
  - Unified 12-dim feature vector across all rows (queue / running / green / cluster)
  - JobEncoder: per-row MLP (shared weights) → d_model-dim embedding
  - Row-wise encodings are aggregated by section, then combined for value/policy
  - GASMARLActor: two-head design — job selection + delay decision
"""

from __future__ import annotations
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Categorical

from .env import (
    MAX_QUEUE_SIZE, RUN_WIN, GREEN_WIN, CLUSTER_WIN, TOTAL_ROWS,
    JOB_FEATURES, ACTION2_NUM,
)

# Total rows in observation tensor
TOTAL_ROWS = MAX_QUEUE_SIZE + RUN_WIN + GREEN_WIN + CLUSTER_WIN


# ─── Masked Categorical ───────────────────────────────────────────────────────

class CategoricalMasked(Categorical):
    """Categorical distribution with boolean action masking."""

    def __init__(self, logits: torch.Tensor, masks: torch.Tensor, device):
        self.masks = masks.bool().to(device)
        masked_logits = torch.where(
            self.masks, logits,
            torch.tensor(-1e8, device=device, dtype=logits.dtype),
        )
        super().__init__(logits=masked_logits)

    def entropy(self) -> torch.Tensor:
        p_log_p = self.logits * self.probs
        return -p_log_p.sum(dim=-1)


# ─── Shared per-row Encoder ───────────────────────────────────────────────────

class _RowEncoder(nn.Module):
    """
    Per-row MLP with shared weights.
    Applied identically to every row in the observation (queue / running / green / cluster).
    """
    def __init__(self, in_feat: int = JOB_FEATURES, d_model: int = 128):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_feat, 64),  nn.LayerNorm(64),  nn.ReLU(),
            nn.Linear(64, d_model),  nn.LayerNorm(d_model), nn.ReLU(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x: [..., in_feat] → [..., d_model]"""
        return self.net(x)


# ─── Aggregator ───────────────────────────────────────────────────────────────

class _SectionAggregator(nn.Module):
    """
    Summarise a section (multiple rows) into a single vector via mean + max pooling.
    """
    def __init__(self, d_model: int = 128, out_dim: int = 64):
        super().__init__()
        self.proj = nn.Sequential(
            nn.Linear(d_model * 2, out_dim), nn.ReLU(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x: [B, N_rows, d_model] → [B, out_dim]"""
        mean_pool = x.mean(dim=1)
        max_pool  = x.max(dim=1).values
        return self.proj(torch.cat([mean_pool, max_pool], dim=-1))


# ─── MaskablePPO Networks ─────────────────────────────────────────────────────

class MaskablePPOActor(nn.Module):
    """
    Actor for MaskablePPO.
    Produces one logit per queue slot → [B, MAX_QUEUE_SIZE].

    Architecture:
      - Shared _RowEncoder applied to all 121 rows
      - Per-queue-slot head scores each job
      - Context (running + green + cluster) injected as bias
    """

    def __init__(self, d_model: int = 128):
        super().__init__()
        self.d_model    = d_model
        self.row_enc    = _RowEncoder(JOB_FEATURES, d_model)
        self.ctx_agg    = _SectionAggregator(d_model, 64)

        # Per-job scoring head (context-conditioned)
        self.job_head = nn.Sequential(
            nn.Linear(d_model + 64, 64), nn.ReLU(),
            nn.Linear(64, 16),           nn.ReLU(),
            nn.Linear(16, 1),
        )

    def forward(self, obs: torch.Tensor) -> torch.Tensor:
        """
        obs: [B, TOTAL_ROWS, JOB_FEATURES]
        returns logits: [B, MAX_QUEUE_SIZE]
        """
        # Encode all rows with shared weights
        enc = self.row_enc(obs)                               # [B, TOTAL_ROWS, d_model]

        # Context: running + green + cluster sections
        ctx_enc = enc[:, MAX_QUEUE_SIZE:, :]                  # [B, R+G+C, d_model]
        ctx     = self.ctx_agg(ctx_enc)                       # [B, 64]

        # Score each queue slot
        q_enc   = enc[:, :MAX_QUEUE_SIZE, :]                  # [B, Q, d_model]
        ctx_exp = ctx.unsqueeze(1).expand(-1, MAX_QUEUE_SIZE, -1)   # [B, Q, 64]
        logits  = self.job_head(
            torch.cat([q_enc, ctx_exp], dim=-1)
        ).squeeze(-1)                                          # [B, Q]
        return logits


class MaskablePPOCritic(nn.Module):
    """
    Critic for MaskablePPO: estimates state value V(s).
    Processes all 121 rows and aggregates into a scalar.
    """

    def __init__(self, d_model: int = 128):
        super().__init__()
        self.row_enc = _RowEncoder(JOB_FEATURES, d_model)
        self.q_agg   = _SectionAggregator(d_model, 64)
        self.r_agg   = _SectionAggregator(d_model, 32)
        self.g_agg   = _SectionAggregator(d_model, 16)

        # Cluster state (1 row)
        self.clust_proj = nn.Linear(d_model, 16)

        # Value head
        combined_dim = 64 + 32 + 16 + 16
        self.value_head = nn.Sequential(
            nn.Linear(combined_dim, 64), nn.ReLU(),
            nn.Linear(64, 16),           nn.ReLU(),
            nn.Linear(16, 1),
        )

    def forward(self, obs: torch.Tensor) -> torch.Tensor:
        """obs: [B, TOTAL_ROWS, JOB_FEATURES] → scalar [B, 1]"""
        enc = self.row_enc(obs)                               # [B, TOTAL_ROWS, d]

        q_vec = self.q_agg(enc[:, :MAX_QUEUE_SIZE, :])       # [B, 64]
        r_vec = self.r_agg(enc[:, MAX_QUEUE_SIZE:MAX_QUEUE_SIZE + RUN_WIN, :])   # [B, 32]
        g_vec = self.g_agg(enc[:, MAX_QUEUE_SIZE + RUN_WIN:MAX_QUEUE_SIZE + RUN_WIN + GREEN_WIN, :])  # [B, 16]
        c_vec = self.clust_proj(enc[:, -1, :])                # [B, 16]

        return self.value_head(torch.cat([q_vec, r_vec, g_vec, c_vec], dim=-1))


# ─── GAS-MARL Networks ────────────────────────────────────────────────────────

class GASMARLActor(nn.Module):
    """
    Two-head actor for GAS-MARL:
      Head 1 — job selection logits [B, MAX_QUEUE_SIZE]
      Head 2 — delay decision logits [B, ACTION2_NUM]  (conditioned on selected job)
    """

    def __init__(self, d_model: int = 128):
        super().__init__()
        self.d_model  = d_model
        self.row_enc  = _RowEncoder(JOB_FEATURES, d_model)
        self.ctx_agg  = _SectionAggregator(d_model, 64)

        # Job selection head (same as MaskablePPO actor)
        self.job_head = nn.Sequential(
            nn.Linear(d_model + 64, 64), nn.ReLU(),
            nn.Linear(64, 16),           nn.ReLU(),
            nn.Linear(16, 1),
        )

        # Selected-job projection into context for delay head
        self.sel_proj = nn.Linear(JOB_FEATURES, d_model)

        # Delay decision head
        # Input: global context + selected job embedding
        self.delay_head = nn.Sequential(
            nn.Linear(64 + d_model, 64), nn.ReLU(),
            nn.Linear(64, 32),           nn.ReLU(),
            nn.Linear(32, ACTION2_NUM),
        )

    def encode(self, obs: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Returns (enc, ctx):
          enc: [B, TOTAL_ROWS, d_model]
          ctx: [B, 64]  — aggregated context (running + green + cluster)
        """
        enc = self.row_enc(obs)
        ctx = self.ctx_agg(enc[:, MAX_QUEUE_SIZE:, :])
        return enc, ctx

    def get_job_logits(
        self, obs: torch.Tensor, inv_mask: torch.Tensor
    ) -> torch.Tensor:
        """
        Job selection logits (masked).
        inv_mask: [B, Q] — 1.0 where action is INVALID
        """
        enc, ctx = self.encode(obs)
        q_enc    = enc[:, :MAX_QUEUE_SIZE, :]
        ctx_exp  = ctx.unsqueeze(1).expand(-1, MAX_QUEUE_SIZE, -1)
        logits   = self.job_head(
            torch.cat([q_enc, ctx_exp], dim=-1)
        ).squeeze(-1)
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
        _, ctx   = self.encode(obs)
        sel_enc  = self.sel_proj(selected_job_features.squeeze(1))  # [B, d_model]
        logits   = self.delay_head(
            torch.cat([ctx, sel_enc], dim=-1)
        )
        return logits - inv_mask * 1e9


class GASMARLCritic(nn.Module):
    """Critic for GAS-MARL (shared architecture with MaskablePPO critic)."""

    def __init__(self, d_model: int = 128):
        super().__init__()
        self.row_enc    = _RowEncoder(JOB_FEATURES, d_model)
        self.q_agg      = _SectionAggregator(d_model, 64)
        self.r_agg      = _SectionAggregator(d_model, 32)
        self.g_agg      = _SectionAggregator(d_model, 16)
        self.clust_proj = nn.Linear(d_model, 16)
        combined_dim    = 64 + 32 + 16 + 16
        self.value_head = nn.Sequential(
            nn.Linear(combined_dim, 64), nn.ReLU(),
            nn.Linear(64, 16),           nn.ReLU(),
            nn.Linear(16, 1),
        )

    def forward(self, obs: torch.Tensor) -> torch.Tensor:
        enc   = self.row_enc(obs)
        q_vec = self.q_agg(enc[:, :MAX_QUEUE_SIZE, :])
        r_vec = self.r_agg(enc[:, MAX_QUEUE_SIZE:MAX_QUEUE_SIZE + RUN_WIN, :])
        g_vec = self.g_agg(enc[:, MAX_QUEUE_SIZE + RUN_WIN:MAX_QUEUE_SIZE + RUN_WIN + GREEN_WIN, :])
        c_vec = self.clust_proj(enc[:, -1, :])
        return self.value_head(torch.cat([q_vec, r_vec, g_vec, c_vec], dim=-1))
