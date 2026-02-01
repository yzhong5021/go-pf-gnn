"""Prediction head combining attention and mean pooling with gating."""

from __future__ import annotations

from typing import Optional

import torch
import torch.nn as nn

from src.modules.attention import LearnedAttentionPooling


class PredictionHead(nn.Module):
    """Classification head for protein function prediction.

    Args:
        channels: Feature dimension (c).
        num_functions: Number of output labels.
        mlp_hidden_dim: Hidden dimension for the shallow MLP.
    """

    def __init__(
        self,
        channels: int,
        num_functions: int,
        mlp_hidden_dim: Optional[int] = None,
    ) -> None:
        super().__init__()
        self.channels = int(channels)
        hidden_dim = int(mlp_hidden_dim) if mlp_hidden_dim is not None else int(channels)
        self.pool_attn = LearnedAttentionPooling(channels)
        self.pool_gcn = LearnedAttentionPooling(channels)
        self.gate = nn.Linear(4 * channels, 4 * channels)
        nn.init.xavier_uniform_(self.gate.weight)
        nn.init.zeros_(self.gate.bias)
        self.mlp = nn.Sequential(
            nn.Linear(channels, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, num_functions),
        )
        for layer in self.mlp:
            if isinstance(layer, nn.Linear):
                nn.init.xavier_uniform_(layer.weight)
                if layer.bias is not None:
                    nn.init.zeros_(layer.bias)

    def forward(
        self,
        attn_features: torch.Tensor,
        gcn_features: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Compute classification logits.

        Args:
            attn_features: Tensor with shape (batch, length, channels).
            gcn_features: Tensor with shape (batch, length, channels).
            mask: Optional boolean tensor with shape (batch, length).

        Returns:
            Tensor with shape (batch, num_functions).
        """
        if attn_features.ndim != 3 or gcn_features.ndim != 3:
            raise ValueError("attn_features and gcn_features must be 3D.")
        if attn_features.shape != gcn_features.shape:
            raise ValueError("attn_features and gcn_features must have matching shapes.")
        if mask is not None and mask.shape != attn_features.shape[:2]:
            raise ValueError("mask must match (batch, length).")

        attn_pool, _ = self.pool_attn(attn_features, mask)
        gcn_pool, _ = self.pool_gcn(gcn_features, mask)
        attn_mean = self._masked_mean(attn_features, mask)
        gcn_mean = self._masked_mean(gcn_features, mask)

        combined = torch.cat([attn_pool, attn_mean, gcn_pool, gcn_mean], dim=-1)
        combined = combined.reshape(combined.size(0), 4, self.channels)
        gates = torch.sigmoid(self.gate(combined.flatten(1))).view_as(combined)
        gated = torch.sum(gates * combined, dim=1)
        return self.mlp(gated)

    @staticmethod
    def _masked_mean(features: torch.Tensor, mask: Optional[torch.Tensor]) -> torch.Tensor:
        if mask is None:
            return features.mean(dim=1)
        mask_f = mask.to(dtype=features.dtype).unsqueeze(-1)
        summed = torch.sum(features * mask_f, dim=1)
        denom = mask_f.sum(dim=1).clamp(min=1.0)
        return summed / denom
