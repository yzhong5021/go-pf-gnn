"""Attention utilities for single-query cross-attention and learned pooling."""

from __future__ import annotations

import math
from typing import Optional, Tuple

import torch
import torch.nn as nn


class LearnedAttentionPooling(nn.Module):
    """Pool per-residue features with a learned vector.

    Args:
        input_dim: Feature dimension per residue.
    """

    def __init__(self, input_dim: int) -> None:
        super().__init__()
        self.weight = nn.Parameter(torch.zeros(input_dim))
        nn.init.normal_(self.weight, mean=0.0, std=0.02)

    def forward(
        self,
        features: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
        *,
        return_weights: bool = False,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """Pool residue features into a single vector.

        Args:
            features: Tensor with shape (batch, length, input_dim).
            mask: Optional boolean tensor with shape (batch, length).
            return_weights: Whether to return attention weights.

        Returns:
            Tuple of pooled tensor with shape (batch, input_dim) and optional
            weights with shape (batch, length).
        """
        if features.ndim != 3:
            raise ValueError("features must be 3D (batch, length, dim).")
        scores = torch.einsum("bld,d->bl", features, self.weight)
        if mask is not None:
            if mask.shape != features.shape[:2]:
                raise ValueError("mask must match (batch, length).")
            scores = scores.masked_fill(~mask, float("-inf"))
        weights = torch.softmax(scores.float(), dim=1).to(features.dtype)
        weights = torch.nan_to_num(weights, nan=0.0)
        pooled = torch.sum(features * weights.unsqueeze(-1), dim=1)
        if return_weights:
            return pooled, weights
        return pooled, None


class SingleQueryCrossAttention(nn.Module):
    """Single-query multi-head cross-attention over residue features.

    Args:
        channels: Input/output feature dimension (c).
        heads: Number of attention heads (h).
        head_dim: Per-head dimension (d), such that channels == heads * head_dim.
        dropout: Dropout rate applied after pre-norm and before the MLP.
    """

    def __init__(
        self,
        channels: int,
        heads: int,
        head_dim: int,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        if channels != heads * head_dim:
            raise ValueError("channels must equal heads * head_dim.")
        self.channels = int(channels)
        self.heads = int(heads)
        self.head_dim = int(head_dim)
        self.pre_norm = nn.LayerNorm(heads * head_dim)
        self.mlp = nn.Sequential(
            nn.Linear(heads * head_dim, channels),
            nn.ReLU(),
            nn.Linear(channels, channels),
        )
        for layer in self.mlp:
            if isinstance(layer, nn.Linear):
                nn.init.xavier_uniform_(layer.weight)
                if layer.bias is not None:
                    nn.init.zeros_(layer.bias)
        self.dropout = nn.Dropout(dropout)
        self.out_norm = nn.LayerNorm(channels)

    def forward(
        self,
        keys: torch.Tensor,
        values: torch.Tensor,
        query: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
        residual: Optional[torch.Tensor] = None,
        *,
        return_attention: bool = False,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """Apply single-query attention over residue features.

        Args:
            keys: Tensor with shape (batch, heads, length, head_dim).
            values: Tensor with shape (batch, heads, length, head_dim).
            query: Tensor with shape (batch, heads, head_dim).
            mask: Optional boolean tensor with shape (batch, length).
            residual: Optional tensor with shape (batch, length, channels).
            return_attention: Whether to return attention weights.

        Returns:
            Tuple of output tensor with shape (batch, length, channels) and
            optional attention weights with shape (batch, heads, length).
        """
        if keys.ndim != 4 or values.ndim != 4 or query.ndim != 3:
            raise ValueError("keys/values/query must be (B,H,L,D)/(B,H,D).")
        scale = 1.0 / math.sqrt(self.head_dim)
        scores = torch.sum(keys * query.unsqueeze(-2), dim=-1) * scale
        if mask is not None:
            if mask.shape != (keys.size(0), keys.size(2)):
                raise ValueError("mask must match (batch, length).")
            scores = scores.masked_fill(~mask.unsqueeze(1), float("-inf"))
        attn = torch.softmax(scores.float(), dim=-1).to(values.dtype)
        attn = torch.nan_to_num(attn, nan=0.0)
        context = attn.unsqueeze(-1) * values
        concat = context.permute(0, 2, 1, 3).reshape(
            context.size(0), context.size(2), -1
        )
        concat = self.pre_norm(concat)
        concat = self.dropout(concat)
        projected = self.mlp(concat)
        if residual is not None:
            projected = projected + residual
        output = self.out_norm(projected)
        if mask is not None:
            output = output * mask.unsqueeze(-1).to(output.dtype)
        if return_attention:
            return output, attn
        return output, None
