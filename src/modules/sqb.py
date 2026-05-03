"""Sequence block (SQB) combining ESM embeddings with DCCN features."""

from __future__ import annotations

from typing import Optional

import torch
import torch.nn as nn

from src.modules.dccn import DCCN_1D


class SQBBlock(nn.Module):
    """Sequence block producing residue-level features.

    Args:
        input_dim: Dimension of incoming ESM embeddings (e).
        channels: Output channel size (c).
        kernel_size: DCCN kernel size.
        dilation: DCCN dilation factor.
        dropout: DCCN dropout rate.
    """

    def __init__(
        self,
        input_dim: int,
        channels: int,
        kernel_size: int,
        dilation: int,
        dropout: float,
    ) -> None:
        super().__init__()
        self.channels = int(channels)
        if input_dim != channels:
            self.input_proj = nn.Linear(input_dim, channels, bias=False)
            nn.init.xavier_uniform_(self.input_proj.weight)
        else:
            self.input_proj = nn.Identity()
        self.dccn = DCCN_1D(
            embed_len=channels,
            k_size=kernel_size,
            dilation=dilation,
            dropout=dropout,
        )
        self.dccn_norm = nn.LayerNorm(channels)
        self.gate = nn.Linear(2 * channels, channels)
        nn.init.xavier_uniform_(self.gate.weight)
        nn.init.zeros_(self.gate.bias)
        self.out_norm = nn.LayerNorm(channels)

    def forward(
        self,
        seq_embeddings: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Compute SQB features from ESM embeddings.

        Args:
            seq_embeddings: Tensor with shape (batch, length, input_dim).
            mask: Optional boolean tensor with shape (batch, length).

        Returns:
            Tensor with shape (batch, length, channels).
        """
        if seq_embeddings.ndim != 3:
            raise ValueError("seq_embeddings must be 3D (batch, length, dim).")
        if mask is not None and mask.shape != seq_embeddings.shape[:2]:
            raise ValueError("mask must match (batch, length).")

        projected = self.input_proj(seq_embeddings)
        dccn_out = self.dccn(projected, mask=mask)
        dccn_out = self.dccn_norm(dccn_out)

        gate = torch.sigmoid(self.gate(torch.cat([projected, dccn_out], dim=-1)))
        fused = gate * projected + (1.0 - gate) * dccn_out
        fused = self.out_norm(fused)

        if mask is not None:
            fused = fused * mask.unsqueeze(-1).to(fused.dtype)
        return fused
