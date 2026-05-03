"""
Attention-based pooling utilities.

Reduces 3D affinity tensors to graph-ready 2D node embeddings using
learned token weights.
"""

from typing import Optional

import torch
import torch.nn as nn

class AdaptivePooling(nn.Module):
    """Attention pooling over token dimensions for graph initialisation.

    Args:
        embed_dim: Feature width of the input tensor (C).
        attn_hidden: Hidden size of the attention scorer MLP.
        dropout: Dropout applied to the token representations prior to scoring.
    """

    def __init__(self, embed_dim: int, attn_hidden: int = 128, dropout: float = 0.1) -> None:
        super().__init__()
        if embed_dim <= 0:
            raise ValueError("embed_dim must be a positive integer.")

        self.embed_dim = embed_dim
        self.scorer = nn.Sequential(
            nn.Linear(embed_dim, attn_hidden),
            nn.GELU(),
            nn.Linear(attn_hidden, 1),
        )
        self.dropout = nn.Dropout(p=dropout)
        self.norm = nn.LayerNorm(embed_dim)

        self._init_weights()

    def _init_weights(self) -> None:
        first = self.scorer[0]
        nn.init.xavier_uniform_(first.weight)
        nn.init.zeros_(first.bias)

        last = self.scorer[2]
        nn.init.xavier_uniform_(last.weight)
        nn.init.zeros_(last.bias)

    def forward(self, x: torch.Tensor, mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """Pool a stack of pairwise features into node-wise embeddings.

        Args:
            x: Tensor shaped (N, T, C) where N is the number of nodes,
                T the number of tokens being pooled, and C the feature width.
            mask: Optional boolean or float mask of shape (N, T) marking valid
                tokens with True/non-zero entries.

        Returns:
            Tensor shaped (N, C) with pooled, layer-normalised representations.
        """

        if x.ndim != 3:
            raise ValueError("Expected a 3D tensor (nodes, tokens, features).")
        if x.size(-1) != self.embed_dim:
            raise ValueError(
                f"Expected feature dimension {self.embed_dim}, got {x.size(-1)}."
            )

        attn_input = self.dropout(x)
        scores = self.scorer(attn_input)

        if mask is not None:
            if mask.shape != x.shape[:2]:
                raise ValueError("mask must match the first two dimensions of the input tensor.")
            mask_bool = mask.to(dtype=torch.bool).unsqueeze(-1)
            scores = scores.masked_fill(~mask_bool, float("-inf"))

        weights = torch.softmax(scores, dim=1)
        weights = torch.nan_to_num(weights, nan=0.0)

        pooled = (weights * x).sum(dim=1)
        return self.norm(pooled)
