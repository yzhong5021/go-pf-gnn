"""
head.py

Feature fusion + classification head for decoupled graph stacks.
"""

from __future__ import annotations

from typing import Tuple

import torch
import torch.nn as nn

from src.modules.pooling import AdaptivePooling


class _Proj(nn.Module):
    def __init__(self, d_in: int, p_drop: float = 0.1) -> None:
        super().__init__()
        self.l1 = nn.Linear(d_in, d_in)
        self.gelu = nn.GELU()
        self.drop = nn.Dropout(p_drop)
        self.l2 = nn.Linear(d_in, d_in)
        self.norm = nn.LayerNorm(d_in)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.norm(self.l2(self.drop(self.gelu(self.l1(x)))) + x)


class ClassificationHead(nn.Module):
    """Attention-pool protein/function tensors before bilinear scoring."""

    def __init__(
        self,
        N_C: int,
        d_in: int,
        *,
        protein_input_dim: int | None = None,
        function_input_dim: int | None = None,
        dropout: float = 0.1,
        attn_hidden: int = 128,
    ) -> None:
        super().__init__()
        self.num_classes = N_C
        self.d_in = d_in
        self.protein_input_dim = protein_input_dim or d_in
        self.function_input_dim = function_input_dim or d_in

        self.protein_pool = AdaptivePooling(
            embed_dim=self.protein_input_dim,
            attn_hidden=attn_hidden,
            dropout=dropout,
        )
        self.function_pool = AdaptivePooling(
            embed_dim=self.function_input_dim,
            attn_hidden=attn_hidden,
            dropout=dropout,
        )
        self.protein_linear = nn.Linear(self.protein_input_dim, d_in)
        self.function_linear = nn.Linear(self.function_input_dim, d_in)

        self.p_proj = _Proj(d_in, p_drop=dropout)
        self.f_proj = _Proj(d_in, p_drop=dropout)

        self.bias = nn.Parameter(torch.zeros(N_C))
        self.raw_tau = nn.Parameter(torch.zeros(()))

        self._init_weights()

    def _init_weights(self) -> None:
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
        nn.init.zeros_(self.bias)
        nn.init.zeros_(self.raw_tau)

    def forward(
        self,
        function_tensor: torch.Tensor,
        protein_tensor: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Return logits and pooled embeddings.

        Args:
            function_tensor: (N_P, N_C, C_f) or (N_C, tokens, C_f) tensor.
            protein_tensor: (N_P, N_C, C_p) or (N_P, tokens, C_p) tensor.

        Returns:
            logits, protein_embeddings, function_embeddings
        """

        protein_embeddings = self._prepare_protein_embeddings(protein_tensor)
        function_embeddings = self._prepare_function_embeddings(function_tensor)

        protein_embeddings = self.p_proj(protein_embeddings)
        function_embeddings = self.f_proj(function_embeddings)

        t_tau = torch.tanh(self.raw_tau) * 4.0
        
        scale = protein_embeddings.size(-1) ** 0.5                                                                                         
        raw_scores = protein_embeddings @ function_embeddings.T                                                                                                                                                       
        logits = torch.exp(t_tau) * raw_scores / scale
        logits = torch.clamp(logits, -50, 50)
        bias = torch.clamp(self.bias, -100.0, 100.0)
        logits = logits + bias

        return logits, protein_embeddings, function_embeddings

    def _prepare_protein_embeddings(self, tensor: torch.Tensor) -> torch.Tensor:
        pooled = self._pool_tensor(
            tensor, expected_dim=self.protein_input_dim, axis="protein"
        )
        return self.protein_linear(pooled)

    def _prepare_function_embeddings(self, tensor: torch.Tensor) -> torch.Tensor:
        pooled = self._pool_tensor(
            tensor, expected_dim=self.function_input_dim, axis="function"
        )
        return self.function_linear(pooled)

    def _pool_tensor(
        self,
        tensor: torch.Tensor,
        *,
        expected_dim: int,
        axis: str,
    ) -> torch.Tensor:
        if tensor.ndim == 2:
            if tensor.size(-1) != expected_dim:
                raise ValueError(
                    f"Expected feature dim {expected_dim}, got {tensor.size(-1)}."
                )
            return tensor

        if tensor.ndim != 3:
            raise ValueError("Inputs must be 2D or 3D tensors.")

        if axis == "protein":
            if tensor.size(-1) != expected_dim:
                raise ValueError(
                    f"Expected protein feature dim {expected_dim}, got {tensor.size(-1)}."
                )
            return self.protein_pool(tensor)

        # axis == "function"
        if tensor.size(-1) != expected_dim:
            raise ValueError(
                f"Expected function feature dim {expected_dim}, got {tensor.size(-1)}."
            )
        if tensor.size(0) == self.num_classes:
            tokens_first = tensor
        else:
            tokens_first = tensor.permute(1, 0, 2)
        return self.function_pool(tokens_first)
