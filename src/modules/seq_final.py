"""
seq_final.py

Builds joint protein-function features for downstream adaptive diffusion.
"""

from __future__ import annotations

from typing import Literal, Tuple

import torch
import torch.nn as nn

SeqFinalMode = Literal["alternating", "decoupled"]


class SeqFinal(nn.Module):
    """Project pooled sequence representations into joint protein/GO features.

    Args:
        in_dim: Dimension of the shared sequence representation (d_shared).
        N_C: Number of GO classes in the prediction task.
        proj: Intermediate projection width for the learned GO embeddings.
        out_ch: Width of the shared protein-function feature tensor C.
        decoupled: Whether decoupled projections (function/protein specific)
            should be produced instead of a shared tensor.
        function_dim: Output channels for the function-specific tensor when
            ``decoupled`` is True. Defaults to ``out_ch``.
        protein_dim: Output channels for the protein-specific tensor when
            ``decoupled`` is True. Defaults to ``out_ch``.
    """

    def __init__(
        self,
        in_dim: int,
        N_C: int,
        proj: int = 64,
        out_ch: int = 64,
        *,
        decoupled: bool = False,
        function_dim: int | None = None,
        protein_dim: int | None = None,
    ) -> None:
        super().__init__()
        if in_dim <= 0 or out_ch <= 0:
            raise ValueError("in_dim and out_ch must be positive integers.")
        if proj <= 0 or N_C <= 0:
            raise ValueError("proj and N_C must be positive integers.")

        self.num_classes = N_C
        self.out_ch = out_ch
        self.decoupled = decoupled
        self.function_dim = function_dim or out_ch
        self.protein_dim = protein_dim or out_ch

        self.protein_proj = nn.Linear(in_dim, out_ch, bias=True)
        self.G = nn.Parameter(torch.empty(N_C, proj))
        self.go_proj = nn.Linear(proj, out_ch, bias=True)

        comp_in = 2 * out_ch
        self.mlp = nn.Sequential(
            nn.Linear(comp_in, 2 * comp_in),
            nn.GELU(),
            nn.Linear(2 * comp_in, out_ch),
        )
        self.norm = nn.LayerNorm(out_ch)

        if self.decoupled:
            self.function_head = nn.Linear(out_ch, self.function_dim)
            self.protein_head = nn.Linear(out_ch, self.protein_dim)
        else:
            self.function_head = None
            self.protein_head = None

        self._init_weights()

    def _init_weights(self) -> None:
        nn.init.xavier_uniform_(self.protein_proj.weight)
        nn.init.zeros_(self.protein_proj.bias)
        nn.init.xavier_uniform_(self.G)
        nn.init.xavier_uniform_(self.go_proj.weight)
        nn.init.zeros_(self.go_proj.bias)

        for layer in self.mlp:
            if isinstance(layer, nn.Linear):
                nn.init.xavier_uniform_(layer.weight)
                nn.init.zeros_(layer.bias)

        if self.function_head is not None:
            nn.init.xavier_uniform_(self.function_head.weight)
            nn.init.zeros_(self.function_head.bias)
        if self.protein_head is not None:
            nn.init.xavier_uniform_(self.protein_head.weight)
            nn.init.zeros_(self.protein_head.bias)

    def forward(
        self,
        proteins: torch.Tensor,
        *,
        mode: SeqFinalMode = "alternating",
    ) -> torch.Tensor | Tuple[torch.Tensor, torch.Tensor]:
        """Construct joint protein-function features.

        Args:
            proteins: Tensor of shape (N_P, d_shared).
            mode: ``"alternating"`` returns a shared tensor (N_P, N_C, C).
                ``"decoupled"`` returns two tensors (function, protein).

        Returns:
            Either a shared tensor of shape (N_P, N_C, C) or a tuple containing
            (function_tensor, protein_tensor) for the decoupled mode.
        """

        shared = self._build_shared_tensor(proteins)

        if mode == "alternating":
            return shared

        if mode != "decoupled":
            raise ValueError(f"Unknown SeqFinal mode '{mode}'.")
        if not self.decoupled or self.function_head is None or self.protein_head is None:
            raise RuntimeError("SeqFinal decoupled mode requested but decoupled projections are disabled.")

        function_tensor = self.function_head(shared)
        protein_tensor = self.protein_head(shared)
        return function_tensor, protein_tensor

    def _build_shared_tensor(self, proteins: torch.Tensor) -> torch.Tensor:
        if proteins.ndim != 2:
            raise ValueError("proteins must be a 2D tensor (batch, d_shared).")

        P = self.protein_proj(proteins)  # (N_P, C)
        G = self.go_proj(self.G)  # (N_C, C)

        sum_term = P.unsqueeze(1) + G.unsqueeze(0)
        prod_term = P.unsqueeze(1) * G.unsqueeze(0)
        feats = torch.cat([sum_term, prod_term], dim=-1)  # (N_P, N_C, 2C)
        shared = self.norm(self.mlp(feats))
        return shared
