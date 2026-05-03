"""
Adaptive diffusion operator shared by protein and function graph stacks.
"""

from __future__ import annotations

from typing import Literal, Optional, Tuple

import torch
import torch.nn as nn

from src.utils.adaptive_helpers import (
    _init_attention_layers,
    _init_linear_stack,
    _build_top_p_attention,
    _prepare_prior,
)

Axis = Literal["protein", "function"]


class AdaptiveDiffusionBlock(nn.Module):
    """Generalised adaptive diffusion over protein or function axes.

    The block reuses the bilinear attention and diffusion operators from the
    legacy AdaptiveProtein/AdaptiveFunction modules, but applies them to 3D
    tensors via einsum so weights are shared across the orthogonal axis.
    """

    def __init__(
        self,
        feature_dim: int,
        *,
        axis: Axis,
        d_attn: int = 64,
        steps: int = 2,
        p: float = 0.9,
        tau: float = 1.0,
        dropout: float = 0.1,
        bidirectional: bool = False,
    ) -> None:
        super().__init__()
        if feature_dim <= 0:
            raise ValueError("feature_dim must be positive.")
        if axis not in {"protein", "function"}:
            raise ValueError("axis must be either 'protein' or 'function'.")

        self.feature_dim = feature_dim
        self.axis = axis
        self.steps = steps
        self.p = float(p)
        self.tau = float(tau)
        self.bidirectional = bidirectional

        self.W1 = nn.Linear(feature_dim, d_attn, bias=False)
        self.W2 = nn.Linear(d_attn, d_attn, bias=False)
        self.W3 = nn.Linear(feature_dim, d_attn, bias=False)

        self.prior_forward = nn.ModuleList(
            [nn.Linear(feature_dim, feature_dim, bias=False) for _ in range(steps)]
        )
        self.prior_backward = (
            nn.ModuleList([nn.Linear(feature_dim, feature_dim, bias=False) for _ in range(steps)])
            if bidirectional
            else None
        )
        self.adaptive_stack = nn.ModuleList(
            [nn.Linear(feature_dim, feature_dim, bias=False) for _ in range(steps)]
        )

        self.dropout = nn.Dropout(dropout)
        self.norm = nn.LayerNorm(feature_dim)
        self.alpha = nn.Parameter(torch.zeros(1))
        self.last_attention: Optional[torch.Tensor] = None

        self._init_weights()

    def _init_weights(self) -> None:
        _init_attention_layers(self.W1, self.W2, self.W3)
        _init_linear_stack(self.prior_forward)
        _init_linear_stack(self.adaptive_stack)
        if self.prior_backward is not None:
            _init_linear_stack(self.prior_backward)

    def forward(
        self,
        x: torch.Tensor,
        prior: Optional[torch.Tensor] = None,
        mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Apply adaptive diffusion along the configured axis.

        Args:
            x: Tensor shaped (N_P, N_C, C).
            prior: Optional adjacency matrix along the diffusion axis.
            mask: Optional boolean/float mask for nodes along the axis.
        """

        if x.ndim != 3:
            raise ValueError("Input must be 3D (N_P, N_C, C).")

        nodes = x.size(0 if self.axis == "protein" else 1)
        node_mask = self._normalise_mask(mask, nodes, x.device)
        broadcast_mask = self._broadcast_mask(node_mask, x) if node_mask is not None else None

        prior_mats = _prepare_prior(
            self._validate_prior(prior, nodes, x.device, x.dtype),
            nodes,
            x.device,
            x.dtype,
            bidirectional=self.bidirectional,
        )
        if self.bidirectional:
            Rf, Rb = prior_mats
        else:
            Rf, Rb = prior_mats, None

        Xf = x.clone()
        Xa = x.clone()
        Xb = x.clone() if self.bidirectional else None
        residual = torch.zeros_like(x)

        if broadcast_mask is not None:
            Xf = Xf * broadcast_mask
            Xa = Xa * broadcast_mask
            if Xb is not None:
                Xb = Xb * broadcast_mask

        for idx in range(self.steps):
            pooled = self._pool_for_attention(Xa)
            attn = self._adj_from_feats(pooled)
            if node_mask is not None:
                attn = self._mask_attention(attn, node_mask)
            # Store a detached view to avoid retaining the full autograd graph.
            self.last_attention = attn.detach()

            Xf = self._apply_graph(Rf, Xf)
            if self.bidirectional and Rb is not None and Xb is not None:
                Xb = self._apply_graph(Rb, Xb)
            Xa = self._apply_graph(attn, Xa)

            delta = self.prior_forward[idx](Xf)
            if self.bidirectional and Xb is not None and self.prior_backward is not None:
                delta = delta + self.prior_backward[idx](Xb)
            delta = delta + self.adaptive_stack[idx](Xa)
            delta = self.dropout(delta)

            if broadcast_mask is not None:
                delta = delta * broadcast_mask
            residual = residual + delta

        gate = torch.sigmoid(self.alpha)
        out = self.norm(x + gate * residual)
        if broadcast_mask is not None:
            out = out * broadcast_mask
        return out

    # ----- internal helpers -----

    def _pool_for_attention(self, tensor: torch.Tensor) -> torch.Tensor:
        if self.axis == "protein":
            return tensor.mean(dim=1)
        return tensor.mean(dim=0)

    def _adj_from_feats(self, pooled: torch.Tensor) -> torch.Tensor:
        return _build_top_p_attention(self.W1, self.W2, self.W3, pooled, self.tau, self.p)

    def _apply_graph(self, mat: torch.Tensor, tensor: torch.Tensor) -> torch.Tensor:
        if self.axis == "protein":
            return torch.einsum("ij,jkc->ikc", mat, tensor)
        return torch.einsum("ij,pjc->pic", mat, tensor)

    def _validate_prior(
        self,
        prior: Optional[torch.Tensor],
        nodes: int,
        device: torch.device,
        dtype: torch.dtype,
    ) -> Optional[torch.Tensor]:
        if prior is None:
            return None
        if prior.ndim != 2 or prior.shape[0] != prior.shape[1]:
            raise ValueError("prior must be a square 2D tensor.")
        if prior.shape[0] != nodes:
            raise ValueError(f"prior size {prior.shape[0]} does not match node count {nodes}.")
        return prior.to(device=device, dtype=dtype)

    def _normalise_mask(
        self,
        mask: Optional[torch.Tensor],
        nodes: int,
        device: torch.device,
    ) -> Optional[torch.Tensor]:
        if mask is None:
            return None
        if mask.ndim == 2:
            if mask.shape[1] != 1:
                raise ValueError("mask must be 1D or (N, 1).")
            mask = mask.squeeze(-1)
        if mask.ndim != 1 or mask.shape[0] != nodes:
            raise ValueError(f"mask must be length {nodes}.")
        return mask.to(device=device, dtype=torch.bool)

    def _broadcast_mask(self, mask: torch.Tensor, tensor: torch.Tensor) -> torch.Tensor:
        if self.axis == "protein":
            return mask.view(-1, 1, 1).to(dtype=tensor.dtype)
        return mask.view(1, -1, 1).to(dtype=tensor.dtype)

    def _mask_attention(self, attn: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        if mask is None or mask.all():
            return attn
        valid = mask.to(dtype=torch.bool)
        eye = torch.eye(attn.size(0), device=attn.device, dtype=attn.dtype)
        attn = attn * valid.unsqueeze(1)
        attn = attn + (~valid).unsqueeze(1) * eye
        attn = attn / attn.sum(dim=1, keepdim=True).clamp_min(1e-12)
        return attn
