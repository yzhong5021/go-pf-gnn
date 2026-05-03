"""ProstT5-only lightweight classifier for PF-AGCN.

Uses cached ProstT5 encoder embeddings with masked mean pooling followed by a
3-layer MLP (d -> d/2 -> d/4 -> num_functions).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping, Optional, Tuple

import torch
import torch.nn as nn

from src.modules.prost_graph import ProstGraphBlock


def _get_cfg_value(cfg: Mapping | object, key: str, default):
    if isinstance(cfg, Mapping):
        return cfg.get(key, default)
    return getattr(cfg, key, default)


@dataclass
class ProstOnlyOutput:
    """Container returned by the Prost-only forward pass."""

    logits: torch.Tensor


class ProstOnlyPFAGCN(nn.Module):
    """ProstT5-only classifier with optional graph diffusion and a 3-layer MLP."""

    def __init__(self, cfg: object) -> None:
        super().__init__()
        model_cfg = _get_cfg_value(cfg, "model", {})
        task_cfg = _get_cfg_value(cfg, "task", {})
        self.num_functions = int(_get_cfg_value(task_cfg, "num_functions", 0))

        prost_cfg = _get_cfg_value(model_cfg, "prostt5_3di", {})
        input_dim = int(_get_cfg_value(prost_cfg, "encoder_dim", 1024))
        if input_dim < 4 or input_dim % 4 != 0:
            raise ValueError("prostt5_3di.encoder_dim must be divisible by 4.")
        self.input_dim = input_dim
        prost_graph_cfg = _get_cfg_value(model_cfg, "prost_graph", {})
        self.prost_graph_enabled = bool(_get_cfg_value(prost_graph_cfg, "enabled", False))
        self.prost_graph = None
        if self.prost_graph_enabled:
            dropout = float(_get_cfg_value(prost_graph_cfg, "dropout", 0.1))
            self.prost_graph = ProstGraphBlock(input_dim=self.input_dim, dropout=dropout)
        hidden_dim = input_dim // 2
        mid_dim = input_dim // 4

        self.mlp = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, mid_dim),
            nn.ReLU(),
            nn.Linear(mid_dim, self.num_functions),
        )
        for layer in self.mlp:
            if isinstance(layer, nn.Linear):
                nn.init.xavier_uniform_(layer.weight)
                if layer.bias is not None:
                    nn.init.zeros_(layer.bias)

    def forward(
        self,
        prostt5_probs: torch.Tensor,
        structure_graph: Optional[torch.Tensor | Mapping[str, torch.Tensor]] = None,
        lengths: Optional[torch.Tensor] = None,
        mask: Optional[torch.Tensor] = None,
    ) -> ProstOnlyOutput:
        """Compute logits from ProstT5 embeddings.

        Args:
            prostt5_probs: Tensor with shape (batch, length, input_dim).
            structure_graph: Optional dense or sparse residue graph for prost diffusion.
            lengths: Optional tensor with shape (batch,) giving residue counts.
            mask: Optional boolean tensor with shape (batch, length).

        Returns:
            ProstOnlyOutput with logits shaped (batch, num_functions).
        """
        if prostt5_probs.ndim != 3:
            raise ValueError("prostt5_probs must be (batch, length, dim).")
        batch, length, dim = prostt5_probs.shape
        if dim != self.input_dim:
            raise ValueError("prostt5_probs last dimension must match encoder_dim.")
        mask_bool, _ = self._normalise_mask(
            lengths, mask, batch, length, prostt5_probs.device
        )
        features = prostt5_probs
        if self.prost_graph is not None:
            if structure_graph is None:
                raise ValueError("structure_graph is required when prost_graph is enabled.")
            features = self.prost_graph(features, structure_graph, mask_bool)
        pooled = self._masked_mean(features, mask_bool)
        logits = self.mlp(pooled)
        return ProstOnlyOutput(logits=logits)

    @staticmethod
    def _masked_mean(features: torch.Tensor, mask: Optional[torch.Tensor]) -> torch.Tensor:
        if mask is None:
            return features.mean(dim=1)
        mask_f = mask.to(dtype=features.dtype).unsqueeze(-1)
        summed = torch.sum(features * mask_f, dim=1)
        denom = mask_f.sum(dim=1).clamp(min=1.0)
        return summed / denom

    @staticmethod
    def _normalise_mask(
        lengths: Optional[torch.Tensor],
        mask: Optional[torch.Tensor],
        batch: int,
        length: int,
        device: torch.device,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        if mask is not None:
            if mask.shape != (batch, length):
                raise ValueError("mask must match the batch and sequence dimensions.")
            mask_bool = mask.to(device=device, dtype=torch.bool)
            lengths_tensor = mask_bool.sum(dim=1, dtype=torch.long)
            if lengths is not None:
                lengths = lengths.to(device=device, dtype=torch.long)
                if not torch.equal(lengths_tensor, lengths):
                    raise ValueError("mask and lengths encode conflicting values.")
        else:
            if lengths is None or lengths.ndim != 1:
                raise ValueError("lengths must be a 1D tensor when mask is absent.")
            lengths_tensor = lengths.to(device=device, dtype=torch.long)
            mask_bool = (
                torch.arange(length, device=device)[None, :]
                < lengths_tensor[:, None]
            )
        return mask_bool, lengths_tensor
