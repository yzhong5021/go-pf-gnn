"""Gated ProstT5+ESM2 pooling model for PF-AGCN.

Combines ProstT5 and ESM2 post-graph residue embeddings via attention/mean pooling,
per-stream gating, and a 3-layer MLP classifier.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping, Optional, Tuple

import torch
import torch.nn as nn

from src.modules.attention import LearnedAttentionPooling
from src.modules.prost_graph import ProstGraphBlock
from src.modules.sqb import SQBBlock
from src.modules.structural_gcn import StructuralGCNBlock


def _get_cfg_value(cfg: Mapping | object, key: str, default):
    if isinstance(cfg, Mapping):
        return cfg.get(key, default)
    return getattr(cfg, key, default)


@dataclass
class GatedPEOutput:
    """Container returned by the gated pooling encoder forward pass."""

    logits: torch.Tensor


class GatedPEPFAGCN(nn.Module):
    """Gated ProstT5+ESM2 pooling model with 3-layer MLP classifier."""

    def __init__(self, cfg: object) -> None:
        super().__init__()
        model_cfg = _get_cfg_value(cfg, "model", {})
        task_cfg = _get_cfg_value(cfg, "task", {})
        self.num_functions = int(_get_cfg_value(task_cfg, "num_functions", 0))

        seq_cfg = _get_cfg_value(model_cfg, "seq_embeddings", {})
        raw_dim = int(
            _get_cfg_value(seq_cfg, "raw_dim", _get_cfg_value(seq_cfg, "feature_dim", 1280))
        )

        sqb_cfg = _get_cfg_value(model_cfg, "sqb", {})
        channels = int(_get_cfg_value(sqb_cfg, "channels", 512))
        dccn_cfg = _get_cfg_value(sqb_cfg, "dccn", {})
        kernel_size = int(_get_cfg_value(dccn_cfg, "kernel_size", 3))
        dilation = int(_get_cfg_value(dccn_cfg, "dilation", 2))
        dccn_dropout = float(_get_cfg_value(dccn_cfg, "dropout", 0.1))

        gcn_cfg = _get_cfg_value(model_cfg, "gcn", {})
        gcn_dropout = float(_get_cfg_value(gcn_cfg, "dropout", 0.1))
        heads = int(_get_cfg_value(gcn_cfg, "heads", 8))
        if channels % heads != 0:
            raise ValueError("gcn.heads must divide sqb.channels for gated_pe.")
        head_dim = channels // heads

        prost_cfg = _get_cfg_value(model_cfg, "prostt5_3di", {})
        prost_dim = int(_get_cfg_value(prost_cfg, "encoder_dim", 1024))
        if prost_dim <= 0:
            raise ValueError("prostt5_3di.encoder_dim must be positive.")

        prost_graph_cfg = _get_cfg_value(model_cfg, "prost_graph", {})
        prost_graph_enabled = bool(_get_cfg_value(prost_graph_cfg, "enabled", True))
        prost_graph_dropout = float(_get_cfg_value(prost_graph_cfg, "dropout", 0.1))

        gated_cfg = _get_cfg_value(model_cfg, "gated_pe", {})
        mlp_dropout = float(_get_cfg_value(gated_cfg, "mlp_dropout", 0.1))

        self.input_dim = prost_dim
        self.channels = channels
        self.prost_graph_enabled = prost_graph_enabled

        self.sqb = SQBBlock(
            input_dim=raw_dim,
            channels=channels,
            kernel_size=kernel_size,
            dilation=dilation,
            dropout=dccn_dropout,
        )
        self.structural_gcn = StructuralGCNBlock(
            channels=channels,
            heads=heads,
            head_dim=head_dim,
            dropout=gcn_dropout,
        )
        self.prost_graph = ProstGraphBlock(input_dim=prost_dim, dropout=prost_graph_dropout)

        self.prost_norm = nn.LayerNorm(prost_dim)
        self.esm_norm = nn.LayerNorm(channels)

        self.prost_residue_proj = nn.Linear(prost_dim, channels)
        self.esm_residue_proj = nn.Linear(channels, channels)
        self.prost_attn_pool = LearnedAttentionPooling(channels)
        self.esm_attn_pool = LearnedAttentionPooling(channels)

        self.stream_gate = nn.Parameter(torch.zeros(4))
        self.mlp = nn.Sequential(
            nn.Linear(channels, channels // 2),
            nn.ReLU(),
            nn.Dropout(mlp_dropout),
            nn.Linear(channels // 2, channels // 2),
            nn.ReLU(),
            nn.Dropout(mlp_dropout),
            nn.Linear(channels // 2, self.num_functions),
        )

        for layer in (
            self.prost_residue_proj,
            self.esm_residue_proj,
        ):
            nn.init.xavier_uniform_(layer.weight)
            if layer.bias is not None:
                nn.init.zeros_(layer.bias)
        nn.init.zeros_(self.stream_gate)
        for layer in self.mlp:
            if isinstance(layer, nn.Linear):
                nn.init.xavier_uniform_(layer.weight)
                if layer.bias is not None:
                    nn.init.zeros_(layer.bias)

    def forward(
        self,
        seq_embeddings: torch.Tensor,
        structure_graph: torch.Tensor | Mapping[str, torch.Tensor],
        prostt5_probs: torch.Tensor,
        lengths: Optional[torch.Tensor] = None,
        mask: Optional[torch.Tensor] = None,
    ) -> GatedPEOutput:
        """Compute logits from gated ProstT5+ESM2 pooled features.

        Args:
            seq_embeddings: Tensor with shape (batch, length, raw_dim).
            structure_graph: Dense tensor with shape (batch, length, length) or
                sparse graph dict with edge_index/edge_weight/node_splits.
            prostt5_probs: Tensor with shape (batch, length, input_dim).
            lengths: Optional tensor with shape (batch,) giving residue counts.
            mask: Optional boolean tensor with shape (batch, length).

        Returns:
            GatedPEOutput with logits shaped (batch, num_functions).
        """
        if seq_embeddings.ndim != 3:
            raise ValueError("seq_embeddings must be (batch, length, dim).")
        if prostt5_probs.ndim != 3:
            raise ValueError("prostt5_probs must be (batch, length, dim).")
        if isinstance(structure_graph, Mapping):
            if "node_splits" not in structure_graph:
                raise ValueError("structure_graph sparse dict must include node_splits.")
        else:
            if structure_graph.ndim != 3:
                raise ValueError("structure_graph must be (batch, length, length).")

        if seq_embeddings.dtype != torch.float32:
            seq_embeddings = seq_embeddings.float()
        if prostt5_probs.dtype != torch.float32:
            prostt5_probs = prostt5_probs.float()

        batch, length, _ = seq_embeddings.shape
        if prostt5_probs.shape[:2] != (batch, length):
            raise ValueError("prostt5_probs must align with seq_embeddings length.")
        if prostt5_probs.size(-1) != self.input_dim:
            raise ValueError("prostt5_probs last dimension must match encoder_dim.")

        mask_bool, lengths_tensor = self._normalise_mask(
            lengths, mask, batch, length, seq_embeddings.device
        )

        if isinstance(structure_graph, Mapping):
            node_splits = structure_graph["node_splits"].to(device=seq_embeddings.device)
            if node_splits.ndim != 1 or node_splits.numel() != batch + 1:
                raise ValueError("node_splits must have shape (batch + 1,).")
            if int(node_splits[-1].item()) != int(lengths_tensor.sum().item()):
                raise ValueError("structure_graph node_splits must match batch lengths.")
        else:
            if structure_graph.shape[:2] != (batch, length):
                raise ValueError("structure_graph must align with seq_embeddings length.")

        sqb_features = self.sqb(seq_embeddings, mask_bool)
        gcn_out = self.structural_gcn(sqb_features, structure_graph, mask_bool)
        esm_features = self.esm_residue_proj(self.esm_norm(gcn_out.features))

        prost_embeddings = prostt5_probs
        if self.prost_graph_enabled:
            prost_embeddings = self.prost_graph(prost_embeddings, structure_graph, mask_bool)
        prost_features = self.prost_residue_proj(self.prost_norm(prost_embeddings))

        prost_attn, _ = self.prost_attn_pool(prost_features, mask_bool)
        prost_mean = self._masked_mean(prost_features, mask_bool)
        esm_attn, _ = self.esm_attn_pool(esm_features, mask_bool)
        esm_mean = self._masked_mean(esm_features, mask_bool)

        combined = torch.stack([prost_attn, prost_mean, esm_attn, esm_mean], dim=1)
        gates = torch.sigmoid(self.stream_gate).view(1, 4, 1)
        gated = torch.sum(combined * gates, dim=1)
        logits = self.mlp(gated)
        return GatedPEOutput(logits=logits)

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
