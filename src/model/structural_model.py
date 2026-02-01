"""Updated PF-AGCN model with ESMFold structural priors and ProstT5 queries."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping, Optional, Tuple

import torch
import torch.nn as nn

from src.modules.attention import SingleQueryCrossAttention
from src.modules.prediction_head import PredictionHead
from src.modules.prost_graph import ProstGraphBlock
from src.modules.prostt5_3di import ProstT5QueryEncoder
from src.modules.sqb import SQBBlock
from src.modules.structural_gcn import StructuralGCNBlock


def _get_cfg_value(cfg: Mapping | object, key: str, default):
    if isinstance(cfg, Mapping):
        return cfg.get(key, default)
    return getattr(cfg, key, default)


@dataclass
class StructuralOutput:
    """Container returned by the updated model forward pass."""

    logits: torch.Tensor


class StructuralPFAGCN(nn.Module):
    """Updated PF-AGCN model following the structural + ProstT5 cross-attention pipeline."""

    def __init__(self, cfg: object) -> None:
        super().__init__()
        model_cfg = _get_cfg_value(cfg, "model", {})
        task_cfg = _get_cfg_value(cfg, "task", {})
        self.num_functions = int(_get_cfg_value(task_cfg, "num_functions", 0))

        seq_cfg = _get_cfg_value(model_cfg, "seq_embeddings", {})
        raw_dim = int(_get_cfg_value(seq_cfg, "raw_dim", _get_cfg_value(seq_cfg, "feature_dim", 1280)))

        sqb_cfg = _get_cfg_value(model_cfg, "sqb", {})
        channels = int(_get_cfg_value(sqb_cfg, "channels", 256))
        dccn_cfg = _get_cfg_value(sqb_cfg, "dccn", {})
        kernel_size = int(_get_cfg_value(dccn_cfg, "kernel_size", 3))
        dilation = int(_get_cfg_value(dccn_cfg, "dilation", 2))
        dccn_dropout = float(_get_cfg_value(dccn_cfg, "dropout", 0.1))

        attn_cfg = _get_cfg_value(model_cfg, "cross_attention", {})
        heads = int(_get_cfg_value(attn_cfg, "heads", 4))
        if channels % heads != 0:
            raise ValueError("channels must be divisible by cross_attention.heads.")
        head_dim = channels // heads
        attn_dropout = float(_get_cfg_value(attn_cfg, "dropout", 0.1))

        prost_attention_cfg = _get_cfg_value(model_cfg, "prost_attention", {})
        self.prost_attention_enabled = bool(
            _get_cfg_value(prost_attention_cfg, "enabled", True)
        )
        prost_graph_cfg = _get_cfg_value(model_cfg, "prost_graph", {})
        self.prost_graph_enabled = bool(
            _get_cfg_value(prost_graph_cfg, "enabled", False)
        )

        gcn_cfg = _get_cfg_value(model_cfg, "gcn", {})
        gcn_dropout = float(_get_cfg_value(gcn_cfg, "dropout", 0.1))

        head_cfg = _get_cfg_value(model_cfg, "prediction_head", {})
        mlp_hidden = _get_cfg_value(head_cfg, "mlp_hidden_dim", None)

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
        self.prost_graph = None
        self.prost_query = None
        self.cross_attention = None
        if self.prost_attention_enabled:
            prost_cfg = _get_cfg_value(model_cfg, "prostt5_3di", {})
            input_dim = int(_get_cfg_value(prost_cfg, "encoder_dim", 1024))
            if self.prost_graph_enabled:
                prost_graph_dropout = float(
                    _get_cfg_value(prost_graph_cfg, "dropout", gcn_dropout)
                )
                self.prost_graph = ProstGraphBlock(
                    input_dim=input_dim,
                    dropout=prost_graph_dropout,
                )
            self.prost_query = ProstT5QueryEncoder(
                heads=heads,
                head_dim=head_dim,
                input_dim=input_dim,
            )
            self.cross_attention = SingleQueryCrossAttention(
                channels=channels,
                heads=heads,
                head_dim=head_dim,
                dropout=attn_dropout,
            )
        self.head = PredictionHead(
            channels=channels,
            num_functions=self.num_functions,
            mlp_hidden_dim=mlp_hidden,
        )

    def forward(
        self,
        seq_embeddings: torch.Tensor,
        structure_graph: torch.Tensor | Mapping[str, torch.Tensor],
        prostt5_probs: Optional[torch.Tensor] = None,
        lengths: Optional[torch.Tensor] = None,
        mask: Optional[torch.Tensor] = None,
    ) -> StructuralOutput:
        """Run the updated structural PF-AGCN forward pass.

        Args:
            seq_embeddings: Tensor with shape (batch, length, raw_dim).
            structure_graph: Dense tensor with shape (batch, length, length) or
                sparse graph dict with edge_index/edge_weight/node_splits.
            prostt5_probs: Tensor with shape (batch, length, prost_dim) when enabled.
            lengths: Optional tensor with shape (batch,) giving residue counts.
            mask: Optional boolean tensor with shape (batch, length).

        Returns:
            StructuralOutput containing logits with shape (batch, num_functions).
        """
        if seq_embeddings.ndim != 3:
            raise ValueError("seq_embeddings must be (batch, length, dim).")
        if isinstance(structure_graph, Mapping):
            if "node_splits" not in structure_graph:
                raise ValueError("structure_graph sparse dict must include node_splits.")
        else:
            if structure_graph.ndim != 3:
                raise ValueError("structure_graph must be (batch, length, length).")
        if self.prost_attention_enabled:
            if prostt5_probs is None:
                raise ValueError("prostt5_probs is required when prost_attention is enabled.")
            if prostt5_probs.ndim != 3:
                raise ValueError("prostt5_probs must be (batch, length, prost_dim).")

        batch, length, _ = seq_embeddings.shape
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
        if self.prost_attention_enabled:
            if prostt5_probs.shape[:2] != (batch, length):
                raise ValueError("prostt5_probs must align with seq_embeddings length.")
            if self.prost_query is not None:
                expected_dim = self.prost_query.input_dim
                if prostt5_probs.size(-1) != expected_dim:
                    raise ValueError(
                        "prostt5_probs last dimension must match prost_query input_dim."
                    )

        sqb_features = self.sqb(seq_embeddings, mask_bool)
        gcn_out = self.structural_gcn(sqb_features, structure_graph, mask_bool)

        if self.prost_attention_enabled:
            prost_embeddings = prostt5_probs
            if self.prost_graph is not None:
                prost_embeddings = self.prost_graph(
                    prost_embeddings, structure_graph, mask_bool
                )
            queries, _ = self.prost_query(prost_embeddings, mask_bool)
            attn_out, _ = self.cross_attention(
                gcn_out.keys,
                gcn_out.values,
                queries,
                mask_bool,
                residual=gcn_out.features,
            )
        else:
            attn_out = gcn_out.features
        logits = self.head(attn_out, gcn_out.features, mask_bool)
        return StructuralOutput(logits=logits)

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
