"""Structural GCN block with weighted adjacency and key/value projections."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


@dataclass
class KeyValueOutput:
    """Container for GCN outputs and head-specific key/value tensors."""

    features: torch.Tensor
    keys: torch.Tensor
    values: torch.Tensor


class StructuralGCNBlock(nn.Module):
    """Two-pass residual GCN over residue graphs with key/value projections.

    Args:
        channels: Input/output feature size (c).
        heads: Number of attention heads (h).
        head_dim: Per-head dimension (d).
        dropout: Dropout rate for GCN passes.
    """

    def __init__(
        self,
        channels: int,
        heads: int,
        head_dim: int,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        self.channels = int(channels)
        self.heads = int(heads)
        self.head_dim = int(head_dim)
        if channels != heads * head_dim:
            raise ValueError("channels must equal heads * head_dim.")
        self.norms = nn.ModuleList([nn.LayerNorm(channels) for _ in range(2)])
        self.linears = nn.ModuleList([nn.Linear(channels, channels, bias=False) for _ in range(2)])
        for linear in self.linears:
            nn.init.xavier_uniform_(linear.weight)
        self.dropouts = nn.ModuleList([nn.Dropout(dropout) for _ in range(2)])
        self.kv_norm = nn.LayerNorm(channels)
        self.key_proj = nn.Linear(channels, heads * head_dim, bias=False)
        self.value_proj = nn.Linear(channels, heads * head_dim, bias=False)
        nn.init.xavier_uniform_(self.key_proj.weight)
        nn.init.xavier_uniform_(self.value_proj.weight)

    def forward(
        self,
        features: torch.Tensor,
        adjacency: torch.Tensor | Mapping[str, Any],
        mask: Optional[torch.Tensor] = None,
    ) -> KeyValueOutput:
        """Run the two-pass GCN and return projected key/value tensors.

        Args:
            features: Tensor with shape (batch, length, channels).
            adjacency: Dense tensor with shape (batch, length, length) or sparse graph
                dict with edge_index/edge_weight/node_splits. Will be symmetrically
                normalized with added self-loops.
            mask: Optional boolean tensor with shape (batch, length).

        Returns:
            KeyValueOutput containing:
            - features: Tensor (batch, length, channels).
            - keys: Tensor (batch, heads, length, head_dim).
            - values: Tensor (batch, heads, length, head_dim).
        """
        if features.ndim != 3:
            raise ValueError("features must be 3D (batch, length, channels).")
        if isinstance(adjacency, Mapping):
            return self._forward_sparse(features, adjacency, mask)
        if adjacency.ndim != 3:
            raise ValueError("adjacency must be 3D (batch, length, length).")
        if mask is not None and mask.shape != features.shape[:2]:
            raise ValueError("mask must match (batch, length).")

        adj_fp32 = adjacency.float()
        length = adj_fp32.size(1)
        eye = torch.eye(length, device=adj_fp32.device, dtype=adj_fp32.dtype).unsqueeze(0)
        adj_fp32 = adj_fp32 + eye
        if mask is not None:
            mask_f = mask.to(dtype=adj_fp32.dtype)
            adj_fp32 = adj_fp32 * mask_f.unsqueeze(1) * mask_f.unsqueeze(2)

        deg = adj_fp32.sum(dim=-1)
        deg_inv_sqrt = torch.pow(deg.clamp(min=1e-6), -0.5)
        norm_adj = adj_fp32 * deg_inv_sqrt.unsqueeze(-1) * deg_inv_sqrt.unsqueeze(-2)
        if mask is not None:
            norm_adj = norm_adj * mask_f.unsqueeze(1) * mask_f.unsqueeze(2)

        out = features
        for norm, linear, drop in zip(self.norms, self.linears, self.dropouts):
            residual = out
            normed = norm(out)
            aggregated = torch.matmul(norm_adj, normed.float())
            updated = linear(aggregated)
            updated = F.relu(updated)
            updated = drop(updated)
            out = residual + updated.to(residual.dtype)
            if mask is not None:
                out = out * mask.unsqueeze(-1).to(out.dtype)

        normed = self.kv_norm(out)
        keys = self.key_proj(normed).view(out.size(0), out.size(1), self.heads, self.head_dim)
        values = self.value_proj(normed).view(out.size(0), out.size(1), self.heads, self.head_dim)
        keys = keys.permute(0, 2, 1, 3)
        values = values.permute(0, 2, 1, 3)
        return KeyValueOutput(features=out, keys=keys, values=values)

    def _forward_sparse(
        self,
        features: torch.Tensor,
        graph: Mapping[str, Any],
        mask: Optional[torch.Tensor],
    ) -> KeyValueOutput:
        if features.ndim != 3:
            raise ValueError("features must be 3D (batch, length, channels).")
        if mask is not None and mask.shape != features.shape[:2]:
            raise ValueError("mask must match (batch, length).")

        batch, max_len, channels = features.shape
        edge_index = torch.as_tensor(
            graph["edge_index"], dtype=torch.long, device=features.device
        )
        edge_weight = torch.as_tensor(
            graph["edge_weight"], dtype=torch.float32, device=features.device
        )
        node_splits = torch.as_tensor(
            graph["node_splits"], dtype=torch.long, device=features.device
        )
        if edge_index.ndim != 2 or edge_index.size(0) != 2:
            raise ValueError("edge_index must have shape (2, edges).")
        if edge_weight.ndim != 1 or edge_weight.numel() != edge_index.size(1):
            raise ValueError("edge_weight must have shape (edges,).")
        if node_splits.ndim != 1 or node_splits.numel() != batch + 1:
            raise ValueError("node_splits must have shape (batch + 1,).")
        lengths = node_splits[1:] - node_splits[:-1]
        total_nodes = int(node_splits[-1].item())
        if total_nodes == 0:
            empty = features.new_zeros((batch, max_len, channels))
            normed = self.kv_norm(empty)
            keys = self.key_proj(normed).view(batch, max_len, self.heads, self.head_dim)
            values = self.value_proj(normed).view(batch, max_len, self.heads, self.head_dim)
            keys = keys.permute(0, 2, 1, 3)
            values = values.permute(0, 2, 1, 3)
            return KeyValueOutput(features=empty, keys=keys, values=values)

        edge_index, edge_weight = self._add_self_loops(edge_index, edge_weight, total_nodes)
        row = edge_index[0]
        col = edge_index[1]
        deg = torch.zeros((total_nodes,), device=features.device, dtype=torch.float32)
        deg.index_add_(0, row, edge_weight)
        deg_inv_sqrt = torch.pow(deg.clamp(min=1e-6), -0.5)
        norm_weight = edge_weight * deg_inv_sqrt[row] * deg_inv_sqrt[col]

        segments = []
        for idx, length in enumerate(lengths.tolist()):
            if length <= 0:
                segments.append(features.new_zeros((0, channels)))
                continue
            segments.append(features[idx, :length])
        flat = torch.cat(segments, dim=0) if segments else features.new_zeros((0, channels))

        out = flat
        for norm, linear, drop in zip(self.norms, self.linears, self.dropouts):
            residual = out
            normed = norm(out)
            aggregated = self._sparse_aggregate(
                row, col, norm_weight, normed.float(), total_nodes
            )
            updated = linear(aggregated)
            updated = F.relu(updated)
            updated = drop(updated)
            out = residual + updated.to(residual.dtype)

        out_padded = features.new_zeros((batch, max_len, channels))
        cursor = 0
        for idx, length in enumerate(lengths.tolist()):
            if length > 0:
                out_padded[idx, :length] = out[cursor : cursor + length]
                cursor += length
        if mask is not None:
            out_padded = out_padded * mask.unsqueeze(-1).to(out_padded.dtype)

        normed = self.kv_norm(out_padded)
        keys = self.key_proj(normed).view(batch, max_len, self.heads, self.head_dim)
        values = self.value_proj(normed).view(batch, max_len, self.heads, self.head_dim)
        keys = keys.permute(0, 2, 1, 3)
        values = values.permute(0, 2, 1, 3)
        return KeyValueOutput(features=out_padded, keys=keys, values=values)

    @staticmethod
    def _add_self_loops(
        edge_index: torch.Tensor,
        edge_weight: torch.Tensor,
        num_nodes: int,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        self_mask = edge_index[0] == edge_index[1]
        if self_mask.any():
            edge_weight = edge_weight.clone()
            edge_weight[self_mask] = edge_weight[self_mask] + 1.0
            return edge_index, edge_weight
        loops = torch.arange(num_nodes, device=edge_index.device, dtype=torch.long)
        loop_index = torch.stack([loops, loops], dim=0)
        loop_weight = torch.ones((num_nodes,), device=edge_weight.device, dtype=edge_weight.dtype)
        edge_index = torch.cat([edge_index, loop_index], dim=1)
        edge_weight = torch.cat([edge_weight, loop_weight], dim=0)
        return edge_index, edge_weight

    @staticmethod
    def _sparse_aggregate(
        row: torch.Tensor,
        col: torch.Tensor,
        weight: torch.Tensor,
        features: torch.Tensor,
        num_nodes: int,
    ) -> torch.Tensor:
        messages = features[col] * weight.unsqueeze(-1)
        out = torch.zeros(
            (num_nodes, features.size(1)),
            device=features.device,
            dtype=features.dtype,
        )
        out.index_add_(0, row, messages)
        return out
