"""ProstT5 graph diffusion block for contact-guided query embeddings."""

from __future__ import annotations

from typing import Any, Mapping, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


class ProstGraphBlock(nn.Module):
    """Single-pass GCN diffusion over ProstT5 encoder embeddings.

    Args:
        input_dim: ProstT5 embedding dimension (d).
        dropout: Dropout rate applied after the diffusion linear layer.
    """

    def __init__(self, input_dim: int = 1024, dropout: float = 0.1) -> None:
        super().__init__()
        self.input_dim = int(input_dim)
        self.input_proj = nn.Linear(self.input_dim, self.input_dim, bias=False)
        self.update_proj = nn.Linear(self.input_dim, self.input_dim, bias=False)
        nn.init.xavier_uniform_(self.input_proj.weight)
        nn.init.xavier_uniform_(self.update_proj.weight)
        self.dropout = nn.Dropout(dropout)
        self.pre_norm = nn.LayerNorm(self.input_dim)

    def forward(
        self,
        embeddings: torch.Tensor,
        adjacency: torch.Tensor | Mapping[str, Any],
        mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Diffuse ProstT5 embeddings with a residue contact graph.

        Args:
            embeddings: Tensor with shape (batch, length, input_dim).
            adjacency: Dense tensor (batch, length, length) or sparse graph dict
                with edge_index/edge_weight/plddt/node_splits.
            mask: Optional boolean tensor with shape (batch, length).

        Returns:
            Tensor with shape (batch, length, input_dim) after diffusion.
        """
        if embeddings.ndim != 3 or embeddings.size(-1) != self.input_dim:
            raise ValueError("embeddings must be (batch, length, input_dim).")
        if mask is not None and mask.shape != embeddings.shape[:2]:
            raise ValueError("mask must match (batch, length).")

        projected = self.input_proj(embeddings)
        if isinstance(adjacency, Mapping):
            return self._forward_sparse(projected, adjacency, mask)
        if adjacency.ndim != 3:
            raise ValueError("adjacency must be 3D (batch, length, length).")
        return self._forward_dense(projected, adjacency, mask)

    def _forward_dense(
        self,
        features: torch.Tensor,
        adjacency: torch.Tensor,
        mask: Optional[torch.Tensor],
    ) -> torch.Tensor:
        adj_fp32 = adjacency.float()
        length = adj_fp32.size(1)
        if adj_fp32.size(2) != length:
            raise ValueError("adjacency must be square (batch, length, length).")

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

        normed = self.pre_norm(features)
        aggregated = torch.matmul(norm_adj, normed.float())
        updated = self.update_proj(aggregated)
        updated = F.relu(updated)
        updated = self.dropout(updated)
        out = features + updated.to(features.dtype)
        if mask is not None:
            out = out * mask.unsqueeze(-1).to(out.dtype)
        return out

    def _forward_sparse(
        self,
        features: torch.Tensor,
        graph: Mapping[str, Any],
        mask: Optional[torch.Tensor],
    ) -> torch.Tensor:
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
        plddt = torch.as_tensor(graph["plddt"], dtype=torch.float32, device=features.device)
        if edge_index.ndim != 2 or edge_index.size(0) != 2:
            raise ValueError("edge_index must have shape (2, edges).")
        if edge_weight.ndim != 1 or edge_weight.numel() != edge_index.size(1):
            raise ValueError("edge_weight must have shape (edges,).")
        if node_splits.ndim != 1 or node_splits.numel() != batch + 1:
            raise ValueError("node_splits must have shape (batch + 1,).")

        lengths = node_splits[1:] - node_splits[:-1]
        total_nodes = int(node_splits[-1].item())
        if total_nodes == 0:
            return features.new_zeros((batch, max_len, channels))
        if plddt.numel() != total_nodes:
            raise ValueError("plddt must have shape (total_nodes,).")

        edge_index, edge_weight = self._add_self_loops_with_plddt(
            edge_index, edge_weight, plddt, total_nodes
        )
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

        normed = self.pre_norm(flat)
        aggregated = self._sparse_aggregate(row, col, norm_weight, normed.float(), total_nodes)
        updated = self.update_proj(aggregated)
        updated = F.relu(updated)
        updated = self.dropout(updated)
        out = flat + updated.to(flat.dtype)

        out_padded = features.new_zeros((batch, max_len, channels))
        cursor = 0
        for idx, length in enumerate(lengths.tolist()):
            if length > 0:
                out_padded[idx, :length] = out[cursor : cursor + length]
                cursor += length
        if mask is not None:
            out_padded = out_padded * mask.unsqueeze(-1).to(out_padded.dtype)
        return out_padded

    @staticmethod
    def _add_self_loops_with_plddt(
        edge_index: torch.Tensor,
        edge_weight: torch.Tensor,
        plddt: torch.Tensor,
        num_nodes: int,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        self_mask = edge_index[0] == edge_index[1]
        if self_mask.any():
            edge_index = edge_index[:, ~self_mask]
            edge_weight = edge_weight[~self_mask]
        loops = torch.arange(num_nodes, device=edge_index.device, dtype=torch.long)
        loop_index = torch.stack([loops, loops], dim=0)
        loop_weight = plddt.to(dtype=edge_weight.dtype)
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
