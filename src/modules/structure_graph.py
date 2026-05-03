"""Build sparse residue graphs from CA coordinates and pLDDT confidence."""

from __future__ import annotations

from typing import Tuple

import torch


class StructureGraphBuilder:
    """Utilities for building dense or sparse residue graphs from CA coords."""

    @staticmethod
    def _normalize_plddt(plddt: torch.Tensor) -> torch.Tensor:
        """Normalize pLDDT to [0, 1] and squeeze to (length,)."""
        plddt = torch.as_tensor(plddt, dtype=torch.float32)
        if plddt.dim() == 3:
            if plddt.size(-1) > 1:
                plddt = plddt.mean(dim=-1)
            else:
                plddt = plddt.squeeze(-1)
        if plddt.dim() == 2:
            if plddt.size(0) == 1:
                plddt = plddt.squeeze(0)
            elif plddt.size(1) == 1:
                plddt = plddt.squeeze(1)
        if plddt.dim() != 1:
            raise ValueError(f"Expected pLDDT to be 1D after squeeze, got {plddt.shape}.")
        plddt = torch.nan_to_num(plddt, nan=0.0, posinf=0.0, neginf=0.0)
        if plddt.numel() and plddt.max() > 1.0:
            plddt = plddt / 100.0
        return plddt.clamp(0.0, 1.0)

    @staticmethod
    def _prepare_coords(ca_coords: torch.Tensor) -> torch.Tensor:
        coords = torch.as_tensor(ca_coords, dtype=torch.float32)
        if coords.dim() != 2 or coords.size(-1) != 3:
            raise ValueError(f"Expected CA coords with shape (length, 3), got {coords.shape}.")
        return coords

    @staticmethod
    def _select_neighbor_mask(
        distances: torch.Tensor,
        *,
        distance_cutoff: float,
        top_k: int,
    ) -> torch.Tensor:
        if top_k <= 0:
            raise ValueError("top_k must be >= 1.")
        n = distances.size(0)
        k = min(int(top_k), n)
        mask = distances <= distance_cutoff
        dist_masked = distances.clone()
        dist_masked[~mask] = float("inf")
        vals, idx = torch.topk(dist_masked, k=k, dim=1, largest=False)
        sel = torch.zeros_like(mask)
        row_idx = torch.arange(n, device=distances.device).unsqueeze(1).expand(n, k)
        sel[row_idx, idx] = vals <= distance_cutoff
        sel = sel | sel.T
        sel.fill_diagonal_(True)
        return sel

    @classmethod
    def build_graph_from_ca(
        cls,
        *,
        ca_coords: torch.Tensor,
        plddt: torch.Tensor,
        distance_cutoff: float,
        top_k: int,
    ) -> torch.Tensor:
        """Return dense adjacency (length x length) with self-confidence on the diagonal."""
        coords = cls._prepare_coords(ca_coords)
        conf = cls._normalize_plddt(plddt).to(device=coords.device)
        if conf.numel() != coords.size(0):
            raise ValueError(
                f"pLDDT length {conf.numel()} does not match coords length {coords.size(0)}."
            )
        distances = torch.cdist(coords, coords)
        sel = cls._select_neighbor_mask(
            distances,
            distance_cutoff=distance_cutoff,
            top_k=top_k,
        )
        weight = torch.minimum(conf[:, None], conf[None, :])
        adj = torch.where(sel, weight, torch.zeros_like(weight))
        adj.diagonal().copy_(conf)
        return adj

    @classmethod
    def build_sparse_graph_from_ca(
        cls,
        *,
        ca_coords: torch.Tensor,
        plddt: torch.Tensor,
        distance_cutoff: float,
        top_k: int,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Return edge_index/edge_weight without self loops."""
        coords = cls._prepare_coords(ca_coords)
        conf = cls._normalize_plddt(plddt).to(device=coords.device)
        if conf.numel() != coords.size(0):
            raise ValueError(
                f"pLDDT length {conf.numel()} does not match coords length {coords.size(0)}."
            )
        distances = torch.cdist(coords, coords)
        sel = cls._select_neighbor_mask(
            distances,
            distance_cutoff=distance_cutoff,
            top_k=top_k,
        )
        eye = torch.eye(sel.size(0), dtype=torch.bool, device=sel.device)
        sel = sel & ~eye
        edge_index = sel.nonzero(as_tuple=False).T
        edge_weight = torch.minimum(conf[edge_index[0]], conf[edge_index[1]])
        return edge_index, edge_weight
