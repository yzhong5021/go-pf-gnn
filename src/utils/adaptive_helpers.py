"""
adaptive_helpers.py

Helper functions for adaptive modules.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

from typing import Iterable, Tuple, Union

def _init_attention_layers(*layers: nn.Linear) -> None:
    """Initialize bilinear attention layers with Xavier uniform weights."""
    for layer in layers:
        nn.init.xavier_uniform_(layer.weight)


def _init_linear_stack(stack: Iterable[nn.Linear]) -> None:
    """Apply Xavier uniform initialization to every linear layer in a stack."""
    for lin in stack:
        nn.init.xavier_uniform_(lin.weight)


def _build_top_p_attention(
    W1: nn.Linear,
    W2: nn.Linear,
    W3: nn.Linear,
    X: torch.Tensor,
    tau: float,
    p: float,
) -> torch.Tensor:
    """Construct a row-stochastic attention matrix with fixed p-mass sparsity."""
    Q, K = W1(X), W3(X)
    logits = (Q @ W2.weight @ K.T) / tau

    vals, idx = torch.sort(logits, dim=1, descending=True)
    probs_sorted = F.softmax(vals, dim=1)
    csum = torch.cumsum(probs_sorted.detach().cpu(), dim=1)
    k_i = (csum < p).sum(dim=1).to(device=logits.device) + 1

    N = logits.size(1)
    ranks = torch.arange(N, device=logits.device).unsqueeze(0).expand_as(vals)
    keep_sorted = ranks < k_i.unsqueeze(1)

    neg_inf = vals.new_full(vals.shape, float("-inf"))
    masked_sorted = torch.where(keep_sorted, vals, neg_inf)
    masked = logits.new_full(logits.shape, float("-inf"))
    masked.scatter_(1, idx, masked_sorted)

    return F.softmax(masked, dim=1)


def _prepare_prior(
    S: Union[torch.Tensor, None],
    N: int,
    device: torch.device,
    dtype: torch.dtype,
    *,
    bidirectional: bool,
) -> Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
    """Normalize optional prior adjacency and add self-loops."""
    if S is None:
        S = torch.eye(N, device=device, dtype=dtype)
    else:
        S = S.to(device=device, dtype=dtype)

    eps = 1e-6

    I = torch.eye(S.size(0), device=S.device, dtype=dtype)
    S = S + I

    #hande NaN blowup
    S = torch.nan_to_num(S, nan=0.0, posinf=0.0, neginf=0.0)
    S = S.clamp(min = eps)

    rowsum = S.sum(dim=1, keepdim=True).clamp_min(eps)
    Rf = S / rowsum

    if not bidirectional:
        return Rf

    colsum = S.sum(dim=0, keepdim=True).clamp_min(eps)
    Rb = (S.T / colsum).T
    return Rf, Rb
