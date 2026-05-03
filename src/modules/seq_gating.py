"""
seq_gating.py

gating mechanism for esm + dccn embeddings (capture both local and global sequence context)
"""

from typing import Optional

import torch
import torch.nn as nn

class SeqGating(nn.Module):
    def __init__(self, d_shared = 128, d_esm=1280, c_dcc=24, attn_hidden=128, dropout=0.1):
        super().__init__()

        #project embeddings to d_shared
        self.proj_esm = nn.Linear(d_esm, d_shared, bias=False)
        self.proj_dcc = nn.Linear(c_dcc, d_shared, bias=False)

        # generate token-wise gating weights
        self.gate_tok = nn.Linear(2*d_shared, d_shared)

        # score tokens for attention-based pooling
        self.scorer   = nn.Sequential(nn.Linear(d_shared, attn_hidden), nn.ReLU(), nn.Linear(attn_hidden, 1))

        self.dropout  = nn.Dropout(dropout)
        self.ln       = nn.LayerNorm(d_shared)

        self._init_weights()

    def _init_weights(self) -> None:
        nn.init.xavier_uniform_(self.proj_esm.weight)
        nn.init.xavier_uniform_(self.proj_dcc.weight)

        nn.init.xavier_uniform_(self.gate_tok.weight)
        nn.init.zeros_(self.gate_tok.bias)

        scorer_in = self.scorer[0]
        nn.init.kaiming_uniform_(scorer_in.weight, nonlinearity="relu")
        nn.init.zeros_(scorer_in.bias)

        scorer_out = self.scorer[2]
        nn.init.xavier_uniform_(scorer_out.weight)
        nn.init.zeros_(scorer_out.bias)

    def forward(
        self,
        H_esm: torch.Tensor,
        H_dcc: torch.Tensor,
        lengths: Optional[torch.Tensor] = None,
        mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:  # (N_P,L,1280),(N_P,L,C_dcc)
        N_P, L, _ = H_esm.shape # N = batch size, L = max sequence length

        if mask is not None:
            if mask.shape != (N_P, L):
                raise ValueError("mask must be shaped (batch, length).")
            mask_bool = mask.to(device=H_esm.device, dtype=torch.bool)
            lengths = mask_bool.sum(dim=1, dtype=torch.long)
        else:
            if lengths is None:
                raise ValueError("Either lengths or mask must be provided.")
            lengths = lengths.to(device=H_esm.device, dtype=torch.long)
            mask_bool = (
                torch.arange(L, device=H_esm.device)[None, :] < lengths[:, None]
            )

        mask_expanded = mask_bool.unsqueeze(-1)

        # project esm and dcc embeddings to shared d_shared dimension
        E = self.dropout(self.proj_esm(H_esm))

        D = self.dropout(self.proj_dcc(H_dcc))   # both (N_P,L,d_shared)

        E = E.masked_fill(~mask_expanded, 0.0)
        D = D.masked_fill(~mask_expanded, 0.0)

        G = torch.sigmoid(self.gate_tok(torch.cat([E, D], dim=-1)))  # (N_P,L,d_shared) ; generate token-wise gating weights
        H = G*E + (1.0-G)*D                                          # (N_P,L,d_shared) ; fuse esm and dcc embeddings
        H = H.masked_fill(~mask_expanded, 0.0)

        scores = self.scorer(H)                                      # (N_P,L,1) ; score each token for attention-based pooling
        scores = scores.masked_fill(~mask_expanded, float('-inf'))
        w = torch.softmax(scores, dim=1)                             # normalize
        w = torch.nan_to_num(w, nan=0.0)
        x = (H * w).sum(dim=1)                                       # (N_P,d_shared) ; pooled embedding
        return self.ln(x)                                            # (N_P,d_shared) ; normalized via layernorm
