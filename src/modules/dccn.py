"""
dccn.py

1-D dilated causal convolutional network (WaveNet). Called after sequences are embedded by ESM-1b.
"""

from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F


class DCCN_1D(nn.Module):
    """
        args:

        - embed_len (int): number of channels for each token
        - k_size (int): kernel size
        - dilation (int): dilation factor, r; will create receptive fields of increasing size from 1+(k-1)*r for i in range 4
        - dropout (float): dropout rate
    """

    def __init__(self, embed_len = 24, k_size = 3, dilation = 2, dropout = 0.1):
        super().__init__()

        self.c = embed_len
        self.k = k_size
        r = dilation
        self.dilations = (1, r, r**2, r**3)

        self.convs = nn.ModuleList([
            nn.Conv1d(in_channels=self.c, out_channels=self.c, kernel_size=self.k, dilation=d, padding=0)
            for d in self.dilations
        ])

        self.norms = nn.ModuleList([nn.LayerNorm(self.c) for _ in self.dilations])
        self.dropouts = nn.ModuleList([nn.Dropout(dropout) for _ in self.dilations])

        self.gating = nn.Conv1d(in_channels = 4 * self.c, out_channels = 4 * self.c, kernel_size = 1) #dilation channelwise linear gating

        self._init_weights()

    def _init_weights(self) -> None:
        for conv in self.convs:
            nn.init.kaiming_normal_(conv.weight, mode="fan_out", nonlinearity="relu")
            if conv.bias is not None:
                nn.init.zeros_(conv.bias)
        nn.init.xavier_uniform_(self.gating.weight)
        if self.gating.bias is not None:
            nn.init.zeros_(self.gating.bias)

    def _causal_pad(self, x, dilation):
        # causal masking via left padding
        return F.pad(x, ((self.k-1)*dilation, 0))

    def forward(self, x: torch.Tensor, mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        # x = B, L, C where B = batch size, L = sequence length, C = embedding length
        xt = x.transpose(1,2) # set up for cnn

        mask_expanded = None
        if mask is not None:
            if mask.ndim != 2:
                raise ValueError("mask must be 2D (batch, length).")
            if mask.shape[0] != x.size(0) or mask.shape[1] != x.size(1):
                raise ValueError("mask must match the batch and sequence dimensions of x.")
            mask_expanded = mask.to(device=x.device, dtype=x.dtype).unsqueeze(1)
            xt = xt * mask_expanded

        qs = []

        y = xt.clone()
        y_prev = y

        for dil, conv, drop, norm in zip(self.dilations, self.convs, self.dropouts, self.norms):
            y_pad = self._causal_pad(y, dil)
            y = F.relu(conv(y_pad))
            y = y.transpose(1,2) # layernorm prep
            y= norm(y)
            y = drop(y)
            y = y.transpose(1,2)
            y = y + y_prev # add residual for stability
            if mask_expanded is not None:
                y = y * mask_expanded
            y_prev = y
            qs.append(y) #apply convolution

        # gating
        q_concat = torch.cat(qs, dim = 1)

        alpha = torch.sigmoid(self.gating(q_concat)) # a = B, 4C, L
        alpha = torch.reshape(alpha, (alpha.size(0), 4, self.c, alpha.size(-1))) # reshape to (B, 4, C, L) to do dilation channel-wise scaling

        q_final = torch.stack(qs, dim = 1)

        out = (alpha * q_final).sum(dim=1) # final per-channel gated sum

        out = out + xt # add in residual skip

        if mask_expanded is not None:
            out = out * mask_expanded

        return out.transpose(1, 2) # revert back to B, L, C
