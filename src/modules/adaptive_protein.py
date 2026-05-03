"""
Wrapper for protein-axis adaptive diffusion stacks.
"""

from __future__ import annotations

from src.modules.adaptive_diffusion import AdaptiveDiffusionBlock


class AdaptiveProteinBlock(AdaptiveDiffusionBlock):
    """Specialisation of AdaptiveDiffusionBlock for protein similarity graphs."""

    def __init__(
        self,
        d_in: int,
        d_attn: int = 64,
        steps: int = 2,
        p: float = 0.9,
        tau: float = 1.0,
        dropout: float = 0.1,
    ) -> None:
        super().__init__(
            feature_dim=d_in,
            axis="protein",
            d_attn=d_attn,
            steps=steps,
            p=p,
            tau=tau,
            dropout=dropout,
            bidirectional=True,
        )
