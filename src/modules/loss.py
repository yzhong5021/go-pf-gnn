"""
loss.py

BCE with logits loss
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class BCEWithLogits(nn.Module):

    def __init__(self, pos_weight):
        super().__init__()
        self.register_buffer("pos_weight", pos_weight if pos_weight is not None else None)

    def forward(self, logits, targets):
        logits = logits.float()
        targets = targets.float()

        loss = F.binary_cross_entropy_with_logits(
            logits, targets, pos_weight=self.pos_weight, reduction="none"
        )

        return loss.mean()


class FocalLoss(nn.Module):
    """Binary focal loss with logits for multi-label classification."""

    def __init__(self, focusing: float = 2.0, balancing: float = 0.25, eps: float = 1e-8):
        super().__init__()
        self.focusing = float(focusing)
        self.balancing = float(balancing)
        self.eps = float(eps)

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        logits = logits.float()
        targets = targets.float()
        bce = F.binary_cross_entropy_with_logits(logits, targets, reduction="none")
        p_t = torch.exp(-bce)
        p_t = torch.clamp(p_t, min=self.eps, max=1.0 - self.eps)
        alpha_t = self.balancing * targets + (1.0 - self.balancing) * (1.0 - targets)
        loss = alpha_t * torch.pow(1.0 - p_t, self.focusing) * bce
        return loss.mean()


class AsymmetricLoss(nn.Module):
    """Asymmetric loss for multi-label classification with logits."""

    def __init__(
        self,
        gamma_positive: float = 0.0,
        gamma_negative: float = 4.0,
        clip: float = 0.025,
        eps: float = 1e-8,
    ):
        super().__init__()
        self.gamma_positive = float(gamma_positive)
        self.gamma_negative = float(gamma_negative)
        self.clip = float(clip)
        self.eps = float(eps)

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        logits = logits.float()
        targets = targets.float()
        probs = torch.sigmoid(logits)
        probs = torch.clamp(probs, min=self.eps, max=1.0 - self.eps)
        probs_neg = probs
        if self.clip > 0.0:
            probs_neg = torch.clamp(probs + self.clip, max=1.0)

        pos_loss = targets * torch.log(probs)
        neg_loss = (1.0 - targets) * torch.log(torch.clamp(1.0 - probs_neg, min=self.eps))

        if self.gamma_positive > 0.0:
            pos_loss = pos_loss * torch.pow(1.0 - probs, self.gamma_positive)
        if self.gamma_negative > 0.0:
            neg_loss = neg_loss * torch.pow(probs, self.gamma_negative)

        return -(pos_loss + neg_loss).mean()
