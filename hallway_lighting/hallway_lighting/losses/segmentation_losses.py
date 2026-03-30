"""Losses for floor mask segmentation."""

from __future__ import annotations

import torch
import torch.nn.functional as F


def dice_loss(logits: torch.Tensor, target: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    """Computes a binary Dice loss from logits."""

    probabilities = torch.sigmoid(logits)
    target = target.float()
    intersection = (probabilities * target).sum(dim=(1, 2, 3))
    union = probabilities.sum(dim=(1, 2, 3)) + target.sum(dim=(1, 2, 3))
    dice = (2.0 * intersection + eps) / (union + eps)
    return 1.0 - dice.mean()


def segmentation_loss(logits: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    """Combines BCE and Dice for stable binary floor-mask learning."""

    bce = F.binary_cross_entropy_with_logits(logits, target.float())
    return bce + dice_loss(logits, target)
