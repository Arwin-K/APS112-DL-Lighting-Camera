"""Losses for hallway floor segmentation."""

from __future__ import annotations

import torch
import torch.nn.functional as F


def _zero_loss(reference: torch.Tensor) -> torch.Tensor:
    """Returns a differentiable zero scalar on the correct device."""

    return reference.new_zeros(())


def _expand_mask(reference: torch.Tensor, valid_mask: torch.Tensor | None) -> torch.Tensor | None:
    """Broadcasts an optional mask to a logits-shaped tensor."""

    if valid_mask is None:
        return None
    mask = valid_mask.float()
    while mask.ndim < reference.ndim:
        mask = mask.unsqueeze(1)
    if mask.shape != reference.shape:
        mask = torch.broadcast_to(mask, reference.shape)
    return mask


def dice_loss(
    logits: torch.Tensor,
    target: torch.Tensor | None,
    valid_mask: torch.Tensor | None = None,
    eps: float = 1e-6,
) -> torch.Tensor:
    """Computes a masked binary Dice loss from logits."""

    if target is None:
        return _zero_loss(logits)

    probabilities = torch.sigmoid(logits)
    target = target.float()
    mask = _expand_mask(logits, valid_mask)
    if mask is not None:
        probabilities = probabilities * mask
        target = target * mask

    intersection = (probabilities * target).sum(dim=(1, 2, 3))
    union = probabilities.sum(dim=(1, 2, 3)) + target.sum(dim=(1, 2, 3))
    dice = (2.0 * intersection + eps) / (union + eps)
    return 1.0 - dice.mean()


def segmentation_loss(
    logits: torch.Tensor,
    target: torch.Tensor | None,
    valid_mask: torch.Tensor | None = None,
) -> torch.Tensor:
    """Combines BCE and Dice while allowing masked or missing targets."""

    if target is None:
        return _zero_loss(logits)

    target = target.float()
    bce_map = F.binary_cross_entropy_with_logits(logits, target, reduction="none")
    if valid_mask is None:
        bce = bce_map.mean()
    else:
        mask = _expand_mask(bce_map, valid_mask)
        bce = (bce_map * mask).sum() / mask.sum().clamp_min(1.0)
    return bce + dice_loss(logits, target, valid_mask=valid_mask)
