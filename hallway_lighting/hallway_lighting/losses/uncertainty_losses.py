"""Losses for predicted uncertainty maps."""

from __future__ import annotations

import torch
import torch.nn.functional as F


def _zero_loss(reference: torch.Tensor) -> torch.Tensor:
    """Returns a differentiable zero scalar on the correct device."""

    return reference.new_zeros(())


def _expand_mask(reference: torch.Tensor, valid_mask: torch.Tensor | None) -> torch.Tensor | None:
    """Broadcasts an optional mask to match a dense tensor."""

    if valid_mask is None:
        return None
    mask = valid_mask.float()
    while mask.ndim < reference.ndim:
        mask = mask.unsqueeze(1)
    if mask.shape != reference.shape:
        mask = torch.broadcast_to(mask, reference.shape)
    return mask


def uncertainty_regularization_loss(
    uncertainty_map: torch.Tensor,
    valid_mask: torch.Tensor | None = None,
    weight: float = 1.0,
) -> torch.Tensor:
    """Penalizes unnecessarily large predicted uncertainty."""

    regularizer = uncertainty_map
    if valid_mask is None:
        return weight * regularizer.mean()
    mask = _expand_mask(regularizer, valid_mask)
    return weight * (regularizer * mask).sum() / mask.sum().clamp_min(1.0)


def heteroscedastic_l1_loss(
    prediction: torch.Tensor,
    target: torch.Tensor | None,
    uncertainty_map: torch.Tensor,
    valid_mask: torch.Tensor | None = None,
    eps: float = 1e-6,
) -> torch.Tensor:
    """Regression loss with a learned positive uncertainty map."""

    if target is None:
        return _zero_loss(prediction)

    variance = torch.clamp(uncertainty_map, min=eps) ** 2
    log_variance = torch.log(variance)
    precision = torch.exp(-log_variance)
    error = torch.abs(prediction - target)
    loss_map = precision * error + log_variance

    if valid_mask is None:
        return loss_map.mean()
    mask = _expand_mask(loss_map, valid_mask)
    return (loss_map * mask).sum() / mask.sum().clamp_min(1.0)
