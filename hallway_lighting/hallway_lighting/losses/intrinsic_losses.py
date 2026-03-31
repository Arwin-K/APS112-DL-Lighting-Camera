"""Losses for material/albedo and gloss proxy predictions."""

from __future__ import annotations

import torch
import torch.nn.functional as F


def _zero_loss(reference: torch.Tensor) -> torch.Tensor:
    """Returns a differentiable zero scalar on the correct device."""

    return reference.new_zeros(())


def _broadcast_mask(reference: torch.Tensor, valid_mask: torch.Tensor | None) -> torch.Tensor | None:
    """Broadcasts an optional valid mask."""

    if valid_mask is None:
        return None
    mask = valid_mask.float()
    while mask.ndim < reference.ndim:
        mask = mask.unsqueeze(1)
    if mask.shape != reference.shape:
        mask = torch.broadcast_to(mask, reference.shape)
    return mask


def _masked_l1(
    prediction: torch.Tensor,
    target: torch.Tensor | None,
    valid_mask: torch.Tensor | None = None,
) -> torch.Tensor:
    """Computes a masked L1 loss and safely handles missing targets."""

    if target is None:
        return _zero_loss(prediction)
    loss_map = torch.abs(prediction - target)
    if valid_mask is None:
        return loss_map.mean()
    mask = _broadcast_mask(loss_map, valid_mask)
    return (loss_map * mask).sum() / mask.sum().clamp_min(1.0)


def albedo_regression_loss(
    prediction: torch.Tensor,
    target: torch.Tensor | None,
    valid_mask: torch.Tensor | None = None,
) -> torch.Tensor:
    """Supervises the material/albedo proxy head."""

    return _masked_l1(prediction, target, valid_mask=valid_mask)


def gloss_regression_loss(
    prediction: torch.Tensor,
    target: torch.Tensor | None,
    valid_mask: torch.Tensor | None = None,
) -> torch.Tensor:
    """Supervises the gloss/specularity proxy head."""

    return _masked_l1(prediction, target, valid_mask=valid_mask)


def intrinsic_reconstruction_loss(
    albedo_prediction: torch.Tensor,
    albedo_target: torch.Tensor | None = None,
    gloss_prediction: torch.Tensor | None = None,
    gloss_target: torch.Tensor | None = None,
    valid_mask: torch.Tensor | None = None,
    gloss_weight: float = 1.0,
) -> torch.Tensor:
    """Compatibility wrapper for combined intrinsic proxy supervision."""

    total = albedo_regression_loss(
        prediction=albedo_prediction,
        target=albedo_target,
        valid_mask=valid_mask,
    )
    if gloss_prediction is not None:
        total = total + gloss_weight * gloss_regression_loss(
            prediction=gloss_prediction,
            target=gloss_target,
            valid_mask=valid_mask,
        )
    return total
