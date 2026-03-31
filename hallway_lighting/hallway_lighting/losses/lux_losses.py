"""Losses for dense and scalar illuminance prediction."""

from __future__ import annotations

from typing import Mapping

import torch
import torch.nn.functional as F


def _zero_loss(reference: torch.Tensor) -> torch.Tensor:
    """Returns a differentiable zero scalar on the correct device."""

    return reference.new_zeros(())


def _prepare_mask(
    reference: torch.Tensor,
    valid_mask: torch.Tensor | None = None,
) -> torch.Tensor | None:
    """Broadcasts an optional valid mask to match a reference tensor."""

    if valid_mask is None:
        return None
    mask = valid_mask.float()
    while mask.ndim < reference.ndim:
        mask = mask.unsqueeze(1)
    if mask.shape != reference.shape:
        mask = torch.broadcast_to(mask, reference.shape)
    return mask


def _masked_reduction(loss_map: torch.Tensor, valid_mask: torch.Tensor | None = None) -> torch.Tensor:
    """Computes a masked mean reduction."""

    if valid_mask is None:
        return loss_map.mean()
    mask = _prepare_mask(loss_map, valid_mask)
    weighted = loss_map * mask
    return weighted.sum() / mask.sum().clamp_min(1.0)


def masked_huber_loss(
    prediction: torch.Tensor,
    target: torch.Tensor | None,
    valid_mask: torch.Tensor | None = None,
    beta: float = 1.0,
) -> torch.Tensor:
    """SmoothL1 / Huber loss with optional masking and absent targets."""

    if target is None:
        return _zero_loss(prediction)
    loss_map = F.smooth_l1_loss(prediction, target, beta=beta, reduction="none")
    return _masked_reduction(loss_map, valid_mask=valid_mask)


def log_lux_smooth_l1_loss(
    prediction: torch.Tensor,
    target: torch.Tensor | None,
    valid_mask: torch.Tensor | None = None,
    beta: float = 0.1,
    eps: float = 1e-3,
) -> torch.Tensor:
    """Dense lux-map loss in log space for stable high-dynamic-range supervision."""

    if target is None:
        return _zero_loss(prediction)
    prediction_log = torch.log1p(torch.clamp(prediction, min=0.0) + eps)
    target_log = torch.log1p(torch.clamp(target, min=0.0) + eps)
    return masked_huber_loss(prediction_log, target_log, valid_mask=valid_mask, beta=beta)


def avg_lux_loss(
    prediction: torch.Tensor,
    target: torch.Tensor | None,
    valid_mask: torch.Tensor | None = None,
    beta: float = 1.0,
) -> torch.Tensor:
    """Scalar average-lux regression loss."""

    return masked_huber_loss(prediction, target, valid_mask=valid_mask, beta=beta)


def p5_lux_loss(
    prediction: torch.Tensor,
    target: torch.Tensor | None,
    valid_mask: torch.Tensor | None = None,
    beta: float = 1.0,
) -> torch.Tensor:
    """Scalar p5 lux regression loss."""

    return masked_huber_loss(prediction, target, valid_mask=valid_mask, beta=beta)


def p95_lux_loss(
    prediction: torch.Tensor,
    target: torch.Tensor | None,
    valid_mask: torch.Tensor | None = None,
    beta: float = 1.0,
) -> torch.Tensor:
    """Scalar p95 lux regression loss."""

    return masked_huber_loss(prediction, target, valid_mask=valid_mask, beta=beta)


def pointwise_lux_loss(
    prediction: Mapping[str, torch.Tensor] | None,
    target: Mapping[str, torch.Tensor | float] | None,
    valid_mask: torch.Tensor | Mapping[str, torch.Tensor] | None = None,
    beta: float = 1.0,
) -> torch.Tensor:
    """Point-wise lux loss across the shared point names present in both dictionaries."""

    if prediction is None:
        raise ValueError("prediction point dictionary must not be None.")
    if not prediction:
        first_tensor = next(iter(prediction.values()), None)
        if first_tensor is None:
            return torch.tensor(0.0)
    reference = next(iter(prediction.values()))
    if target is None:
        return _zero_loss(reference)

    shared_names = [name for name in prediction.keys() if name in target]
    if not shared_names:
        return _zero_loss(reference)

    losses: list[torch.Tensor] = []
    for point_name in shared_names:
        prediction_value = prediction[point_name]
        target_value = target[point_name]
        if not isinstance(target_value, torch.Tensor):
            target_value = torch.full_like(prediction_value, float(target_value))
        point_mask: torch.Tensor | None = None
        if isinstance(valid_mask, Mapping):
            point_mask = valid_mask.get(point_name)
        elif isinstance(valid_mask, torch.Tensor):
            point_mask = valid_mask
        losses.append(masked_huber_loss(prediction_value, target_value, valid_mask=point_mask, beta=beta))

    return torch.stack(losses).mean()


def lux_map_loss(
    prediction: torch.Tensor,
    target: torch.Tensor | None,
    mask: torch.Tensor | None = None,
    beta: float = 0.1,
    eps: float = 1e-3,
) -> torch.Tensor:
    """Compatibility wrapper for the primary dense lux loss."""

    return log_lux_smooth_l1_loss(prediction, target, valid_mask=mask, beta=beta, eps=eps)
