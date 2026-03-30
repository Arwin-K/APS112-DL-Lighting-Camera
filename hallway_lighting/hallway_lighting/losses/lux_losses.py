"""Losses for floor-plane illuminance regression."""

from __future__ import annotations

import torch
import torch.nn.functional as F


def masked_l1_loss(
    prediction: torch.Tensor,
    target: torch.Tensor,
    mask: torch.Tensor | None = None,
) -> torch.Tensor:
    """Computes an L1 loss with an optional binary mask."""

    error = torch.abs(prediction - target)
    if mask is None:
        return error.mean()

    valid = mask.float()
    return (error * valid).sum() / valid.sum().clamp_min(1.0)


def percentile_stat_loss(
    prediction: torch.Tensor,
    target: torch.Tensor,
    percentiles: tuple[float, float] = (5.0, 95.0),
) -> torch.Tensor:
    """Matches lower and upper percentile statistics between predicted and target lux."""

    flattened_prediction = prediction.flatten(start_dim=1)
    flattened_target = target.flatten(start_dim=1)
    losses: list[torch.Tensor] = []

    for percentile in percentiles:
        pred_value = torch.quantile(flattened_prediction, percentile / 100.0, dim=1)
        target_value = torch.quantile(flattened_target, percentile / 100.0, dim=1)
        losses.append(F.l1_loss(pred_value, target_value))

    return torch.stack(losses).mean()


def lux_map_loss(
    prediction: torch.Tensor,
    target: torch.Tensor,
    mask: torch.Tensor | None = None,
    percentile_weight: float = 0.2,
) -> torch.Tensor:
    """Combines dense and distributional illuminance losses."""

    dense = masked_l1_loss(prediction, target, mask=mask)
    stats = percentile_stat_loss(prediction, target)
    return dense + percentile_weight * stats
