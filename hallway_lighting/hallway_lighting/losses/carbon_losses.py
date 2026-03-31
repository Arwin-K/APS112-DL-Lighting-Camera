"""Losses for power, energy, and carbon estimation."""

from __future__ import annotations

import torch
import torch.nn.functional as F


def _zero_loss(reference: torch.Tensor) -> torch.Tensor:
    """Returns a differentiable zero scalar on the correct device."""

    return reference.new_zeros(())


def _masked_scalar_loss(
    prediction: torch.Tensor,
    target: torch.Tensor | None,
    valid_mask: torch.Tensor | None = None,
    beta: float = 1.0,
) -> torch.Tensor:
    """SmoothL1 loss with optional missing or masked targets."""

    if target is None:
        return _zero_loss(prediction)
    loss_map = F.smooth_l1_loss(prediction, target, beta=beta, reduction="none")
    if valid_mask is None:
        return loss_map.mean()
    mask = valid_mask.float()
    while mask.ndim < loss_map.ndim:
        mask = mask.unsqueeze(-1)
    mask = torch.broadcast_to(mask, loss_map.shape)
    return (loss_map * mask).sum() / mask.sum().clamp_min(1.0)


def power_regression_loss(
    predicted_power_w: torch.Tensor,
    target_power_w: torch.Tensor | None,
    valid_mask: torch.Tensor | None = None,
    beta: float = 1.0,
) -> torch.Tensor:
    """Supervises an estimated lighting-power head when labels are available."""

    return _masked_scalar_loss(predicted_power_w, target_power_w, valid_mask=valid_mask, beta=beta)


def estimate_energy_from_power(
    predicted_power_w: torch.Tensor,
    interval_hours: float,
) -> torch.Tensor:
    """Converts predicted power to interval energy."""

    return predicted_power_w / 1000.0 * interval_hours


def estimate_carbon_from_power(
    predicted_power_w: torch.Tensor,
    carbon_factor_kg_per_kwh: float,
    interval_hours: float,
) -> torch.Tensor:
    """Converts predicted power to interval carbon emissions."""

    predicted_energy_kwh = estimate_energy_from_power(predicted_power_w, interval_hours=interval_hours)
    return predicted_energy_kwh * carbon_factor_kg_per_kwh


def carbon_interval_loss(
    predicted_power_w: torch.Tensor,
    target_carbon_kg: torch.Tensor | None,
    carbon_factor_kg_per_kwh: float,
    interval_hours: float,
    valid_mask: torch.Tensor | None = None,
    beta: float = 1.0,
) -> torch.Tensor:
    """Matches derived interval carbon emissions against supervision targets."""

    predicted_carbon_kg = estimate_carbon_from_power(
        predicted_power_w=predicted_power_w,
        carbon_factor_kg_per_kwh=carbon_factor_kg_per_kwh,
        interval_hours=interval_hours,
    )
    return _masked_scalar_loss(predicted_carbon_kg, target_carbon_kg, valid_mask=valid_mask, beta=beta)
