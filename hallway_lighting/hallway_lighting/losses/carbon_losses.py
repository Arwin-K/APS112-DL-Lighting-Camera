"""Losses for carbon-related supervision."""

from __future__ import annotations

import torch
import torch.nn.functional as F


def carbon_interval_loss(
    predicted_power_w: torch.Tensor,
    target_carbon_kg: torch.Tensor,
    carbon_factor_kg_per_kwh: float,
    interval_hours: float,
) -> torch.Tensor:
    """Matches derived interval carbon against a supervision target."""

    predicted_energy_kwh = predicted_power_w / 1000.0 * interval_hours
    predicted_carbon_kg = predicted_energy_kwh * carbon_factor_kg_per_kwh
    return F.l1_loss(predicted_carbon_kg, target_carbon_kg)
