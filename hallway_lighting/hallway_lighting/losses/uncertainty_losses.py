"""Losses that use predicted uncertainty."""

from __future__ import annotations

import torch


def heteroscedastic_l1_loss(
    prediction: torch.Tensor,
    target: torch.Tensor,
    log_variance: torch.Tensor,
) -> torch.Tensor:
    """Regression loss with learned per-pixel uncertainty."""

    precision = torch.exp(-log_variance)
    error = torch.abs(prediction - target)
    return (precision * error + log_variance).mean()
