"""Losses for intrinsic-image related outputs."""

from __future__ import annotations

import torch
import torch.nn.functional as F


def reconstruct_intensity(reflectance: torch.Tensor, shading: torch.Tensor) -> torch.Tensor:
    """Reconstructs grayscale intensity from intrinsic components."""

    return torch.clamp(reflectance * shading, min=0.0, max=1.0)


def intrinsic_reconstruction_loss(
    image: torch.Tensor,
    reflectance: torch.Tensor,
    shading: torch.Tensor,
) -> torch.Tensor:
    """Encourages intrinsic outputs to explain the observed image intensity."""

    grayscale = image.mean(dim=1, keepdim=True)
    reconstruction = reconstruct_intensity(reflectance, shading)
    return F.l1_loss(reconstruction, grayscale)
