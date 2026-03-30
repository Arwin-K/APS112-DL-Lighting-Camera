"""Task-specific decoder heads."""

from __future__ import annotations

import torch
from torch import nn


class PixelRegressionHead(nn.Module):
    """Predicts a single-channel dense regression map."""

    def __init__(self, in_channels: int, out_channels: int = 1) -> None:
        super().__init__()
        self.head = nn.Sequential(
            nn.Conv2d(in_channels, in_channels, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels, out_channels, kernel_size=1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.head(x)


class SegmentationHead(nn.Module):
    """Predicts segmentation logits."""

    def __init__(self, in_channels: int, out_channels: int = 1) -> None:
        super().__init__()
        self.head = nn.Conv2d(in_channels, out_channels, kernel_size=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.head(x)


class ScalarRegressionHead(nn.Module):
    """Predicts scene-level scalar values from the bottleneck feature map."""

    def __init__(self, in_channels: int, out_channels: int = 1) -> None:
        super().__init__()
        self.pool = nn.AdaptiveAvgPool2d((1, 1))
        self.head = nn.Sequential(
            nn.Flatten(),
            nn.Linear(in_channels, max(in_channels // 2, 8)),
            nn.ReLU(inplace=True),
            nn.Linear(max(in_channels // 2, 8), out_channels),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.head(self.pool(x))
