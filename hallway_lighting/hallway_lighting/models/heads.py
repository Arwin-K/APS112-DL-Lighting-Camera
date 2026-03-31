"""Task-specific heads for hallway illuminance estimation."""

from __future__ import annotations

import torch
from torch import nn


class DensePredictionHead(nn.Module):
    """Small convolutional head for dense pixel-wise predictions."""

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        hidden_channels: int | None = None,
    ) -> None:
        super().__init__()
        hidden_channels = hidden_channels or in_channels
        self.head = nn.Sequential(
            nn.Conv2d(in_channels, hidden_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(hidden_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(hidden_channels, out_channels, kernel_size=1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Applies the dense prediction head."""

        if x.ndim != 4:
            raise ValueError(f"Expected a 4D feature map, got shape {tuple(x.shape)}")
        return self.head(x)


class SegmentationHead(DensePredictionHead):
    """Alias for a binary segmentation head."""


class ScalarRegressionHead(nn.Module):
    """Predicts one or more scene-level scalar values from a feature map."""

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        hidden_channels: int | None = None,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        hidden_channels = hidden_channels or max(in_channels // 2, 32)
        self.pool = nn.AdaptiveAvgPool2d((1, 1))
        self.mlp = nn.Sequential(
            nn.Flatten(),
            nn.Linear(in_channels, hidden_channels),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(hidden_channels, out_channels),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Applies global pooling followed by a small MLP."""

        if x.ndim != 4:
            raise ValueError(f"Expected a 4D feature map, got shape {tuple(x.shape)}")
        return self.mlp(self.pool(x))


class LuxStatisticsHead(nn.Module):
    """Predicts ordered lux statistics: p5, avg, and p95."""

    def __init__(self, in_channels: int) -> None:
        super().__init__()
        self.scalar_head = ScalarRegressionHead(in_channels=in_channels, out_channels=3)
        self.softplus = nn.Softplus(beta=1.0)

    def forward(self, x: torch.Tensor) -> dict[str, torch.Tensor]:
        """Returns ordered scalar statistics.

        The parameterization enforces:
        `low_lux_p5 <= avg_lux <= high_lux_p95`.
        """

        raw_values = self.scalar_head(x)
        low_lux_p5 = self.softplus(raw_values[:, 0:1])
        avg_offset = self.softplus(raw_values[:, 1:2])
        high_offset = self.softplus(raw_values[:, 2:3])
        avg_lux = low_lux_p5 + avg_offset
        high_lux_p95 = avg_lux + high_offset
        return {
            "low_lux_p5": low_lux_p5,
            "avg_lux": avg_lux,
            "high_lux_p95": high_lux_p95,
        }
