"""Readable convolutional backbone used by the multitask U-Net."""

from __future__ import annotations

import torch
from torch import nn


class ConvBlock(nn.Module):
    """Two-layer convolution block with batch normalization."""

    def __init__(self, in_channels: int, out_channels: int) -> None:
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.block(x)


class DownsampleBlock(nn.Module):
    """Downsamples once, then applies a convolution block."""

    def __init__(self, in_channels: int, out_channels: int) -> None:
        super().__init__()
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.conv = ConvBlock(in_channels, out_channels)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.conv(self.pool(x))


class SimpleConvBackbone(nn.Module):
    """Small encoder that exposes intermediate feature maps for skip connections."""

    def __init__(self, in_channels: int = 3, encoder_channels: list[int] | tuple[int, ...] = (32, 64, 128, 256)) -> None:
        super().__init__()
        if len(encoder_channels) < 2:
            raise ValueError("encoder_channels must contain at least two stages.")

        self.stem = ConvBlock(in_channels, encoder_channels[0])
        self.down_blocks = nn.ModuleList(
            [
                DownsampleBlock(encoder_channels[index], encoder_channels[index + 1])
                for index in range(len(encoder_channels) - 1)
            ]
        )

    def forward(self, x: torch.Tensor) -> list[torch.Tensor]:
        features = [self.stem(x)]
        for block in self.down_blocks:
            features.append(block(features[-1]))
        return features


def build_backbone(config: dict[str, object]) -> SimpleConvBackbone:
    """Builds the default backbone from a config dictionary."""

    return SimpleConvBackbone(
        in_channels=int(config.get("in_channels", 3)),
        encoder_channels=list(config.get("encoder_channels", [32, 64, 128, 256])),
    )
