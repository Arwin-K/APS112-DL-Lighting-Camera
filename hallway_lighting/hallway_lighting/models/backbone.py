"""Encoder backbones for hallway lighting estimation."""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import nn
from torchvision.models import ResNet18_Weights, resnet18


class ConvBlock(nn.Module):
    """A readable two-layer convolution block used throughout the decoder."""

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
        """Applies the convolution block."""

        return self.block(x)


@dataclass(frozen=True)
class BackboneSpec:
    """Configuration summary for the chosen encoder."""

    name: str
    in_channels: int
    feature_channels: tuple[int, ...]


def _adapt_resnet_input_conv(conv: nn.Conv2d, in_channels: int) -> nn.Conv2d:
    """Adapts the first ResNet convolution to a custom input channel count."""

    if conv.in_channels == in_channels:
        return conv

    adapted_conv = nn.Conv2d(
        in_channels,
        conv.out_channels,
        kernel_size=conv.kernel_size,
        stride=conv.stride,
        padding=conv.padding,
        bias=conv.bias is not None,
    )

    with torch.no_grad():
        original_weight = conv.weight.data
        adapted_conv.weight.zero_()

        shared_channels = min(in_channels, conv.in_channels)
        adapted_conv.weight[:, :shared_channels] = original_weight[:, :shared_channels]

        if in_channels > conv.in_channels:
            mean_channel = original_weight.mean(dim=1, keepdim=True)
            for channel_index in range(conv.in_channels, in_channels):
                adapted_conv.weight[:, channel_index : channel_index + 1] = mean_channel

    return adapted_conv


class ResNet18Backbone(nn.Module):
    """ResNet-18 encoder that exposes skip features for a U-Net decoder."""

    def __init__(
        self,
        in_channels: int = 3,
        pretrained: bool = False,
    ) -> None:
        super().__init__()
        weights = ResNet18_Weights.DEFAULT if pretrained else None
        encoder = resnet18(weights=weights)
        encoder.conv1 = _adapt_resnet_input_conv(encoder.conv1, in_channels)

        self.conv1 = encoder.conv1
        self.bn1 = encoder.bn1
        self.relu = encoder.relu
        self.maxpool = encoder.maxpool
        self.layer1 = encoder.layer1
        self.layer2 = encoder.layer2
        self.layer3 = encoder.layer3
        self.layer4 = encoder.layer4
        self.spec = BackboneSpec(
            name="resnet18",
            in_channels=in_channels,
            feature_channels=(64, 64, 128, 256, 512),
        )

    def forward(self, x: torch.Tensor) -> list[torch.Tensor]:
        """Returns multi-scale encoder features for skip connections.

        Output feature order:
        - stem: /2 resolution
        - layer1: /4 resolution
        - layer2: /8 resolution
        - layer3: /16 resolution
        - layer4: /32 resolution
        """

        if x.ndim != 4:
            raise ValueError(f"Expected 4D input tensor [B, C, H, W], got shape {tuple(x.shape)}")
        if x.shape[1] != self.spec.in_channels:
            raise ValueError(
                f"Backbone expected {self.spec.in_channels} input channels, got {x.shape[1]}."
            )

        stem = self.relu(self.bn1(self.conv1(x)))
        layer1 = self.layer1(self.maxpool(stem))
        layer2 = self.layer2(layer1)
        layer3 = self.layer3(layer2)
        layer4 = self.layer4(layer3)
        return [stem, layer1, layer2, layer3, layer4]


def build_backbone(config: dict[str, object]) -> ResNet18Backbone:
    """Builds the configured backbone.

    Supported encoders:
    - `resnet18`
    """

    encoder_name = str(config.get("encoder_name", "resnet18")).lower()
    in_channels = int(config.get("in_channels", 3))
    pretrained = bool(config.get("pretrained", False))

    if encoder_name != "resnet18":
        raise ValueError(
            f"Unsupported encoder '{encoder_name}'. Only 'resnet18' is implemented in this phase."
        )

    return ResNet18Backbone(in_channels=in_channels, pretrained=pretrained)
