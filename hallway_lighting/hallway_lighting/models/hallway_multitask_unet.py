"""Multitask U-Net for hallway illuminance and related outputs."""

from __future__ import annotations

import torch
from torch import nn
import torch.nn.functional as F

from .backbone import ConvBlock, build_backbone
from .heads import PixelRegressionHead, ScalarRegressionHead, SegmentationHead


class UpBlock(nn.Module):
    """Upsamples decoder features and merges them with a skip connection."""

    def __init__(self, in_channels: int, skip_channels: int, out_channels: int) -> None:
        super().__init__()
        self.upsample = nn.Upsample(scale_factor=2, mode="bilinear", align_corners=False)
        self.conv = ConvBlock(in_channels + skip_channels, out_channels)

    def forward(self, x: torch.Tensor, skip: torch.Tensor) -> torch.Tensor:
        x = self.upsample(x)
        if x.shape[-2:] != skip.shape[-2:]:
            x = F.interpolate(x, size=skip.shape[-2:], mode="bilinear", align_corners=False)
        x = torch.cat([x, skip], dim=1)
        return self.conv(x)


class HallwayMultitaskUNet(nn.Module):
    """Predicts dense illuminance, intrinsic, segmentation, and scalar power outputs.

    TODO:
        Add dataset-aware conditioning or missing-label masking once the exact
        multi-dataset batching strategy is finalized.
    """

    def __init__(self, config: dict[str, object]) -> None:
        super().__init__()
        self.backbone = build_backbone(config)
        encoder_channels = list(config.get("encoder_channels", [32, 64, 128, 256]))
        decoder_channels = list(config.get("decoder_channels", [256, 128, 64, 32]))

        if len(decoder_channels) != len(encoder_channels):
            raise ValueError("decoder_channels must match encoder_channels in length.")

        self.bottleneck = ConvBlock(encoder_channels[-1], decoder_channels[0])
        self.up_blocks = nn.ModuleList(
            [
                UpBlock(
                    in_channels=decoder_channels[index],
                    skip_channels=encoder_channels[-(index + 2)],
                    out_channels=decoder_channels[index + 1],
                )
                for index in range(len(encoder_channels) - 1)
            ]
        )

        final_channels = decoder_channels[-1]
        self.lux_head = PixelRegressionHead(final_channels, out_channels=1)
        self.reflectance_head = PixelRegressionHead(final_channels, out_channels=1)
        self.shading_head = PixelRegressionHead(final_channels, out_channels=1)
        self.floor_mask_head = SegmentationHead(final_channels, out_channels=1)
        self.uncertainty_head = PixelRegressionHead(final_channels, out_channels=1)
        self.power_head = ScalarRegressionHead(decoder_channels[0], out_channels=1)

    def forward(self, x: torch.Tensor) -> dict[str, torch.Tensor]:
        features = self.backbone(x)
        decoder = self.bottleneck(features[-1])

        for block, skip in zip(self.up_blocks, reversed(features[:-1]), strict=False):
            decoder = block(decoder, skip)

        lux_map = F.relu(self.lux_head(decoder))
        reflectance = torch.sigmoid(self.reflectance_head(decoder))
        shading = torch.sigmoid(self.shading_head(decoder))
        floor_mask_logits = self.floor_mask_head(decoder)
        uncertainty_log_var = self.uncertainty_head(decoder)
        estimated_power_w = F.relu(self.power_head(features[-1]))

        return {
            "lux_map": lux_map,
            "reflectance": reflectance,
            "shading": shading,
            "floor_mask_logits": floor_mask_logits,
            "uncertainty_log_var": uncertainty_log_var,
            "estimated_power_w": estimated_power_w,
        }
