"""Shared encoder-decoder model for hallway illuminance estimation."""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import nn
import torch.nn.functional as F

from hallway_lighting.data.point_sampling import (
    PointTarget,
    build_canonical_point_targets,
    sample_point_values_batch,
)
from hallway_lighting.utils.geometry import (
    HallwayGeometryConfig,
    expand_coordinate_channels,
)

from .backbone import ConvBlock, build_backbone
from .heads import DensePredictionHead, LuxStatisticsHead, ScalarRegressionHead, SegmentationHead


class UpBlock(nn.Module):
    """Upsamples decoder features and fuses them with a skip connection."""

    def __init__(self, in_channels: int, skip_channels: int, out_channels: int) -> None:
        super().__init__()
        self.projection = nn.Conv2d(in_channels, out_channels, kernel_size=1)
        self.fuse = ConvBlock(out_channels + skip_channels, out_channels)

    def forward(self, x: torch.Tensor, skip: torch.Tensor) -> torch.Tensor:
        """Upsamples and merges with a same-scale skip feature."""

        x = F.interpolate(x, size=skip.shape[-2:], mode="bilinear", align_corners=False)
        x = self.projection(x)
        if x.shape[-2:] != skip.shape[-2:]:
            raise AssertionError("Upsampled decoder feature does not match skip spatial shape.")
        return self.fuse(torch.cat([x, skip], dim=1))


@dataclass(frozen=True)
class ModelInputSpec:
    """Tracks which optional input channels are expected by the model."""

    image_channels: int = 3
    use_floor_mask_input: bool = False
    use_coordinate_channels: bool = False

    @property
    def total_channels(self) -> int:
        """Returns the full expected channel count after concatenation."""

        return (
            self.image_channels
            + int(self.use_floor_mask_input)
            + (2 if self.use_coordinate_channels else 0)
        )


class HallwayMultitaskUNet(nn.Module):
    """Multitask hallway illuminance model with a shared encoder-decoder.

    Outputs:
    - dense floor lux map
    - scalar average lux
    - scalar p5 lux
    - scalar p95 lux
    - point-wise lux under and between fixtures
    - floor segmentation logits and probabilities
    - albedo/material proxy
    - gloss/specularity proxy
    - per-pixel uncertainty map
    - optional estimated power for downstream carbon heads
    """

    def __init__(self, config: dict[str, object]) -> None:
        super().__init__()
        self.input_spec = ModelInputSpec(
            image_channels=int(config.get("image_channels", config.get("in_channels", 3))),
            use_floor_mask_input=bool(config.get("use_floor_mask_input", False)),
            use_coordinate_channels=bool(config.get("use_coordinate_channels", False)),
        )
        self.fixture_count = int(config.get("fixture_count", 3))
        self.geometry = HallwayGeometryConfig(
            fixture_count=self.fixture_count,
            floor_y=float(config.get("floor_y", 0.72)),
            start_x=float(config.get("point_start_x", 0.2)),
            end_x=float(config.get("point_end_x", 0.8)),
        )

        backbone_config = dict(config)
        backbone_config["in_channels"] = self.input_spec.total_channels
        self.backbone = build_backbone(backbone_config)

        encoder_channels = list(self.backbone.spec.feature_channels)
        decoder_channels = list(config.get("decoder_channels", [256, 128, 64, 64]))
        if len(decoder_channels) != 4:
            raise ValueError("decoder_channels must contain four stages for the ResNet-18 decoder.")

        self.bottleneck = ConvBlock(encoder_channels[-1], decoder_channels[0])
        self.decoder_blocks = nn.ModuleList(
            [
                UpBlock(decoder_channels[0], encoder_channels[-2], decoder_channels[0]),
                UpBlock(decoder_channels[0], encoder_channels[-3], decoder_channels[1]),
                UpBlock(decoder_channels[1], encoder_channels[-4], decoder_channels[2]),
                UpBlock(decoder_channels[2], encoder_channels[-5], decoder_channels[3]),
            ]
        )
        self.final_refine = nn.Sequential(
            nn.Upsample(scale_factor=2, mode="bilinear", align_corners=False),
            ConvBlock(decoder_channels[-1], decoder_channels[-1]),
        )

        head_channels = decoder_channels[-1]
        self.floor_head = SegmentationHead(head_channels, out_channels=1)
        self.lux_head = DensePredictionHead(head_channels, out_channels=1)
        self.albedo_head = DensePredictionHead(head_channels, out_channels=3)
        self.gloss_head = DensePredictionHead(head_channels, out_channels=1)
        self.uncertainty_head = DensePredictionHead(head_channels, out_channels=1)
        self.statistics_head = LuxStatisticsHead(head_channels)
        self.power_head = ScalarRegressionHead(head_channels, out_channels=1)
        self.softplus = nn.Softplus(beta=1.0)

    def _prepare_model_inputs(
        self,
        image: torch.Tensor,
        floor_mask: torch.Tensor | None = None,
        coordinate_channels: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Concatenates optional inputs into the backbone input tensor."""

        if image.ndim != 4:
            raise ValueError(f"Expected image tensor [B, C, H, W], got shape {tuple(image.shape)}")
        if image.shape[1] != self.input_spec.image_channels:
            raise ValueError(
                f"Model expected {self.input_spec.image_channels} image channels, got {image.shape[1]}."
            )

        batch_size, _, height, width = image.shape
        input_tensors = [image]

        if self.input_spec.use_floor_mask_input:
            if floor_mask is None:
                floor_mask = image.new_zeros((batch_size, 1, height, width))
            elif floor_mask.ndim == 3:
                floor_mask = floor_mask.unsqueeze(1)
            elif floor_mask.ndim != 4:
                raise ValueError(
                    f"Expected floor_mask shape [B, 1, H, W] or [B, H, W], got {tuple(floor_mask.shape)}"
                )
            if floor_mask.shape[-2:] != (height, width):
                floor_mask = F.interpolate(floor_mask.float(), size=(height, width), mode="nearest")
            input_tensors.append(floor_mask.float())

        if self.input_spec.use_coordinate_channels:
            if coordinate_channels is None:
                coordinate_channels = expand_coordinate_channels(
                    batch_size=batch_size,
                    height=height,
                    width=width,
                    device=image.device,
                    dtype=image.dtype,
                )
            elif coordinate_channels.ndim == 3:
                coordinate_channels = coordinate_channels.unsqueeze(0).repeat(batch_size, 1, 1, 1)
            elif coordinate_channels.ndim != 4:
                raise ValueError(
                    "coordinate_channels must have shape [2, H, W] or [B, 2, H, W]."
                )
            if coordinate_channels.shape[1] != 2:
                raise ValueError(
                    f"Expected coordinate_channels to contain x/y planes, got shape {tuple(coordinate_channels.shape)}"
                )
            if coordinate_channels.shape[-2:] != (height, width):
                coordinate_channels = F.interpolate(
                    coordinate_channels.float(),
                    size=(height, width),
                    mode="bilinear",
                    align_corners=False,
                )
            input_tensors.append(coordinate_channels.float())

        combined = torch.cat(input_tensors, dim=1)
        if combined.shape[1] != self.input_spec.total_channels:
            raise AssertionError(
                f"Prepared input has {combined.shape[1]} channels, expected {self.input_spec.total_channels}."
            )
        return combined

    def _decode(self, features: list[torch.Tensor], output_size: tuple[int, int]) -> torch.Tensor:
        """Runs the U-Net decoder to the original image resolution."""

        stem, layer1, layer2, layer3, layer4 = features
        decoder = self.bottleneck(layer4)
        decoder = self.decoder_blocks[0](decoder, layer3)
        decoder = self.decoder_blocks[1](decoder, layer2)
        decoder = self.decoder_blocks[2](decoder, layer1)
        decoder = self.decoder_blocks[3](decoder, stem)
        decoder = self.final_refine(decoder)
        if decoder.shape[-2:] != output_size:
            decoder = F.interpolate(decoder, size=output_size, mode="bilinear", align_corners=False)
        return decoder

    def forward(
        self,
        image: torch.Tensor,
        floor_mask: torch.Tensor | None = None,
        coordinate_channels: torch.Tensor | None = None,
        point_targets: list[PointTarget] | None = None,
        fixture_count: int | None = None,
    ) -> dict[str, torch.Tensor | dict[str, torch.Tensor] | list[PointTarget]]:
        """Runs the multitask model and returns a structured output dictionary."""

        model_input = self._prepare_model_inputs(
            image=image,
            floor_mask=floor_mask,
            coordinate_channels=coordinate_channels,
        )
        features = self.backbone(model_input)
        decoder_features = self._decode(features, output_size=image.shape[-2:])

        floor_logits = self.floor_head(decoder_features)
        floor_mask_pred = torch.sigmoid(floor_logits)
        lux_map = self.softplus(self.lux_head(decoder_features))
        albedo_pred = torch.sigmoid(self.albedo_head(decoder_features))
        gloss_pred = torch.sigmoid(self.gloss_head(decoder_features))
        uncertainty_map = self.softplus(self.uncertainty_head(decoder_features)) + 1e-6

        statistics = self.statistics_head(decoder_features)
        estimated_power_w = self.softplus(self.power_head(decoder_features))

        effective_fixture_count = fixture_count or self.fixture_count
        effective_point_targets = point_targets or build_canonical_point_targets(
            fixture_count=effective_fixture_count,
            geometry=HallwayGeometryConfig(
                fixture_count=effective_fixture_count,
                floor_y=self.geometry.floor_y,
                start_x=self.geometry.start_x,
                end_x=self.geometry.end_x,
            ),
        )
        point_lux = sample_point_values_batch(lux_map, effective_point_targets)

        return {
            "lux_map": lux_map,
            "avg_lux": statistics["avg_lux"],
            "low_lux_p5": statistics["low_lux_p5"],
            "high_lux_p95": statistics["high_lux_p95"],
            "point_lux": point_lux,
            "point_targets": effective_point_targets,
            "floor_logits": floor_logits,
            "floor_mask_pred": floor_mask_pred,
            "albedo_pred": albedo_pred,
            "gloss_pred": gloss_pred,
            "uncertainty_map": uncertainty_map,
            "estimated_power_w": estimated_power_w,
        }
