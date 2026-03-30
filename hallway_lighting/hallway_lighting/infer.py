"""Single-image inference helpers for notebooks and deployment experiments."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Sequence

from PIL import Image
import torch

from hallway_lighting.data.point_sampling import PointTarget, sample_values_at_points
from hallway_lighting.data.transforms import build_image_transform
from hallway_lighting.utils.carbon import (
    estimate_interval_carbon_kg,
    estimate_interval_energy_kwh,
)
from hallway_lighting.utils.metrics import summarize_lux_map


@dataclass
class InferenceBatch:
    """Packaged single-image batch for model inference."""

    image_tensor: torch.Tensor
    image_path: str


@dataclass
class InferenceOutput:
    """Compact user-facing inference result."""

    lux_summary: dict[str, float]
    point_lux: dict[str, float]
    estimated_power_w: float
    interval_energy_kwh: float
    interval_carbon_kg: float
    raw_outputs: dict[str, Any]


def preprocess_single_image(
    image_path: str | Path,
    image_size: tuple[int, int] = (256, 256),
) -> InferenceBatch:
    """Loads and preprocesses a single RGB image for model inference."""

    transform = build_image_transform(image_size=image_size)
    image = Image.open(image_path).convert("RGB")
    image_tensor = transform(image).unsqueeze(0)
    return InferenceBatch(image_tensor=image_tensor, image_path=str(image_path))


@torch.no_grad()
def run_single_image_inference(
    model: torch.nn.Module,
    image_path: str | Path,
    device: str = "cpu",
    image_size: tuple[int, int] = (256, 256),
    point_targets: Sequence[PointTarget] | None = None,
    carbon_factor_kg_per_kwh: float = 0.35,
    interval_hours: float = 1.0,
) -> InferenceOutput:
    """Runs a single-image forward pass and produces lux and carbon summaries."""

    batch = preprocess_single_image(image_path=image_path, image_size=image_size)
    model = model.to(device)
    model.eval()
    outputs = model(batch.image_tensor.to(device))

    lux_map = outputs["lux_map"].detach().cpu()
    lux_summary = summarize_lux_map(lux_map)
    point_lux = {}
    if point_targets:
        point_lux = sample_values_at_points(lux_map, point_targets)

    estimated_power_w = float(outputs["estimated_power_w"].detach().cpu().view(-1)[0])
    interval_energy_kwh = estimate_interval_energy_kwh(
        power_w=estimated_power_w,
        interval_hours=interval_hours,
    )
    interval_carbon_kg = estimate_interval_carbon_kg(
        energy_kwh=interval_energy_kwh,
        carbon_factor_kg_per_kwh=carbon_factor_kg_per_kwh,
    )

    return InferenceOutput(
        lux_summary=lux_summary,
        point_lux=point_lux,
        estimated_power_w=estimated_power_w,
        interval_energy_kwh=interval_energy_kwh,
        interval_carbon_kg=interval_carbon_kg,
        raw_outputs={key: value.detach().cpu() for key, value in outputs.items()},
    )
