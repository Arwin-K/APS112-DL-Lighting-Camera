"""Canonical hallway point sampling helpers."""

from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path
from typing import Mapping, Sequence

import numpy as np
import torch
import torch.nn.functional as F

from hallway_lighting.utils.geometry import (
    HallwayGeometryConfig,
    canonical_between_fixture_positions,
    canonical_fixture_positions,
    normalized_to_grid_sample_coordinates,
    normalized_to_pixel_coordinates,
)


@dataclass(frozen=True)
class PointTarget:
    """Named normalized point used for hallway lux sampling."""

    name: str
    x: float
    y: float
    group: str


def build_canonical_point_targets(
    fixture_count: int = 3,
    geometry: HallwayGeometryConfig | None = None,
) -> list[PointTarget]:
    """Builds canonical hallway floor points under and between fixtures."""

    geometry = geometry or HallwayGeometryConfig(fixture_count=fixture_count)
    fixture_positions = canonical_fixture_positions(
        fixture_count=fixture_count,
        start_x=geometry.start_x,
        end_x=geometry.end_x,
        floor_y=geometry.floor_y,
    )
    between_positions = canonical_between_fixture_positions(
        fixture_count=fixture_count,
        start_x=geometry.start_x,
        end_x=geometry.end_x,
        floor_y=geometry.floor_y,
    )

    points: list[PointTarget] = []
    for fixture_index, (x_value, y_value) in enumerate(fixture_positions, start=1):
        points.append(
            PointTarget(
                name=f"under_fixture_{fixture_index}",
                x=x_value,
                y=y_value,
                group="under_fixture",
            )
        )

    for pair_index, (x_value, y_value) in enumerate(between_positions, start=1):
        points.append(
            PointTarget(
                name=f"between_fixture_{pair_index}_{pair_index + 1}",
                x=x_value,
                y=y_value,
                group="between_fixture",
            )
        )

    return points


def default_hallway_points(fixture_count: int = 3) -> list[PointTarget]:
    """Returns the default canonical hallway point layout."""

    return build_canonical_point_targets(fixture_count=fixture_count)


def load_point_targets(path: str | Path) -> list[PointTarget]:
    """Loads coordinate-based point definitions from JSON.

    Supported formats:
    - `{"points": [{"name": ..., "x": ..., "y": ..., "group": ...}, ...]}`
    - `{"point_name": {"x": 0.3, "y": 0.7, "group": "under_fixture"}, ...}`

    This loader is for coordinate definitions used at inference time.
    Flat numeric point-target JSON files used by the custom hallway dataset
    should be loaded with `load_point_target_values(...)` in `custom_hallway.py`.
    """

    with Path(path).open("r", encoding="utf-8") as handle:
        payload = json.load(handle)

    if isinstance(payload, dict) and "points" in payload:
        return [PointTarget(**item) for item in payload["points"]]

    if isinstance(payload, dict):
        point_targets: list[PointTarget] = []
        for point_name, point_payload in payload.items():
            if not isinstance(point_payload, Mapping):
                raise ValueError(
                    f"Point definition '{point_name}' must be an object with x/y coordinates."
                )
            point_targets.append(
                PointTarget(
                    name=point_name,
                    x=float(point_payload["x"]),
                    y=float(point_payload["y"]),
                    group=str(point_payload.get("group", "custom")),
                )
            )
        return point_targets

    raise ValueError(f"Unsupported point definition JSON format: {path}")


def _as_lux_tensor(lux_map: torch.Tensor | np.ndarray) -> torch.Tensor:
    """Normalizes a lux map to shape `[B, 1, H, W]`."""

    if isinstance(lux_map, np.ndarray):
        tensor = torch.from_numpy(lux_map)
    else:
        tensor = lux_map

    if tensor.ndim == 2:
        tensor = tensor.unsqueeze(0).unsqueeze(0)
    elif tensor.ndim == 3:
        if tensor.shape[0] == 1:
            tensor = tensor.unsqueeze(0)
        elif tensor.shape[0] > 1:
            tensor = tensor.unsqueeze(1)
    elif tensor.ndim != 4:
        raise ValueError(f"Expected a 2D, 3D, or 4D lux map, got shape {tuple(tensor.shape)}")

    if tensor.shape[1] != 1:
        raise ValueError(f"Expected a single-channel lux map, got shape {tuple(tensor.shape)}")
    return tensor.float()


def sample_point_values_batch(
    lux_map: torch.Tensor,
    points: Sequence[PointTarget],
) -> dict[str, torch.Tensor]:
    """Samples a batched lux map at named normalized points.

    Returns:
        Dictionary mapping point names to tensors of shape `[B]`.
    """

    lux_tensor = _as_lux_tensor(lux_map)
    if not points:
        return {}

    batch_size = lux_tensor.shape[0]
    grid_coordinates = [
        normalized_to_grid_sample_coordinates(point.x, point.y)
        for point in points
    ]
    grid = torch.tensor(
        grid_coordinates,
        device=lux_tensor.device,
        dtype=lux_tensor.dtype,
    ).view(1, len(points), 1, 2)
    grid = grid.repeat(batch_size, 1, 1, 1)

    sampled = F.grid_sample(
        lux_tensor,
        grid,
        mode="bilinear",
        padding_mode="border",
        align_corners=True,
    )
    sampled = sampled.squeeze(1).squeeze(-1)

    return {
        point.name: sampled[:, index]
        for index, point in enumerate(points)
    }


def sample_values_at_points(
    lux_map: torch.Tensor | np.ndarray,
    points: Sequence[PointTarget],
) -> dict[str, float]:
    """Samples a single lux map and returns float values."""

    lux_tensor = _as_lux_tensor(lux_map)
    if lux_tensor.shape[0] != 1:
        raise ValueError(
            "sample_values_at_points expects a single sample. Use sample_point_values_batch for batched input."
        )

    point_values_batch = sample_point_values_batch(lux_tensor, points)
    return {name: float(values[0].item()) for name, values in point_values_batch.items()}


def sample_values_with_nearest_pixels(
    lux_map: torch.Tensor | np.ndarray,
    points: Sequence[PointTarget],
) -> dict[str, float]:
    """Samples a single lux map with nearest-pixel indexing.

    This helper is useful for quick debugging when bilinear interpolation is not
    desired in notebook experiments.
    """

    if isinstance(lux_map, torch.Tensor):
        lux_array = lux_map.detach().cpu().squeeze().numpy()
    else:
        lux_array = np.asarray(lux_map).squeeze()

    if lux_array.ndim != 2:
        raise ValueError(f"Expected a single 2D lux map, got shape {tuple(lux_array.shape)}")

    height, width = lux_array.shape
    point_values: dict[str, float] = {}
    for point in points:
        x_pixel, y_pixel = normalized_to_pixel_coordinates(point.x, point.y, width, height)
        point_values[point.name] = float(lux_array[y_pixel, x_pixel])
    return point_values
