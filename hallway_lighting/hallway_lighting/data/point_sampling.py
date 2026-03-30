"""Point-wise sampling helpers for hallway illuminance reporting."""

from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path
from typing import Sequence

import numpy as np
import torch

from hallway_lighting.utils.geometry import normalized_to_pixel_coordinates


@dataclass(frozen=True)
class PointTarget:
    """Normalized point annotation for hallway lux sampling."""

    name: str
    x: float
    y: float
    group: str


def default_hallway_points() -> list[PointTarget]:
    """Returns a readable default point layout along a hallway center line."""

    return [
        PointTarget(name="under_fixture_1", x=0.2, y=0.72, group="under_fixture"),
        PointTarget(name="between_fixtures_1", x=0.35, y=0.72, group="between_fixtures"),
        PointTarget(name="under_fixture_2", x=0.5, y=0.72, group="under_fixture"),
        PointTarget(name="between_fixtures_2", x=0.65, y=0.72, group="between_fixtures"),
        PointTarget(name="under_fixture_3", x=0.8, y=0.72, group="under_fixture"),
    ]


def load_point_targets(path: str | Path) -> list[PointTarget]:
    """Loads point targets from the JSON template format."""

    with Path(path).open("r", encoding="utf-8") as handle:
        payload = json.load(handle)
    return [PointTarget(**item) for item in payload.get("points", [])]


def sample_values_at_points(
    lux_map: torch.Tensor | np.ndarray,
    points: Sequence[PointTarget],
) -> dict[str, float]:
    """Samples a 2D lux map at normalized point target locations."""

    if isinstance(lux_map, torch.Tensor):
        lux_array = lux_map.detach().cpu().squeeze().numpy()
    else:
        lux_array = np.asarray(lux_map).squeeze()

    height, width = lux_array.shape[-2:]
    point_values: dict[str, float] = {}

    for point in points:
        x_px, y_px = normalized_to_pixel_coordinates(point.x, point.y, width, height)
        point_values[point.name] = float(lux_array[y_px, x_px])

    return point_values
