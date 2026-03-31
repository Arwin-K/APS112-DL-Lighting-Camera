"""Geometry helpers for hallway coordinate systems and point layouts."""

from __future__ import annotations

from dataclasses import dataclass

import torch


@dataclass(frozen=True)
class HallwayGeometryConfig:
    """Defines a canonical hallway floor coordinate system in image space."""

    fixture_count: int = 3
    floor_y: float = 0.72
    start_x: float = 0.2
    end_x: float = 0.8

    def __post_init__(self) -> None:
        """Validates normalized coordinate ranges."""

        if self.fixture_count < 1:
            raise ValueError("fixture_count must be at least 1.")
        if not 0.0 <= self.floor_y <= 1.0:
            raise ValueError("floor_y must be in [0, 1].")
        if not 0.0 <= self.start_x < self.end_x <= 1.0:
            raise ValueError("Expected 0 <= start_x < end_x <= 1.")


def normalized_to_pixel_coordinates(
    x_normalized: float,
    y_normalized: float,
    width: int,
    height: int,
) -> tuple[int, int]:
    """Converts normalized [0, 1] coordinates to valid pixel indices."""

    if width <= 0 or height <= 0:
        raise ValueError("Image width and height must be positive.")

    x_pixel = int(round(x_normalized * (width - 1)))
    y_pixel = int(round(y_normalized * (height - 1)))
    x_pixel = max(0, min(width - 1, x_pixel))
    y_pixel = max(0, min(height - 1, y_pixel))
    return x_pixel, y_pixel


def normalized_to_grid_sample_coordinates(
    x_normalized: float,
    y_normalized: float,
) -> tuple[float, float]:
    """Converts [0, 1] coordinates to `grid_sample` coordinates in [-1, 1]."""

    return x_normalized * 2.0 - 1.0, y_normalized * 2.0 - 1.0


def build_coordinate_channels(
    height: int,
    width: int,
    device: torch.device | None = None,
    dtype: torch.dtype | None = None,
) -> torch.Tensor:
    """Builds normalized x/y coordinate channels with shape `[2, H, W]`."""

    if height <= 0 or width <= 0:
        raise ValueError("Coordinate channel dimensions must be positive.")

    y_values = torch.linspace(-1.0, 1.0, steps=height, device=device, dtype=dtype)
    x_values = torch.linspace(-1.0, 1.0, steps=width, device=device, dtype=dtype)
    y_grid, x_grid = torch.meshgrid(y_values, x_values, indexing="ij")
    return torch.stack([x_grid, y_grid], dim=0)


def expand_coordinate_channels(
    batch_size: int,
    height: int,
    width: int,
    device: torch.device | None = None,
    dtype: torch.dtype | None = None,
) -> torch.Tensor:
    """Repeats coordinate channels for a batch and returns `[B, 2, H, W]`."""

    coordinates = build_coordinate_channels(height, width, device=device, dtype=dtype)
    return coordinates.unsqueeze(0).repeat(batch_size, 1, 1, 1)


def canonical_fixture_positions(
    fixture_count: int,
    start_x: float = 0.2,
    end_x: float = 0.8,
    floor_y: float = 0.72,
) -> list[tuple[float, float]]:
    """Returns normalized coordinates directly under each hallway fixture."""

    if fixture_count < 1:
        raise ValueError("fixture_count must be at least 1.")
    if fixture_count == 1:
        return [((start_x + end_x) / 2.0, floor_y)]

    spacing = (end_x - start_x) / float(fixture_count - 1)
    return [(start_x + spacing * index, floor_y) for index in range(fixture_count)]


def canonical_between_fixture_positions(
    fixture_count: int,
    start_x: float = 0.2,
    end_x: float = 0.8,
    floor_y: float = 0.72,
) -> list[tuple[float, float]]:
    """Returns midpoints between adjacent fixtures along the hallway floor."""

    fixture_positions = canonical_fixture_positions(
        fixture_count=fixture_count,
        start_x=start_x,
        end_x=end_x,
        floor_y=floor_y,
    )
    between_positions: list[tuple[float, float]] = []
    for left, right in zip(fixture_positions[:-1], fixture_positions[1:], strict=False):
        between_positions.append(((left[0] + right[0]) / 2.0, floor_y))
    return between_positions


def planar_distance_meters(
    point_a: tuple[float, float],
    point_b: tuple[float, float],
    meters_per_unit: float = 1.0,
) -> float:
    """Computes planar distance between two 2D points."""

    dx = point_a[0] - point_b[0]
    dy = point_a[1] - point_b[1]
    return ((dx**2 + dy**2) ** 0.5) * meters_per_unit
