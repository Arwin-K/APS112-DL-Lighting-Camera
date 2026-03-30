"""Geometry helpers for normalized hallway points and floor-plane reasoning."""

from __future__ import annotations


def normalized_to_pixel_coordinates(
    x_normalized: float,
    y_normalized: float,
    width: int,
    height: int,
) -> tuple[int, int]:
    """Converts normalized [0, 1] coordinates to valid pixel indices."""

    x_pixel = int(round(x_normalized * (width - 1)))
    y_pixel = int(round(y_normalized * (height - 1)))
    x_pixel = max(0, min(width - 1, x_pixel))
    y_pixel = max(0, min(height - 1, y_pixel))
    return x_pixel, y_pixel


def planar_distance_meters(
    point_a: tuple[float, float],
    point_b: tuple[float, float],
    meters_per_unit: float = 1.0,
) -> float:
    """Computes planar distance between two 2D points."""

    dx = point_a[0] - point_b[0]
    dy = point_a[1] - point_b[1]
    return ((dx**2 + dy**2) ** 0.5) * meters_per_unit
