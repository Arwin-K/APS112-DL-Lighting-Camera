"""Automatic lighting-fixture detection and between-region analysis helpers."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any, Sequence

import numpy as np

@dataclass(frozen=True)
class FixtureDetection:
    """Represents one detected ceiling fixture in normalized image coordinates."""

    name: str
    x: float
    y: float
    confidence: float
    bbox: tuple[float, float, float, float]


@dataclass(frozen=True)
class BetweenFixtureRegion:
    """Represents a floor-plane region between two adjacent detected fixtures."""

    name: str
    left_fixture: str
    right_fixture: str
    polygon: list[tuple[float, float]]
    area_pixels: int
    area_ratio: float
    estimated_area_m2: float | None


@dataclass(frozen=True)
class FixturePointTarget:
    """Point target compatible with the inference sampler interface."""

    name: str
    x: float
    y: float
    group: str


@dataclass(frozen=True)
class FixtureLayout:
    """Structured output for automatic fixture detection and floor projection."""

    source: str
    fallback_used: bool
    search_region_bottom_y: float
    floor_reference_y: float
    fixtures: list[FixtureDetection]
    point_targets: list[FixturePointTarget]
    between_regions: list[BetweenFixtureRegion]

    def to_summary_dict(self) -> dict[str, Any]:
        """Returns a JSON-serializable representation."""

        return {
            "source": self.source,
            "fallback_used": self.fallback_used,
            "inferred_fixture_count": len(self.fixtures),
            "search_region_bottom_y": self.search_region_bottom_y,
            "floor_reference_y": self.floor_reference_y,
            "fixtures": [asdict(fixture) for fixture in self.fixtures],
            "point_targets": [asdict(point) for point in self.point_targets],
            "between_regions": [asdict(region) for region in self.between_regions],
        }


def _ensure_rgb_float(image: np.ndarray) -> np.ndarray:
    """Normalizes an image array to HxWx3 float space in [0, 1]."""

    image_np = np.asarray(image)
    if image_np.ndim != 3:
        raise ValueError(f"Expected an HxWxC image, got shape {tuple(image_np.shape)}")
    if image_np.shape[-1] == 1:
        image_np = np.repeat(image_np, 3, axis=-1)
    if image_np.shape[-1] != 3:
        raise ValueError(f"Expected 3 image channels, got shape {tuple(image_np.shape)}")

    image_np = image_np.astype(np.float32)
    if image_np.max(initial=0.0) > 1.0 or image_np.min(initial=0.0) < 0.0:
        image_np = image_np / 255.0
    return np.clip(image_np, 0.0, 1.0)


def _normalize_score_map(values: np.ndarray) -> np.ndarray:
    """Applies robust percentile normalization to a score map."""

    values = np.asarray(values, dtype=np.float32)
    low = float(np.percentile(values, 2))
    high = float(np.percentile(values, 98))
    if high <= low + 1e-6:
        return np.clip(values, 0.0, 1.0)
    return np.clip((values - low) / (high - low), 0.0, 1.0)


def _resolve_floor_mask(floor_mask: np.ndarray | None, height: int, width: int) -> np.ndarray | None:
    """Normalizes an optional floor mask to a 2D boolean array."""

    if floor_mask is None:
        return None

    floor_mask_np = np.asarray(floor_mask)
    if floor_mask_np.ndim == 3:
        if floor_mask_np.shape[0] == 1:
            floor_mask_np = floor_mask_np[0]
        elif floor_mask_np.shape[-1] == 1:
            floor_mask_np = floor_mask_np[..., 0]
        else:
            floor_mask_np = floor_mask_np[..., 0]
    if floor_mask_np.ndim != 2:
        raise ValueError(f"Expected a 2D floor mask, got shape {tuple(floor_mask_np.shape)}")
    if floor_mask_np.shape != (height, width):
        raise ValueError(
            f"Expected floor mask shape {(height, width)}, got {tuple(floor_mask_np.shape)}"
        )
    return floor_mask_np.astype(np.float32) > 0.5


def _estimate_search_region_bottom(floor_mask: np.ndarray | None, height: int) -> int:
    """Chooses the lower boundary of the ceiling search region."""

    default_bottom = max(1, min(height - 1, int(round(height * 0.68))))
    if floor_mask is None or float(floor_mask.mean()) <= 0.01:
        return default_bottom

    floor_rows = np.where(floor_mask.any(axis=1))[0]
    if floor_rows.size == 0:
        return default_bottom

    floor_top = int(floor_rows[0])
    padded_bottom = floor_top - max(4, int(round(height * 0.04)))
    return max(1, min(default_bottom, padded_bottom))


def _gaussian_kernel1d(sigma: float) -> np.ndarray:
    """Builds a normalized 1D Gaussian kernel."""

    sigma = max(float(sigma), 0.5)
    radius = max(1, int(round(sigma * 3.0)))
    positions = np.arange(-radius, radius + 1, dtype=np.float32)
    kernel = np.exp(-(positions**2) / (2.0 * sigma**2))
    kernel_sum = float(kernel.sum())
    if kernel_sum <= 1e-8:
        return np.array([1.0], dtype=np.float32)
    return kernel / kernel_sum


def _smooth_profile(profile: np.ndarray, sigma: float) -> np.ndarray:
    """Smooths a 1D response profile."""

    kernel = _gaussian_kernel1d(sigma)
    return np.convolve(profile.astype(np.float32), kernel, mode="same")


def _build_fixture_score_map(image: np.ndarray) -> np.ndarray:
    """Builds a heuristic score map for likely ceiling fixtures."""

    rgb = _ensure_rgb_float(image)
    luminance = 0.2126 * rgb[..., 0] + 0.7152 * rgb[..., 1] + 0.0722 * rgb[..., 2]
    saturation = np.max(rgb, axis=-1) - np.min(rgb, axis=-1)

    horizontal_detail = np.abs(rgb - np.roll(rgb, 1, axis=1)).mean(axis=-1)
    vertical_detail = np.abs(rgb - np.roll(rgb, 1, axis=0)).mean(axis=-1)
    local_detail = np.clip(horizontal_detail + vertical_detail, 0.0, 1.0)

    score = 0.70 * luminance + 0.20 * local_detail - 0.18 * saturation
    return _normalize_score_map(score)


def _build_column_profile(score_map: np.ndarray, search_bottom: int) -> np.ndarray:
    """Builds a smoothed horizontal response profile over the search region."""

    ceiling_region = score_map[:search_bottom, :]
    if ceiling_region.size == 0:
        return np.zeros(score_map.shape[1], dtype=np.float32)

    profile = np.max(ceiling_region, axis=0)
    sigma = max(1.0, score_map.shape[1] * 0.012)
    return _smooth_profile(profile=profile, sigma=sigma)


def _detect_profile_peaks(
    profile: np.ndarray,
    min_horizontal_gap_px: int,
    max_fixture_count: int,
) -> tuple[list[int], bool]:
    """Returns horizontal peak positions from the fixture profile."""

    if profile.size == 0 or float(profile.max(initial=0.0)) <= 1e-6:
        return [], False

    min_height = max(0.14, float(np.percentile(profile, 88)))
    candidate_indices: list[int] = []
    for index in range(profile.size):
        left = profile[index - 1] if index > 0 else profile[index]
        right = profile[index + 1] if index + 1 < profile.size else profile[index]
        if profile[index] >= min_height and profile[index] >= left and profile[index] >= right:
            candidate_indices.append(index)

    selected: list[int] = []
    for index in sorted(candidate_indices, key=lambda item: float(profile[item]), reverse=True):
        if all(abs(index - existing) >= max(1, min_horizontal_gap_px) for existing in selected):
            selected.append(index)
        if len(selected) >= max_fixture_count:
            break

    fallback_used = False
    if not selected:
        fallback_used = True
        suppressed_profile = profile.copy()
        for _ in range(max_fixture_count):
            peak_index = int(np.argmax(suppressed_profile))
            peak_value = float(suppressed_profile[peak_index])
            if peak_value < max(0.18, float(np.percentile(profile, 92))):
                break
            selected.append(peak_index)
            left = max(0, peak_index - min_horizontal_gap_px)
            right = min(profile.size, peak_index + min_horizontal_gap_px + 1)
            suppressed_profile[left:right] = 0.0
        selected.sort()

    return selected, fallback_used


def _estimate_floor_reference_y(
    floor_mask: np.ndarray | None,
    height: int,
) -> float:
    """Estimates a normalized floor sampling y-coordinate for point targets."""

    if floor_mask is None or float(floor_mask.mean()) <= 0.01:
        return 0.72

    floor_rows = np.where(floor_mask.any(axis=1))[0]
    if floor_rows.size == 0:
        return 0.72

    floor_top = float(floor_rows[0])
    floor_bottom = float(floor_rows[-1])
    sampling_y = floor_top + 0.18 * (floor_bottom - floor_top)
    return float(np.clip(sampling_y / max(1, height - 1), 0.0, 1.0))


def _local_peak_y(score_map: np.ndarray, search_bottom: int, peak_x: int, window_radius: int) -> int:
    """Finds the best vertical position for a fixture near a horizontal peak."""

    left = max(0, peak_x - window_radius)
    right = min(score_map.shape[1], peak_x + window_radius + 1)
    local_region = score_map[:search_bottom, left:right]
    if local_region.size == 0:
        return max(0, min(search_bottom - 1, int(round(search_bottom * 0.25))))
    y_index, _ = np.unravel_index(int(np.argmax(local_region)), local_region.shape)
    return int(y_index)


def _bbox_from_peak(
    score_map: np.ndarray,
    peak_x: int,
    peak_y: int,
    search_bottom: int,
    window_radius: int,
) -> tuple[float, float, float, float]:
    """Builds a rough bounding box around a detected fixture peak."""

    height, width = score_map.shape
    local_peak = float(score_map[peak_y, peak_x])
    threshold = max(0.45, local_peak * 0.65)

    min_col = peak_x
    max_col = peak_x
    while min_col > 0 and score_map[peak_y, min_col - 1] >= threshold:
        min_col -= 1
    while max_col + 1 < width and score_map[peak_y, max_col + 1] >= threshold:
        max_col += 1

    min_row = peak_y
    max_row = peak_y
    while min_row > 0 and score_map[min_row - 1, peak_x] >= threshold:
        min_row -= 1
    while max_row + 1 < search_bottom and score_map[max_row + 1, peak_x] >= threshold:
        max_row += 1

    if max_col == min_col:
        min_col = max(0, peak_x - window_radius // 2)
        max_col = min(width - 1, peak_x + window_radius // 2)
    if max_row == min_row:
        min_row = max(0, peak_y - window_radius // 3)
        max_row = min(search_bottom - 1, peak_y + window_radius // 3)

    return (
        float(np.clip(min_col / max(1, width - 1), 0.0, 1.0)),
        float(np.clip(min_row / max(1, height - 1), 0.0, 1.0)),
        float(np.clip(max_col / max(1, width - 1), 0.0, 1.0)),
        float(np.clip(max_row / max(1, height - 1), 0.0, 1.0)),
    )


def _project_points_to_floor(
    fixtures: Sequence[FixtureDetection],
    floor_reference_y: float,
) -> list[FixturePointTarget]:
    """Creates under- and between-fixture floor point targets."""

    point_targets: list[FixturePointTarget] = []
    for index, fixture in enumerate(fixtures, start=1):
        point_targets.append(
            FixturePointTarget(
                name=f"under_fixture_{index}",
                x=fixture.x,
                y=floor_reference_y,
                group="under_fixture",
            )
        )

    for index, (left, right) in enumerate(zip(fixtures[:-1], fixtures[1:]), start=1):
        point_targets.append(
            FixturePointTarget(
                name=f"between_fixture_{index}_{index + 1}",
                x=(left.x + right.x) / 2.0,
                y=floor_reference_y,
                group="between_fixture",
            )
        )
    return point_targets


def _build_between_regions(
    fixtures: Sequence[FixtureDetection],
    floor_mask: np.ndarray | None,
    floor_reference_y: float,
    floor_area_m2: float | None,
    height: int,
    width: int,
) -> list[BetweenFixtureRegion]:
    """Creates simple floor-plane regions between adjacent fixtures."""

    if len(fixtures) < 2:
        return []

    floor_reference_row = int(round(floor_reference_y * max(1, height - 1)))
    floor_pixels = int(floor_mask.sum()) if floor_mask is not None else 0

    regions: list[BetweenFixtureRegion] = []
    for index, (left, right) in enumerate(zip(fixtures[:-1], fixtures[1:]), start=1):
        left_x = int(round(left.x * max(1, width - 1)))
        right_x = int(round(right.x * max(1, width - 1)))
        if right_x < left_x:
            left_x, right_x = right_x, left_x

        region_mask = np.zeros((height, width), dtype=bool)
        region_mask[floor_reference_row:, left_x : right_x + 1] = True
        if floor_mask is not None:
            region_mask &= floor_mask

        area_pixels = int(region_mask.sum())
        area_ratio = float(area_pixels / float(height * width))
        estimated_area_m2 = None
        if floor_area_m2 is not None and floor_pixels > 0:
            estimated_area_m2 = float(floor_area_m2 * (area_pixels / float(floor_pixels)))

        polygon = [
            (float(np.clip(left.x, 0.0, 1.0)), floor_reference_y),
            (float(np.clip(right.x, 0.0, 1.0)), floor_reference_y),
            (float(np.clip(right.x, 0.0, 1.0)), 1.0),
            (float(np.clip(left.x, 0.0, 1.0)), 1.0),
        ]
        regions.append(
            BetweenFixtureRegion(
                name=f"between_fixture_{index}_{index + 1}",
                left_fixture=left.name,
                right_fixture=right.name,
                polygon=polygon,
                area_pixels=area_pixels,
                area_ratio=area_ratio,
                estimated_area_m2=estimated_area_m2,
            )
        )
    return regions


def infer_fixture_layout(
    image: np.ndarray,
    floor_mask: np.ndarray | None = None,
    min_fixture_count: int = 1,
    max_fixture_count: int = 8,
    min_horizontal_gap_ratio: float = 0.08,
    floor_area_m2: float | None = None,
) -> FixtureLayout | None:
    """Infers fixture count, fixture positions, and between-fixture regions from an image."""

    rgb = _ensure_rgb_float(image)
    height, width = rgb.shape[:2]
    floor_mask_np = _resolve_floor_mask(floor_mask, height=height, width=width)
    search_bottom = _estimate_search_region_bottom(floor_mask_np, height=height)
    if search_bottom <= 1:
        return None

    score_map = _build_fixture_score_map(rgb)
    score_map[search_bottom:, :] = 0.0
    profile = _build_column_profile(score_map, search_bottom=search_bottom)
    min_horizontal_gap_px = max(6, int(round(width * min_horizontal_gap_ratio)))
    peak_positions, fallback_used = _detect_profile_peaks(
        profile=profile,
        min_horizontal_gap_px=min_horizontal_gap_px,
        max_fixture_count=max_fixture_count,
    )
    if len(peak_positions) < min_fixture_count:
        return None

    floor_reference_y = _estimate_floor_reference_y(floor_mask_np, height=height)
    window_radius = max(3, min_horizontal_gap_px // 2)
    fixtures: list[FixtureDetection] = []
    for index, peak_x in enumerate(peak_positions, start=1):
        peak_y = _local_peak_y(
            score_map=score_map,
            search_bottom=search_bottom,
            peak_x=peak_x,
            window_radius=window_radius,
        )
        confidence = float(np.clip(profile[peak_x], 0.0, 1.0))
        fixtures.append(
            FixtureDetection(
                name=f"fixture_{index}",
                x=float(np.clip(peak_x / max(1, width - 1), 0.0, 1.0)),
                y=float(np.clip(peak_y / max(1, height - 1), 0.0, 1.0)),
                confidence=confidence,
                bbox=_bbox_from_peak(
                    score_map=score_map,
                    peak_x=peak_x,
                    peak_y=peak_y,
                    search_bottom=search_bottom,
                    window_radius=window_radius,
                ),
            )
        )

    point_targets = _project_points_to_floor(fixtures=fixtures, floor_reference_y=floor_reference_y)
    between_regions = _build_between_regions(
        fixtures=fixtures,
        floor_mask=floor_mask_np,
        floor_reference_y=floor_reference_y,
        floor_area_m2=floor_area_m2,
        height=height,
        width=width,
    )
    return FixtureLayout(
        source="automatic_fixture_detector",
        fallback_used=fallback_used,
        search_region_bottom_y=float(search_bottom / max(1, height - 1)),
        floor_reference_y=floor_reference_y,
        fixtures=fixtures,
        point_targets=point_targets,
        between_regions=between_regions,
    )
