"""Manifest helpers for public and custom hallway datasets."""

from __future__ import annotations

from pathlib import Path
from typing import Iterable

import pandas as pd


DEFAULT_CUSTOM_HALLWAY_COLUMNS: list[str] = [
    "image_path",
    "split",
    "lux_map_path",
    "floor_mask_path",
    "reflectance_path",
    "shading_path",
    "point_targets_path",
    "measured_power_w",
    "interval_hours",
    "notes",
]


def load_manifest(path: str | Path) -> pd.DataFrame:
    """Loads a CSV manifest into a DataFrame."""

    return pd.read_csv(Path(path))


def validate_manifest_columns(
    manifest: pd.DataFrame,
    required_columns: Iterable[str],
) -> None:
    """Raises a helpful error if a manifest is missing columns."""

    missing = [column for column in required_columns if column not in manifest.columns]
    if missing:
        raise ValueError(f"Manifest is missing required columns: {missing}")


def create_custom_manifest_template(path: str | Path) -> pd.DataFrame:
    """Creates a single-row custom hallway manifest template."""

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    template = pd.DataFrame(
        [
            {
                "image_path": "path/to/hallway_image.png",
                "split": "train",
                "lux_map_path": "path/to/lux_map.npy",
                "floor_mask_path": "path/to/floor_mask.png",
                "reflectance_path": "path/to/reflectance.png",
                "shading_path": "path/to/shading.png",
                "point_targets_path": "path/to/point_targets.json",
                "measured_power_w": 0.0,
                "interval_hours": 1.0,
                "notes": "replace_with_real_paths",
            }
        ],
        columns=DEFAULT_CUSTOM_HALLWAY_COLUMNS,
    )
    template.to_csv(path, index=False)
    return template


def build_image_only_manifest(
    image_dir: str | Path,
    output_path: str | Path,
    split: str = "train",
    extensions: tuple[str, ...] = (".png", ".jpg", ".jpeg"),
) -> pd.DataFrame:
    """Builds a simple image manifest for datasets that start from RGB only.

    TODO:
        Replace this helper with dataset-specific manifest builders once exact
        folder structures for each public dataset are locked down.
    """

    image_dir = Path(image_dir)
    output_path = Path(output_path)
    rows: list[dict[str, str]] = []

    for image_path in sorted(image_dir.rglob("*")):
        if image_path.suffix.lower() not in extensions:
            continue
        rows.append({"image_path": str(image_path), "split": split})

    manifest = pd.DataFrame(rows, columns=["image_path", "split"])
    output_path.parent.mkdir(parents=True, exist_ok=True)
    manifest.to_csv(output_path, index=False)
    return manifest
