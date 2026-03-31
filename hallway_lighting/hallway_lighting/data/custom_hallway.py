"""Custom hallway dataset adapter."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pandas as pd

from .manifests import (
    DEFAULT_CUSTOM_HALLWAY_COLUMNS,
    NORMALIZED_MANIFEST_COLUMNS,
    create_manifest_dataframe,
    load_manifest,
    make_manifest_row,
    resolve_optional_path,
    save_manifest,
    validate_manifest_columns,
)


def load_point_target_values(path: str | Path) -> dict[str, float]:
    """Loads point-wise lux labels from a JSON file.

    Supported format:
        {
          "under_fixture_1": 0.0,
          "under_fixture_2": 0.0,
          "between_fixture_1_2": 0.0
        }
    """

    with Path(path).open("r", encoding="utf-8") as handle:
        payload = json.load(handle)

    if not isinstance(payload, dict):
        raise ValueError(f"Point target JSON must be a top-level object: {path}")

    point_values: dict[str, float] = {}
    for key, value in payload.items():
        if not isinstance(key, str) or not key.strip():
            raise ValueError(f"Point target JSON contains an invalid key in {path}")
        if not isinstance(value, (int, float)):
            raise ValueError(f"Point target '{key}' in {path} must be numeric.")
        point_values[key] = float(value)

    if not point_values:
        raise ValueError(f"Point target JSON is empty: {path}")

    return point_values


def _find_manifest_csv(dataset_root: Path) -> Path:
    """Finds the custom hallway manifest CSV inside a dataset directory."""

    candidates = sorted(dataset_root.glob("*.csv"))
    if not candidates:
        raise FileNotFoundError(
            f"No custom hallway manifest CSV found in {dataset_root}. "
            "Provide a CSV directly or place one in the dataset directory."
        )

    preferred_names = [
        "custom_hallway_manifest.csv",
        "hallway_manifest.csv",
        "manifest.csv",
    ]
    for name in preferred_names:
        for candidate in candidates:
            if candidate.name == name:
                return candidate

    if len(candidates) > 1:
        raise ValueError(
            f"Multiple CSV files found in {dataset_root}. "
            "Rename the intended file to custom_hallway_manifest.csv or pass the CSV directly."
        )
    return candidates[0]


def _coerce_numeric(value: Any) -> float | None:
    """Converts optional numeric CSV values to floats."""

    if value is None:
        return None
    if isinstance(value, float) and pd.isna(value):
        return None
    if isinstance(value, str) and not value.strip():
        return None
    return float(value)


def build_custom_hallway_manifest(
    dataset_root: str | Path,
    output_path: str | Path | None = None,
) -> pd.DataFrame:
    """Normalizes a user-provided custom hallway manifest CSV."""

    dataset_root = Path(dataset_root)
    manifest_csv = dataset_root if dataset_root.is_file() else _find_manifest_csv(dataset_root)
    manifest_base_dir = manifest_csv.parent
    manifest_df = pd.read_csv(manifest_csv)

    if "image_path" not in manifest_df.columns:
        raise ValueError("Custom hallway manifest must include an 'image_path' column.")

    rows: list[dict[str, Any]] = []
    for row_index, raw_row in manifest_df.iterrows():
        sample_id = raw_row.get("sample_id", f"custom_hallway_{row_index:06d}")
        image_path = resolve_optional_path(raw_row.get("image_path"), manifest_base_dir)
        if not image_path:
            raise ValueError(f"Row {row_index} is missing image_path.")
        if not Path(image_path).exists():
            raise FileNotFoundError(f"Custom hallway image does not exist: {image_path}")

        point_targets_json = raw_row.get("point_targets_json", raw_row.get("point_targets_path"))
        point_targets_json = resolve_optional_path(point_targets_json, manifest_base_dir)
        if point_targets_json:
            load_point_target_values(point_targets_json)

        normalized_row = make_manifest_row(
            dataset_name="custom_hallway",
            sample_id=str(sample_id),
            image_path=image_path,
            split=str(raw_row.get("split", "unspecified") or "unspecified"),
            floor_mask_path=resolve_optional_path(raw_row.get("floor_mask_path"), manifest_base_dir),
            lux_map_path=resolve_optional_path(raw_row.get("lux_map_path"), manifest_base_dir),
            avg_lux=_coerce_numeric(raw_row.get("avg_lux")),
            low_lux_p5=_coerce_numeric(raw_row.get("low_lux_p5")),
            high_lux_p95=_coerce_numeric(raw_row.get("high_lux_p95")),
            point_targets_json=point_targets_json,
            material_label=raw_row.get("material_label"),
            floor_finish_label=raw_row.get("floor_finish_label"),
            albedo_path=resolve_optional_path(raw_row.get("albedo_path"), manifest_base_dir),
            gloss_path=resolve_optional_path(raw_row.get("gloss_path"), manifest_base_dir),
            measured_power_w=_coerce_numeric(raw_row.get("measured_power_w")),
            interval_hours=_coerce_numeric(raw_row.get("interval_hours")),
            notes=raw_row.get("notes"),
        )
        rows.append(normalized_row)

    manifest = create_manifest_dataframe(rows)
    validate_manifest_columns(manifest, NORMALIZED_MANIFEST_COLUMNS)
    if output_path is not None:
        save_manifest(manifest, output_path)
    return manifest
