"""Manifest helpers shared across all dataset adapters."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Iterable, Sequence

import pandas as pd


IMAGE_EXTENSIONS: tuple[str, ...] = (".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff", ".exr")
ARRAY_EXTENSIONS: tuple[str, ...] = (".npy", ".npz")
JSON_EXTENSIONS: tuple[str, ...] = (".json",)
TEXT_EXTENSIONS: tuple[str, ...] = (".txt", ".csv")

NORMALIZED_MANIFEST_COLUMNS: list[str] = [
    "sample_id",
    "dataset_name",
    "split",
    "image_path",
    "depth_path",
    "floor_mask_path",
    "lux_map_path",
    "avg_lux",
    "low_lux_p5",
    "high_lux_p95",
    "point_targets_json",
    "material_label",
    "floor_finish_label",
    "albedo_path",
    "reflectance_path",
    "shading_path",
    "gloss_path",
    "measured_power_w",
    "interval_hours",
    "notes",
]

DEFAULT_CUSTOM_HALLWAY_COLUMNS: list[str] = [
    "sample_id",
    "image_path",
    "split",
    "floor_mask_path",
    "lux_map_path",
    "avg_lux",
    "low_lux_p5",
    "high_lux_p95",
    "point_targets_json",
    "material_label",
    "floor_finish_label",
    "albedo_path",
    "gloss_path",
    "measured_power_w",
    "interval_hours",
    "notes",
]

_STRING_COLUMNS = {
    "sample_id",
    "dataset_name",
    "split",
    "image_path",
    "depth_path",
    "floor_mask_path",
    "lux_map_path",
    "point_targets_json",
    "material_label",
    "floor_finish_label",
    "albedo_path",
    "reflectance_path",
    "shading_path",
    "gloss_path",
    "notes",
}
_NUMERIC_COLUMNS = {
    "avg_lux",
    "low_lux_p5",
    "high_lux_p95",
    "measured_power_w",
    "interval_hours",
}


def load_manifest(path: str | Path) -> pd.DataFrame:
    """Loads a CSV manifest using the normalized schema."""

    manifest = pd.read_csv(Path(path))
    validate_manifest_columns(manifest, NORMALIZED_MANIFEST_COLUMNS)
    return manifest


def save_manifest(manifest: pd.DataFrame, path: str | Path) -> Path:
    """Writes a normalized manifest to disk."""

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    manifest.to_csv(path, index=False)
    return path


def validate_manifest_columns(
    manifest: pd.DataFrame,
    required_columns: Iterable[str],
) -> None:
    """Raises a helpful error if a manifest is missing required columns."""

    missing = [column for column in required_columns if column not in manifest.columns]
    if missing:
        raise ValueError(f"Manifest is missing required columns: {missing}")


def normalize_path_value(value: str | Path | None) -> str:
    """Converts optional path-like values to normalized strings."""

    if value is None:
        return ""
    try:
        if pd.isna(value):
            return ""
    except TypeError:
        pass
    if isinstance(value, str) and not value.strip():
        return ""
    return str(Path(value))


def normalize_float_value(value: Any) -> float | None:
    """Converts an optional numeric field to float."""

    if value is None:
        return None
    if isinstance(value, str) and not value.strip():
        return None
    return float(value)


def resolve_optional_path(value: Any, base_dir: str | Path) -> str:
    """Resolves an optional relative path against a manifest base directory."""

    if value is None:
        return ""
    if isinstance(value, float) and pd.isna(value):
        return ""
    path_text = str(value).strip()
    if not path_text:
        return ""
    path = Path(path_text)
    if not path.is_absolute():
        path = Path(base_dir) / path
    return str(path.resolve())


def make_manifest_row(
    dataset_name: str,
    sample_id: str,
    image_path: str | Path,
    split: str = "unspecified",
    **overrides: Any,
) -> dict[str, Any]:
    """Creates a normalized manifest row with empty defaults."""

    row: dict[str, Any] = {}
    for column in NORMALIZED_MANIFEST_COLUMNS:
        if column in _STRING_COLUMNS:
            row[column] = ""
        elif column in _NUMERIC_COLUMNS:
            row[column] = None

    row["dataset_name"] = dataset_name
    row["sample_id"] = sample_id
    row["split"] = split
    row["image_path"] = normalize_path_value(image_path)

    for key, value in overrides.items():
        if key not in NORMALIZED_MANIFEST_COLUMNS:
            continue
        if key in _STRING_COLUMNS:
            row[key] = normalize_path_value(value) if key.endswith("_path") or key.endswith("_json") else ("" if value is None else str(value))
        elif key in _NUMERIC_COLUMNS:
            row[key] = normalize_float_value(value)
        else:
            row[key] = value

    return row


def create_manifest_dataframe(rows: Sequence[dict[str, Any]]) -> pd.DataFrame:
    """Builds a DataFrame in the normalized column order."""

    manifest = pd.DataFrame(rows)
    if manifest.empty:
        return pd.DataFrame(columns=NORMALIZED_MANIFEST_COLUMNS)

    for column in NORMALIZED_MANIFEST_COLUMNS:
        if column not in manifest.columns:
            manifest[column] = "" if column in _STRING_COLUMNS else None

    return manifest[NORMALIZED_MANIFEST_COLUMNS]


def infer_split_from_path(path: str | Path) -> str:
    """Infers split names from a file path when directories encode them."""

    path = Path(path)
    for part in path.parts:
        lower = part.lower()
        if lower == "train":
            return "train"
        if lower in {"val", "valid", "validation"}:
            return "val"
        if lower == "test":
            return "test"
    return "unspecified"


def load_text_split_assignments(root: str | Path) -> dict[str, str]:
    """Loads simple `train.txt` / `val.txt` / `test.txt` split files if present."""

    root = Path(root)
    assignments: dict[str, str] = {}
    split_files = {
        "train": ("train.txt",),
        "val": ("val.txt", "valid.txt", "validation.txt"),
        "test": ("test.txt",),
    }

    for split_name, file_names in split_files.items():
        for file_name in file_names:
            for candidate in root.rglob(file_name):
                lines = candidate.read_text(encoding="utf-8").splitlines()
                for line in lines:
                    item = line.strip()
                    if not item or item.startswith("#"):
                        continue
                    assignments[Path(item).stem] = split_name
    return assignments


def iter_files(
    directory: str | Path,
    extensions: tuple[str, ...] = IMAGE_EXTENSIONS,
    recursive: bool = False,
) -> list[Path]:
    """Returns files under a directory filtered by extension."""

    directory = Path(directory)
    iterator = directory.rglob("*") if recursive else directory.iterdir()
    return sorted(
        path
        for path in iterator
        if path.is_file() and path.suffix.lower() in extensions
    )


def find_first_file(
    directory: str | Path,
    keywords: Sequence[str],
    extensions: tuple[str, ...] = IMAGE_EXTENSIONS,
    recursive: bool = False,
) -> Path | None:
    """Finds the first file whose lowercased stem contains any keyword."""

    keyword_set = tuple(keyword.lower() for keyword in keywords)
    for path in iter_files(directory, extensions=extensions, recursive=recursive):
        lower_name = path.stem.lower()
        if any(keyword in lower_name for keyword in keyword_set):
            return path
    return None


def iter_scene_directories(root: str | Path) -> list[Path]:
    """Returns every directory under a root, including the root itself."""

    root = Path(root)
    directories = [root]
    directories.extend(sorted(path for path in root.rglob("*") if path.is_dir()))
    return directories


def preview_manifest_rows(manifest: pd.DataFrame, limit: int = 5) -> pd.DataFrame:
    """Returns the first few rows for notebook display."""

    return manifest.head(limit)


def create_custom_manifest_template(path: str | Path) -> pd.DataFrame:
    """Creates a single-row custom hallway manifest template."""

    template = pd.DataFrame(
        [
            {
                "sample_id": "hallway_sample_0001",
                "image_path": "path/to/hallway_image.png",
                "split": "train",
                "floor_mask_path": "path/to/floor_mask.png",
                "lux_map_path": "path/to/lux_map.npy",
                "avg_lux": 150.0,
                "low_lux_p5": 95.0,
                "high_lux_p95": 230.0,
                "point_targets_json": "path/to/point_targets.json",
                "material_label": "painted_drywall",
                "floor_finish_label": "vinyl",
                "albedo_path": "path/to/albedo.png",
                "gloss_path": "path/to/gloss.png",
                "measured_power_w": 120.0,
                "interval_hours": 1.0,
                "notes": "replace_with_real_paths",
            }
        ],
        columns=DEFAULT_CUSTOM_HALLWAY_COLUMNS,
    )
    save_manifest(template, path)
    return template
