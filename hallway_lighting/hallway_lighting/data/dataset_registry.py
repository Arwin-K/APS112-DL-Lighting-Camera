"""Registry and high-level orchestration for supported datasets."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import pandas as pd

from .archive_utils import PreparedDatasetInput, prepare_dataset_input
from .custom_hallway import build_custom_hallway_manifest
from .fast_indoor_light import build_fast_indoor_light_manifest
from .manifests import NORMALIZED_MANIFEST_COLUMNS, save_manifest
from .mid_intrinsics import build_mid_intrinsics_manifest
from .mit_intrinsic import build_mit_intrinsic_manifest
from .nyu_depth_v2 import build_nyu_depth_v2_manifest


@dataclass(frozen=True)
class DatasetSpec:
    """Describes a dataset, accepted inputs, and real labels."""

    name: str
    description: str
    tasks: tuple[str, ...]
    accepted_inputs: tuple[str, ...]
    real_labels: tuple[str, ...]
    optional_labels: tuple[str, ...]
    required_manifest_columns: tuple[str, ...]
    is_public_auxiliary: bool = True
    is_optional: bool = False


@dataclass
class DatasetPreparationResult:
    """Bundle returned after preparing a dataset input and manifest."""

    spec: DatasetSpec
    prepared_input: PreparedDatasetInput
    manifest: pd.DataFrame
    manifest_path: Path


DATASET_REGISTRY: dict[str, DatasetSpec] = {
    "nyu_depth_v2": DatasetSpec(
        name="nyu_depth_v2",
        description="RGB-D indoor dataset used for geometric and depth priors.",
        tasks=("depth_prior", "indoor_geometry"),
        accepted_inputs=("directory", "mat", "zip", "tar", "tar.gz", "tgz"),
        real_labels=("image_path", "depth_path"),
        optional_labels=("notes",),
        required_manifest_columns=("dataset_name", "sample_id", "split", "image_path", "depth_path"),
    ),
    "mit_intrinsic_images": DatasetSpec(
        name="mit_intrinsic_images",
        description="Intrinsic decomposition dataset with reflectance and shading targets.",
        tasks=("reflectance", "shading", "intrinsics"),
        accepted_inputs=("directory", "zip", "tar", "tar.gz", "tgz"),
        real_labels=("image_path", "reflectance_path", "shading_path"),
        optional_labels=("albedo_path", "notes"),
        required_manifest_columns=("dataset_name", "sample_id", "split", "image_path"),
    ),
    "mid_intrinsics": DatasetSpec(
        name="mid_intrinsics",
        description="Intrinsic dataset with image, albedo/reflectance, and shading supervision.",
        tasks=("reflectance", "shading", "intrinsics"),
        accepted_inputs=("directory", "zip", "tar", "tar.gz", "tgz"),
        real_labels=("image_path", "albedo_path", "shading_path"),
        optional_labels=("gloss_path", "notes"),
        required_manifest_columns=("dataset_name", "sample_id", "split", "image_path"),
    ),
    "fast_sv_indoor_lighting": DatasetSpec(
        name="fast_sv_indoor_lighting",
        description="Indoor lighting estimation dataset used for appearance and lighting priors.",
        tasks=("lighting", "appearance"),
        accepted_inputs=("directory", "zip", "tar", "tar.gz", "tgz"),
        real_labels=("image_path",),
        optional_labels=("albedo_path", "gloss_path", "notes"),
        required_manifest_columns=("dataset_name", "sample_id", "split", "image_path"),
    ),
    "custom_hallway": DatasetSpec(
        name="custom_hallway",
        description="Optional hallway-specific data with user-provided lux, floor, point, and power labels.",
        tasks=("lux_map", "point_lux", "power", "carbon"),
        accepted_inputs=("directory", "csv"),
        real_labels=("image_path",),
        optional_labels=(
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
        ),
        required_manifest_columns=("dataset_name", "sample_id", "split", "image_path"),
        is_public_auxiliary=False,
        is_optional=True,
    ),
}

DATASET_BUILDERS = {
    "nyu_depth_v2": build_nyu_depth_v2_manifest,
    "mit_intrinsic_images": build_mit_intrinsic_manifest,
    "mid_intrinsics": build_mid_intrinsics_manifest,
    "fast_sv_indoor_lighting": build_fast_indoor_light_manifest,
    "custom_hallway": build_custom_hallway_manifest,
}


def list_supported_datasets() -> list[str]:
    """Returns the supported dataset keys in a stable order."""

    return list(DATASET_REGISTRY.keys())


def get_dataset_spec(name: str) -> DatasetSpec:
    """Fetches a dataset specification by key."""

    if name not in DATASET_REGISTRY:
        supported = ", ".join(list_supported_datasets())
        raise KeyError(f"Unsupported dataset '{name}'. Supported datasets: {supported}")
    return DATASET_REGISTRY[name]


def validate_enabled_datasets(enabled: list[str]) -> list[DatasetSpec]:
    """Validates a config-provided dataset list and returns the matching specs."""

    return [get_dataset_spec(name) for name in enabled]


def build_dataset_manifest(
    dataset_name: str,
    dataset_input_path: str | Path,
    working_dir: str | Path,
    manifest_output_path: str | Path | None = None,
    overwrite: bool = False,
) -> DatasetPreparationResult:
    """Prepares a dataset input and builds its normalized manifest."""

    spec = get_dataset_spec(dataset_name)
    prepared_input = prepare_dataset_input(
        input_path=dataset_input_path,
        dataset_name=dataset_name,
        working_dir=working_dir,
        overwrite=overwrite,
    )
    if prepared_input.input_type not in spec.accepted_inputs:
        raise ValueError(
            f"Dataset '{dataset_name}' does not accept input type '{prepared_input.input_type}'. "
            f"Accepted inputs: {spec.accepted_inputs}"
        )

    builder = DATASET_BUILDERS[dataset_name]
    builder_input = prepared_input.primary_file if prepared_input.input_type == "csv" else prepared_input.prepared_root
    manifest = builder(builder_input)

    if manifest_output_path is None:
        manifest_output_path = Path(working_dir) / "manifests" / f"{dataset_name}.csv"
    manifest_path = save_manifest(manifest, manifest_output_path)

    missing = [column for column in spec.required_manifest_columns if column not in manifest.columns]
    if missing:
        raise ValueError(f"Manifest for '{dataset_name}' is missing required columns: {missing}")
    if any(column not in manifest.columns for column in NORMALIZED_MANIFEST_COLUMNS):
        raise ValueError(f"Manifest for '{dataset_name}' does not follow the normalized schema.")

    return DatasetPreparationResult(
        spec=spec,
        prepared_input=prepared_input,
        manifest=manifest,
        manifest_path=manifest_path,
    )


def build_all_dataset_manifests(
    dataset_inputs: dict[str, str | Path],
    working_dir: str | Path,
    output_dir: str | Path | None = None,
    overwrite: bool = False,
) -> dict[str, DatasetPreparationResult]:
    """Builds manifests for every dataset path supplied by the notebook."""

    working_dir = Path(working_dir)
    output_dir = working_dir / "manifests" if output_dir is None else Path(output_dir)
    results: dict[str, DatasetPreparationResult] = {}

    for dataset_name in list_supported_datasets():
        raw_input = dataset_inputs.get(dataset_name, "")
        if raw_input in ("", None):
            continue
        results[dataset_name] = build_dataset_manifest(
            dataset_name=dataset_name,
            dataset_input_path=raw_input,
            working_dir=working_dir,
            manifest_output_path=output_dir / f"{dataset_name}.csv",
            overwrite=overwrite,
        )

    return results
