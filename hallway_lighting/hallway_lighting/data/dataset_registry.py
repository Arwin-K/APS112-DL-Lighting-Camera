"""Registry of public datasets used by the project."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class DatasetSpec:
    """Describes a dataset and the auxiliary roles it can play."""

    name: str
    description: str
    tasks: tuple[str, ...]
    required_manifest_columns: tuple[str, ...]
    is_public_auxiliary: bool = True
    is_optional: bool = False


DATASET_REGISTRY: dict[str, DatasetSpec] = {
    "nyu_depth_v2": DatasetSpec(
        name="nyu_depth_v2",
        description="RGB-D indoor dataset useful for geometric priors and scene layout.",
        tasks=("depth_prior", "indoor_geometry", "layout"),
        required_manifest_columns=("image_path", "split"),
    ),
    "mit_intrinsic_images": DatasetSpec(
        name="mit_intrinsic_images",
        description="Intrinsic decomposition dataset for reflectance and shading priors.",
        tasks=("reflectance", "shading", "intrinsics"),
        required_manifest_columns=("image_path", "split"),
    ),
    "mid_intrinsics": DatasetSpec(
        name="mid_intrinsics",
        description="Additional intrinsic image supervision for indoor appearance effects.",
        tasks=("reflectance", "shading", "intrinsics"),
        required_manifest_columns=("image_path", "split"),
    ),
    "fast_sv_indoor_lighting": DatasetSpec(
        name="fast_sv_indoor_lighting",
        description="Indoor lighting estimation dataset for illumination appearance priors.",
        tasks=("lighting", "illumination", "appearance"),
        required_manifest_columns=("image_path", "split"),
    ),
    "custom_hallway": DatasetSpec(
        name="custom_hallway",
        description="Optional hallway-specific dataset with lux, point, and power annotations.",
        tasks=("lux_map", "point_lux", "power", "carbon"),
        required_manifest_columns=("image_path", "split"),
        is_public_auxiliary=False,
        is_optional=True,
    ),
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
