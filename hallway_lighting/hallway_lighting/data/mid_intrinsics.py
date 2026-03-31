"""MID Intrinsics dataset adapter."""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from .manifests import (
    create_manifest_dataframe,
    find_first_file,
    infer_split_from_path,
    iter_scene_directories,
    make_manifest_row,
    save_manifest,
)


def build_mid_intrinsics_manifest(
    dataset_root: str | Path,
    output_path: str | Path | None = None,
) -> pd.DataFrame:
    """Builds a normalized manifest for MID Intrinsics."""

    dataset_root = Path(dataset_root)
    rows = []
    seen_image_paths: set[Path] = set()

    for scene_dir in iter_scene_directories(dataset_root):
        image_path = find_first_file(scene_dir, keywords=("input", "rgb", "image", "img"), recursive=False)
        albedo_path = find_first_file(scene_dir, keywords=("albedo", "reflectance"), recursive=False)
        shading_path = find_first_file(scene_dir, keywords=("shading",), recursive=False)
        gloss_path = find_first_file(scene_dir, keywords=("gloss", "specular", "roughness"), recursive=False)

        if image_path is None:
            continue
        if albedo_path is None and shading_path is None and gloss_path is None:
            continue
        if image_path in seen_image_paths:
            continue

        seen_image_paths.add(image_path)
        sample_id = scene_dir.relative_to(dataset_root).as_posix().replace("/", "_") or image_path.stem
        rows.append(
            make_manifest_row(
                dataset_name="mid_intrinsics",
                sample_id=sample_id,
                image_path=image_path.resolve(),
                split=infer_split_from_path(image_path),
                albedo_path=albedo_path.resolve() if albedo_path else "",
                reflectance_path=albedo_path.resolve() if albedo_path else "",
                shading_path=shading_path.resolve() if shading_path else "",
                gloss_path=gloss_path.resolve() if gloss_path else "",
                notes="Official MID Intrinsics sample.",
            )
        )

    if not rows:
        raise ValueError(
            f"No MID Intrinsics samples found under {dataset_root}. "
            "Expected scene folders with an input image and intrinsic target files."
        )

    manifest = create_manifest_dataframe(rows)
    if output_path is not None:
        save_manifest(manifest, output_path)
    return manifest
