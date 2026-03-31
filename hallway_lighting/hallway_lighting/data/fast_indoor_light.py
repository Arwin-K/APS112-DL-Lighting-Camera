"""Fast Spatially-Varying Indoor Lighting Estimation dataset adapter."""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from .manifests import (
    IMAGE_EXTENSIONS,
    JSON_EXTENSIONS,
    TEXT_EXTENSIONS,
    create_manifest_dataframe,
    find_first_file,
    infer_split_from_path,
    iter_files,
    iter_scene_directories,
    make_manifest_row,
    save_manifest,
)


def build_fast_indoor_light_manifest(
    dataset_root: str | Path,
    output_path: str | Path | None = None,
) -> pd.DataFrame:
    """Builds a normalized manifest for the Fast Indoor Lighting dataset."""

    dataset_root = Path(dataset_root)
    rows = []
    seen_image_paths: set[Path] = set()

    for scene_dir in iter_scene_directories(dataset_root):
        image_candidates = [
            path
            for path in iter_files(scene_dir, extensions=IMAGE_EXTENSIONS, recursive=False)
            if any(keyword in path.stem.lower() for keyword in ("rgb", "image", "input", "ldr", "photo"))
        ]
        albedo_path = find_first_file(scene_dir, keywords=("albedo", "diffuse", "reflectance"), recursive=False)
        gloss_path = find_first_file(scene_dir, keywords=("gloss", "roughness", "specular"), recursive=False)
        metadata_path = find_first_file(
            scene_dir,
            keywords=("metadata", "lighting", "light", "params"),
            extensions=JSON_EXTENSIONS + TEXT_EXTENSIONS,
            recursive=False,
        )

        if not image_candidates:
            continue

        for image_path in image_candidates:
            if image_path in seen_image_paths:
                continue
            seen_image_paths.add(image_path)
            sample_id = image_path.relative_to(dataset_root).with_suffix("").as_posix().replace("/", "_")
            notes = "Official Fast Spatially-Varying Indoor Lighting sample."
            if metadata_path is not None:
                notes = f"{notes} Metadata: {metadata_path.name}"
            rows.append(
                make_manifest_row(
                    dataset_name="fast_sv_indoor_lighting",
                    sample_id=sample_id,
                    image_path=image_path.resolve(),
                    split=infer_split_from_path(image_path),
                    albedo_path=albedo_path.resolve() if albedo_path else "",
                    gloss_path=gloss_path.resolve() if gloss_path else "",
                    notes=notes,
                )
            )

    if not rows:
        raise ValueError(
            f"No Fast Indoor Lighting samples found under {dataset_root}. "
            "Expected scene folders containing RGB/input images from the official dataset."
        )

    manifest = create_manifest_dataframe(rows)
    if output_path is not None:
        save_manifest(manifest, output_path)
    return manifest
