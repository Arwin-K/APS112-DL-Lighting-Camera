"""MIT Intrinsic Images dataset adapter."""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from .manifests import (
    IMAGE_EXTENSIONS,
    create_manifest_dataframe,
    find_first_file,
    infer_split_from_path,
    iter_scene_directories,
    make_manifest_row,
    save_manifest,
)


def build_mit_intrinsic_manifest(
    dataset_root: str | Path,
    output_path: str | Path | None = None,
) -> pd.DataFrame:
    """Builds a normalized manifest for MIT Intrinsic Images."""

    dataset_root = Path(dataset_root)
    rows = []
    seen_image_paths: set[Path] = set()

    for scene_dir in iter_scene_directories(dataset_root):
        image_path = find_first_file(scene_dir, keywords=("original", "input", "image", "rgb"), recursive=False)
        reflectance_path = find_first_file(scene_dir, keywords=("reflectance", "albedo"), recursive=False)
        shading_path = find_first_file(scene_dir, keywords=("shading",), recursive=False)

        if image_path is None:
            continue
        if reflectance_path is None and shading_path is None:
            continue
        if image_path in seen_image_paths:
            continue

        seen_image_paths.add(image_path)
        sample_id = scene_dir.relative_to(dataset_root).as_posix().replace("/", "_") or image_path.stem
        rows.append(
            make_manifest_row(
                dataset_name="mit_intrinsic_images",
                sample_id=sample_id,
                image_path=image_path.resolve(),
                split=infer_split_from_path(image_path),
                albedo_path=reflectance_path.resolve() if reflectance_path else "",
                reflectance_path=reflectance_path.resolve() if reflectance_path else "",
                shading_path=shading_path.resolve() if shading_path else "",
                notes="Official MIT Intrinsic Images sample.",
            )
        )

    if not rows:
        raise ValueError(
            f"No MIT Intrinsic Images samples found under {dataset_root}. "
            "Expected scene folders with an RGB/original image and reflectance or shading target files."
        )

    manifest = create_manifest_dataframe(rows)
    if output_path is not None:
        save_manifest(manifest, output_path)
    return manifest
