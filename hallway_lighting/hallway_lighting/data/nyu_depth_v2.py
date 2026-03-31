"""NYU Depth V2 dataset adapter."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from PIL import Image

from .manifests import (
    ARRAY_EXTENSIONS,
    IMAGE_EXTENSIONS,
    create_manifest_dataframe,
    load_text_split_assignments,
    make_manifest_row,
    save_manifest,
)


def _find_nyu_mat_file(dataset_root: Path) -> Path | None:
    """Finds a NYU Depth V2 `.mat` file under the prepared root."""

    candidates = sorted(dataset_root.rglob("*.mat"))
    if not candidates:
        return None
    preferred = [candidate for candidate in candidates if "nyu" in candidate.name.lower()]
    return preferred[0] if preferred else candidates[0]


def _find_subdirectory(root: Path, names: tuple[str, ...]) -> Path | None:
    """Finds a subdirectory by case-insensitive name."""

    name_set = {name.lower() for name in names}
    for candidate in [root, *sorted(path for path in root.rglob("*") if path.is_dir())]:
        if candidate.name.lower() in name_set:
            return candidate
    return None


def _load_mat_with_scipy(mat_path: Path) -> dict[str, Any]:
    """Loads a MATLAB file with scipy when possible."""

    from scipy.io import loadmat

    return loadmat(mat_path)


def _image_layout(shape: tuple[int, ...]) -> str:
    """Infers the image tensor layout for NYU `.mat` content."""

    if len(shape) != 4:
        raise ValueError(f"Expected a 4D NYU image tensor, found shape {shape}")
    if shape[2] == 3 and shape[-1] > 3:
        return "hwcn"
    if shape[-1] == 3:
        return "nhwc"
    if shape[1] == 3:
        return "nchw"
    if shape[0] == 3:
        return "chwn"
    raise ValueError(f"Could not infer NYU image layout from shape {shape}")


def _depth_layout(shape: tuple[int, ...], sample_count: int) -> str:
    """Infers the depth tensor layout for NYU `.mat` content."""

    if len(shape) != 3:
        raise ValueError(f"Expected a 3D NYU depth tensor, found shape {shape}")
    if shape[0] == sample_count:
        return "nhw"
    if shape[-1] == sample_count:
        return "hwn"
    raise ValueError(f"Could not infer NYU depth layout from shape {shape}")


def _extract_image_sample(images: Any, index: int, layout: str) -> np.ndarray:
    """Extracts a single RGB frame from a NYU image tensor or dataset."""

    if layout == "hwcn":
        sample = np.asarray(images[..., index])
    elif layout == "nhwc":
        sample = np.asarray(images[index, ...])
    elif layout == "nchw":
        sample = np.asarray(images[index, ...]).transpose(1, 2, 0)
    elif layout == "chwn":
        sample = np.asarray(images[:, :, :, index]).transpose(1, 2, 0)
    else:
        raise ValueError(f"Unsupported NYU image layout: {layout}")

    if sample.dtype != np.uint8:
        sample = np.asarray(np.clip(sample, 0, 255), dtype=np.uint8)
    return sample


def _extract_depth_sample(depths: Any, index: int, layout: str) -> np.ndarray:
    """Extracts a single depth map from a NYU depth tensor or dataset."""

    if layout == "nhw":
        sample = np.asarray(depths[index, ...])
    elif layout == "hwn":
        sample = np.asarray(depths[..., index])
    else:
        raise ValueError(f"Unsupported NYU depth layout: {layout}")
    return np.asarray(sample, dtype=np.float32)


def _extract_nyu_mat(dataset_root: Path, mat_path: Path, overwrite: bool = False) -> Path:
    """Extracts RGB and depth samples from the official NYU `.mat` file."""

    output_root = dataset_root / "nyu_depth_v2_extracted"
    rgb_dir = output_root / "rgb"
    depth_dir = output_root / "depth"
    if rgb_dir.exists() and depth_dir.exists() and any(rgb_dir.iterdir()) and any(depth_dir.iterdir()) and not overwrite:
        return output_root

    output_root.mkdir(parents=True, exist_ok=True)
    rgb_dir.mkdir(parents=True, exist_ok=True)
    depth_dir.mkdir(parents=True, exist_ok=True)

    try:
        payload = _load_mat_with_scipy(mat_path)
        images = payload.get("images")
        depths = payload.get("rawDepths") if payload.get("rawDepths") is not None else payload.get("depths")
        if images is None or depths is None:
            raise ValueError("NYU `.mat` file must contain 'images' and 'depths' or 'rawDepths'.")

        image_layout = _image_layout(tuple(images.shape))
        sample_count = images.shape[-1] if image_layout in {"hwcn", "chwn"} else images.shape[0]
        depth_layout = _depth_layout(tuple(depths.shape), sample_count)
        for index in range(sample_count):
            image = _extract_image_sample(images, index, image_layout)
            depth = _extract_depth_sample(depths, index, depth_layout)
            sample_id = f"{index:06d}"
            Image.fromarray(image).save(rgb_dir / f"{sample_id}.png")
            np.save(depth_dir / f"{sample_id}.npy", depth)
        return output_root
    except NotImplementedError:
        pass

    import h5py

    with h5py.File(mat_path, "r") as handle:
        if "images" not in handle or ("depths" not in handle and "rawDepths" not in handle):
            raise ValueError("NYU `.mat` file must contain 'images' and 'depths' or 'rawDepths'.")
        images = handle["images"]
        depths = handle["rawDepths"] if "rawDepths" in handle else handle["depths"]
        image_layout = _image_layout(tuple(images.shape))
        sample_count = images.shape[-1] if image_layout in {"hwcn", "chwn"} else images.shape[0]
        depth_layout = _depth_layout(tuple(depths.shape), sample_count)

        for index in range(sample_count):
            image = _extract_image_sample(images, index, image_layout)
            depth = _extract_depth_sample(depths, index, depth_layout)
            sample_id = f"{index:06d}"
            Image.fromarray(image).save(rgb_dir / f"{sample_id}.png")
            np.save(depth_dir / f"{sample_id}.npy", depth)

    return output_root


def build_nyu_depth_v2_manifest(
    dataset_root: str | Path,
    output_path: str | Path | None = None,
) -> pd.DataFrame:
    """Builds a normalized manifest for NYU Depth V2."""

    dataset_root = Path(dataset_root)
    mat_path = _find_nyu_mat_file(dataset_root)

    rgb_dir = _find_subdirectory(dataset_root, ("rgb", "images"))
    depth_dir = _find_subdirectory(dataset_root, ("depth", "depths", "rawdepths"))

    if mat_path is not None and (rgb_dir is None or depth_dir is None):
        extracted_root = _extract_nyu_mat(dataset_root, mat_path)
        rgb_dir = extracted_root / "rgb"
        depth_dir = extracted_root / "depth"

    if rgb_dir is None or depth_dir is None:
        raise FileNotFoundError(
            f"Could not locate NYU Depth V2 RGB and depth content under {dataset_root}. "
            "Provide the official `.mat` file or an extracted folder with rgb/ and depth/ content."
        )

    rgb_files = sorted(path for path in rgb_dir.rglob("*") if path.suffix.lower() in IMAGE_EXTENSIONS)
    depth_files = sorted(path for path in depth_dir.rglob("*") if path.suffix.lower() in IMAGE_EXTENSIONS + ARRAY_EXTENSIONS)
    depth_by_stem = {path.stem: path for path in depth_files}
    if not rgb_files:
        raise ValueError(f"No NYU RGB images found under {rgb_dir}")
    if not depth_by_stem:
        raise ValueError(f"No NYU depth files found under {depth_dir}")

    split_assignments = load_text_split_assignments(dataset_root)
    rows = []
    for image_path in rgb_files:
        depth_path = depth_by_stem.get(image_path.stem)
        if depth_path is None:
            continue
        rows.append(
            make_manifest_row(
                dataset_name="nyu_depth_v2",
                sample_id=image_path.stem,
                image_path=image_path.resolve(),
                split=split_assignments.get(image_path.stem, "unspecified"),
                depth_path=depth_path.resolve(),
                notes="Official NYU Depth V2 RGB-D sample.",
            )
        )

    if not rows:
        raise ValueError("NYU Depth V2 manifest generation found no matched RGB-depth pairs.")

    manifest = create_manifest_dataframe(rows)
    if output_path is not None:
        save_manifest(manifest, output_path)
    return manifest
