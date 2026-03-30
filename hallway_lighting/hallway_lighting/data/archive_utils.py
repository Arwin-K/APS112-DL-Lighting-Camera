"""Archive extraction helpers for Colab dataset preparation."""

from __future__ import annotations

from pathlib import Path
import tarfile
import zipfile


SUPPORTED_ARCHIVE_SUFFIXES = (".zip", ".tar", ".gz", ".tgz", ".tar.gz")


def is_supported_archive(path: str | Path) -> bool:
    """Returns True when a path looks like an extractable dataset archive."""

    path_str = str(Path(path))
    return path_str.endswith(SUPPORTED_ARCHIVE_SUFFIXES)


def extract_archive(
    archive_path: str | Path,
    destination_dir: str | Path,
    overwrite: bool = False,
) -> Path:
    """Extracts a supported archive into the destination directory."""

    archive_path = Path(archive_path)
    destination_dir = Path(destination_dir)
    destination_dir.mkdir(parents=True, exist_ok=True)

    if any(destination_dir.iterdir()) and not overwrite:
        return destination_dir

    if archive_path.suffix == ".zip":
        with zipfile.ZipFile(archive_path, "r") as zip_handle:
            zip_handle.extractall(destination_dir)
        return destination_dir

    if archive_path.suffix in {".tar", ".gz"} or str(archive_path).endswith((".tgz", ".tar.gz")):
        with tarfile.open(archive_path, "r:*") as tar_handle:
            tar_handle.extractall(destination_dir)
        return destination_dir

    raise ValueError(f"Unsupported archive format: {archive_path}")


def prepare_dataset_archives(
    archive_dir: str | Path,
    extraction_root: str | Path,
    overwrite: bool = False,
) -> list[Path]:
    """Extracts every supported archive found in a directory."""

    archive_dir = Path(archive_dir)
    extraction_root = Path(extraction_root)
    extraction_root.mkdir(parents=True, exist_ok=True)
    extracted: list[Path] = []

    for archive_path in sorted(archive_dir.iterdir()):
        if not archive_path.is_file() or not is_supported_archive(archive_path):
            continue
        destination = extraction_root / archive_path.stem.replace(".tar", "")
        extracted.append(extract_archive(archive_path, destination, overwrite=overwrite))

    return extracted
