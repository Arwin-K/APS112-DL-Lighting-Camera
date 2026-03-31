"""Archive and input preparation helpers for notebook-driven datasets."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import shutil
import tarfile
import zipfile


SUPPORTED_INPUT_TYPES = {"directory", "zip", "tar", "tar.gz", "tgz", "mat", "csv"}


@dataclass(frozen=True)
class PreparedDatasetInput:
    """Describes a dataset input after normalization or extraction."""

    dataset_name: str
    source_path: Path
    input_type: str
    prepared_root: Path
    primary_file: Path | None = None


def detect_input_type(path: str | Path) -> str:
    """Detects whether an input is a directory, archive, or `.mat` file."""

    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Dataset input does not exist: {path}")
    if path.is_dir():
        return "directory"

    suffixes = [suffix.lower() for suffix in path.suffixes]
    if suffixes[-2:] == [".tar", ".gz"]:
        return "tar.gz"
    if path.suffix.lower() == ".tgz":
        return "tgz"
    if path.suffix.lower() == ".tar":
        return "tar"
    if path.suffix.lower() == ".zip":
        return "zip"
    if path.suffix.lower() == ".mat":
        return "mat"
    if path.suffix.lower() == ".csv":
        return "csv"

    raise ValueError(
        f"Unsupported dataset input type for '{path}'. "
        "Expected an extracted folder, .zip, .tar, .tar.gz, .tgz, or .mat file."
    )


def is_supported_archive(path: str | Path) -> bool:
    """Returns True when a path is one of the supported archive types."""

    return detect_input_type(path) in {"zip", "tar", "tar.gz", "tgz"}


def _normalized_destination_name(path: Path) -> str:
    """Builds a stable directory name for extracted content."""

    name = path.name
    for suffix in (".tar.gz", ".tgz", ".zip", ".tar", ".mat"):
        if name.endswith(suffix):
            return name[: -len(suffix)]
    return path.stem


def _ensure_within_directory(path: Path, directory: Path) -> None:
    """Guards against archive path traversal."""

    resolved_path = path.resolve()
    resolved_directory = directory.resolve()
    if resolved_directory not in resolved_path.parents and resolved_path != resolved_directory:
        raise ValueError(f"Unsafe archive member path detected: {resolved_path}")


def _safe_extract_zip(archive_path: Path, destination_dir: Path) -> None:
    """Safely extracts a zip archive."""

    with zipfile.ZipFile(archive_path, "r") as zip_handle:
        for member in zip_handle.infolist():
            candidate = destination_dir / member.filename
            _ensure_within_directory(candidate, destination_dir)
        zip_handle.extractall(destination_dir)


def _safe_extract_tar(archive_path: Path, destination_dir: Path) -> None:
    """Safely extracts a tar archive."""

    with tarfile.open(archive_path, "r:*") as tar_handle:
        for member in tar_handle.getmembers():
            candidate = destination_dir / member.name
            _ensure_within_directory(candidate, destination_dir)
        tar_handle.extractall(destination_dir)


def extract_archive(
    archive_path: str | Path,
    destination_dir: str | Path,
    overwrite: bool = False,
) -> Path:
    """Extracts a supported archive into a destination directory."""

    archive_path = Path(archive_path)
    destination_dir = Path(destination_dir)
    destination_dir.mkdir(parents=True, exist_ok=True)

    if any(destination_dir.iterdir()) and not overwrite:
        return destination_dir

    input_type = detect_input_type(archive_path)
    if input_type == "zip":
        _safe_extract_zip(archive_path, destination_dir)
        return destination_dir
    if input_type in {"tar", "tar.gz", "tgz"}:
        _safe_extract_tar(archive_path, destination_dir)
        return destination_dir

    raise ValueError(f"Cannot extract unsupported archive type '{input_type}' from {archive_path}")


def stage_mat_file(
    mat_path: str | Path,
    destination_dir: str | Path,
    overwrite: bool = False,
) -> Path:
    """Copies a `.mat` dataset file into a normalized working directory."""

    mat_path = Path(mat_path)
    destination_dir = Path(destination_dir)
    destination_dir.mkdir(parents=True, exist_ok=True)
    destination_file = destination_dir / mat_path.name

    if destination_file.exists() and not overwrite:
        return destination_dir

    shutil.copy2(mat_path, destination_file)
    return destination_dir


def prepare_dataset_input(
    input_path: str | Path,
    dataset_name: str,
    working_dir: str | Path,
    overwrite: bool = False,
) -> PreparedDatasetInput:
    """Normalizes a dataset input path into a prepared working root.

    This function is safe to call from the notebook with Google Drive paths.
    Directories are returned as-is, archives are extracted into `working_dir`,
    and `.mat` files are staged into a dedicated dataset folder.
    """

    input_path = Path(input_path).expanduser().resolve()
    working_dir = Path(working_dir).expanduser().resolve()
    working_dir.mkdir(parents=True, exist_ok=True)
    input_type = detect_input_type(input_path)

    if input_type == "directory":
        return PreparedDatasetInput(
            dataset_name=dataset_name,
            source_path=input_path,
            input_type=input_type,
            prepared_root=input_path,
            primary_file=None,
        )

    if input_type == "csv":
        return PreparedDatasetInput(
            dataset_name=dataset_name,
            source_path=input_path,
            input_type=input_type,
            prepared_root=input_path.parent,
            primary_file=input_path,
        )

    destination_dir = working_dir / dataset_name / _normalized_destination_name(input_path)
    if input_type == "mat":
        stage_mat_file(input_path, destination_dir, overwrite=overwrite)
        return PreparedDatasetInput(
            dataset_name=dataset_name,
            source_path=input_path,
            input_type=input_type,
            prepared_root=destination_dir,
            primary_file=destination_dir / input_path.name,
        )

    extract_archive(input_path, destination_dir, overwrite=overwrite)
    return PreparedDatasetInput(
        dataset_name=dataset_name,
        source_path=input_path,
        input_type=input_type,
        prepared_root=destination_dir,
        primary_file=None,
    )


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
        if not archive_path.is_file():
            continue
        try:
            input_type = detect_input_type(archive_path)
        except ValueError:
            continue
        if input_type not in {"zip", "tar", "tar.gz", "tgz"}:
            continue
        destination = extraction_root / _normalized_destination_name(archive_path)
        extracted.append(extract_archive(archive_path, destination, overwrite=overwrite))

    return extracted
