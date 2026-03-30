"""I/O helpers for configs, checkpoints, and simple artifact saving."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import torch
import yaml


def ensure_dir(path: str | Path) -> Path:
    """Creates a directory if needed and returns it as a Path."""

    path = Path(path)
    path.mkdir(parents=True, exist_ok=True)
    return path


def load_yaml(path: str | Path) -> dict[str, Any]:
    """Loads a YAML file into a dictionary."""

    with Path(path).open("r", encoding="utf-8") as handle:
        return yaml.safe_load(handle)


def save_json(payload: dict[str, Any], path: str | Path) -> Path:
    """Saves a small JSON artifact."""

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2)
    return path


def read_json(path: str | Path) -> dict[str, Any]:
    """Reads JSON into a dictionary."""

    with Path(path).open("r", encoding="utf-8") as handle:
        return json.load(handle)


def save_checkpoint(
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer | None,
    epoch: int,
    path: str | Path,
    extra_state: dict[str, Any] | None = None,
) -> Path:
    """Stores a PyTorch checkpoint under the requested path."""

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "epoch": epoch,
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict() if optimizer is not None else None,
        "extra_state": extra_state or {},
    }
    torch.save(payload, path)
    return path
