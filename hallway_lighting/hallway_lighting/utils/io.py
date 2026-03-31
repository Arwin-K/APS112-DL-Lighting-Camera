"""I/O helpers for configs, checkpoints, and notebook artifacts."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import torch
import yaml


def ensure_dir(path: str | Path) -> Path:
    """Creates a directory if needed and returns it as a `Path`."""

    path = Path(path)
    path.mkdir(parents=True, exist_ok=True)
    return path


def load_yaml(path: str | Path) -> dict[str, Any]:
    """Loads a YAML file into a dictionary."""

    with Path(path).open("r", encoding="utf-8") as handle:
        return yaml.safe_load(handle)


def save_yaml(payload: dict[str, Any], path: str | Path) -> Path:
    """Writes a dictionary as YAML."""

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        yaml.safe_dump(payload, handle, sort_keys=False)
    return path


def save_json(payload: dict[str, Any] | list[Any], path: str | Path) -> Path:
    """Saves a JSON artifact."""

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2)
    return path


def read_json(path: str | Path) -> dict[str, Any]:
    """Reads JSON into a dictionary."""

    with Path(path).open("r", encoding="utf-8") as handle:
        return json.load(handle)


def save_config_snapshot(configs: dict[str, dict[str, Any]], output_dir: str | Path) -> dict[str, Path]:
    """Saves a named collection of config dictionaries under an output directory."""

    output_dir = ensure_dir(output_dir)
    saved_paths: dict[str, Path] = {}
    for config_name, payload in configs.items():
        saved_paths[config_name] = save_yaml(payload, output_dir / f"{config_name}.yaml")
    return saved_paths


def save_training_history(history: dict[str, Any], path: str | Path) -> Path:
    """Saves training history as JSON."""

    return save_json(history, path)


def save_checkpoint(
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer | None,
    epoch: int,
    path: str | Path,
    scheduler: torch.optim.lr_scheduler.LRScheduler | None = None,
    scaler: torch.cuda.amp.GradScaler | None = None,
    history: dict[str, Any] | None = None,
    extra_state: dict[str, Any] | None = None,
) -> Path:
    """Stores a full training checkpoint."""

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "epoch": epoch,
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict() if optimizer is not None else None,
        "scheduler_state_dict": scheduler.state_dict() if scheduler is not None else None,
        "scaler_state_dict": scaler.state_dict() if scaler is not None else None,
        "history": history or {},
        "extra_state": extra_state or {},
    }
    torch.save(payload, path)
    return path


def load_checkpoint(
    path: str | Path,
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer | None = None,
    scheduler: torch.optim.lr_scheduler.LRScheduler | None = None,
    scaler: torch.cuda.amp.GradScaler | None = None,
    map_location: str | torch.device = "cpu",
) -> dict[str, Any]:
    """Loads a checkpoint and restores model/training state when requested."""

    checkpoint = torch.load(Path(path), map_location=map_location)
    model.load_state_dict(checkpoint["model_state_dict"])

    if optimizer is not None and checkpoint.get("optimizer_state_dict") is not None:
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
    if scheduler is not None and checkpoint.get("scheduler_state_dict") is not None:
        scheduler.load_state_dict(checkpoint["scheduler_state_dict"])
    if scaler is not None and checkpoint.get("scaler_state_dict") is not None:
        scaler.load_state_dict(checkpoint["scaler_state_dict"])

    return checkpoint
