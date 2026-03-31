"""Notebook-facing training helpers for the hallway lighting project."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import pandas as pd
from PIL import Image
from skimage.io import imread
import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset
import torch.nn.functional as F

from hallway_lighting.data.custom_hallway import load_point_target_values
from hallway_lighting.data.point_sampling import PointTarget
from hallway_lighting.data.transforms import build_image_transform
from hallway_lighting.losses import (
    albedo_regression_loss,
    avg_lux_loss,
    carbon_interval_loss,
    gloss_regression_loss,
    heteroscedastic_l1_loss,
    log_lux_smooth_l1_loss,
    p5_lux_loss,
    p95_lux_loss,
    pointwise_lux_loss,
    power_regression_loss,
    segmentation_loss,
    uncertainty_regularization_loss,
)
from hallway_lighting.utils.metrics import multitask_lux_metrics, summarize_lux_map
from hallway_lighting.utils.seed import make_torch_generator, seed_worker


DATASET_SUPERVISION_RULES: dict[str, set[str]] = {
    "nyu_depth_v2": {"floor"},
    "mit_intrinsic_images": {"albedo"},
    "mid_intrinsics": {"albedo", "gloss"},
    "fast_sv_indoor_lighting": {"albedo", "gloss"},
    "custom_hallway": {"floor", "lux", "stats", "point", "albedo", "gloss", "power", "carbon"},
}


def _read_rgb_image(path: str | Path) -> Image.Image:
    """Loads an RGB image, with a NumPy fallback for non-standard formats."""

    path = Path(path)
    try:
        return Image.open(path).convert("RGB")
    except Exception:
        array = np.asarray(imread(path))
        if array.ndim == 2:
            array = np.repeat(array[..., None], 3, axis=-1)
        if array.shape[-1] > 3:
            array = array[..., :3]
        if array.dtype != np.uint8:
            array = array.astype(np.float32)
            array = array - array.min()
            max_value = float(array.max())
            if max_value > 0:
                array = array / max_value
            array = (array * 255.0).clip(0, 255).astype(np.uint8)
        return Image.fromarray(array)


def _load_map_array(path: str | Path) -> np.ndarray:
    """Loads a dense map from `.npy`, `.npz`, or image files."""

    path = Path(path)
    suffix = path.suffix.lower()
    if suffix == ".npy":
        return np.asarray(np.load(path))
    if suffix == ".npz":
        archive = np.load(path)
        if not archive.files:
            raise ValueError(f"Empty npz archive: {path}")
        return np.asarray(archive[archive.files[0]])

    image = Image.open(path)
    return np.asarray(image)


def _resize_tensor_map(
    array: np.ndarray,
    image_size: tuple[int, int],
    mode: str = "bilinear",
    channels: int = 1,
) -> torch.Tensor:
    """Resizes a dense map to the requested tensor shape."""

    tensor = torch.as_tensor(array, dtype=torch.float32)
    if tensor.ndim == 2:
        tensor = tensor.unsqueeze(0).unsqueeze(0)
    elif tensor.ndim == 3:
        if tensor.shape[-1] in {1, 3}:
            tensor = tensor.permute(2, 0, 1).unsqueeze(0)
        else:
            tensor = tensor.unsqueeze(0)
    else:
        raise ValueError(f"Unsupported map shape {tuple(tensor.shape)}")

    tensor = F.interpolate(
        tensor,
        size=image_size,
        mode=mode,
        align_corners=False if mode != "nearest" else None,
    )
    tensor = tensor.squeeze(0)

    if channels == 1 and tensor.shape[0] != 1:
        tensor = tensor[:1]
    if channels == 3 and tensor.shape[0] == 1:
        tensor = tensor.repeat(3, 1, 1)
    return tensor


def _load_optional_dense_target(
    path_value: Any,
    image_size: tuple[int, int],
    channels: int,
    mode: str,
) -> torch.Tensor | None:
    """Loads an optional dense target tensor."""

    if path_value is None:
        return None
    if isinstance(path_value, float) and np.isnan(path_value):
        return None
    path_text = str(path_value).strip()
    if not path_text:
        return None
    path = Path(path_text)
    if not path.exists():
        raise FileNotFoundError(f"Target file does not exist: {path}")
    array = _load_map_array(path)
    return _resize_tensor_map(array, image_size=image_size, mode=mode, channels=channels)


def _optional_scalar(value: Any) -> float | None:
    """Converts manifest scalars to floats or None."""

    if value is None:
        return None
    if isinstance(value, float) and np.isnan(value):
        return None
    if isinstance(value, str) and not value.strip():
        return None
    return float(value)


class ManifestMultitaskDataset(Dataset):
    """Loads images and optional supervision from a normalized manifest."""

    def __init__(
        self,
        manifest: pd.DataFrame,
        image_size: tuple[int, int] = (256, 256),
        normalize_images: bool = True,
    ) -> None:
        super().__init__()
        self.manifest = manifest.reset_index(drop=True).copy()
        self.image_size = image_size
        self.image_transform = build_image_transform(image_size=image_size, normalize=normalize_images)

    def __len__(self) -> int:
        """Returns the number of manifest rows."""

        return len(self.manifest)

    def __getitem__(self, index: int) -> dict[str, Any]:
        """Loads one manifest row into a multitask sample dictionary."""

        row = self.manifest.iloc[index]
        image_path = Path(row["image_path"])
        if not image_path.exists():
            raise FileNotFoundError(f"Image file does not exist: {image_path}")

        image = self.image_transform(_read_rgb_image(image_path))
        dataset_name = str(row["dataset_name"])
        sample: dict[str, Any] = {
            "sample_id": str(row["sample_id"]),
            "dataset_name": dataset_name,
            "split": str(row["split"]),
            "image_path": str(image_path),
            "image": image,
            "floor_mask": _load_optional_dense_target(row.get("floor_mask_path"), self.image_size, channels=1, mode="nearest"),
            "lux_map": _load_optional_dense_target(row.get("lux_map_path"), self.image_size, channels=1, mode="bilinear"),
            "albedo": _load_optional_dense_target(
                row.get("albedo_path") or row.get("reflectance_path"),
                self.image_size,
                channels=3,
                mode="bilinear",
            ),
            "gloss": _load_optional_dense_target(row.get("gloss_path"), self.image_size, channels=1, mode="bilinear"),
            "avg_lux": _optional_scalar(row.get("avg_lux")),
            "low_lux_p5": _optional_scalar(row.get("low_lux_p5")),
            "high_lux_p95": _optional_scalar(row.get("high_lux_p95")),
            "measured_power_w": _optional_scalar(row.get("measured_power_w")),
            "interval_hours": _optional_scalar(row.get("interval_hours")),
            "notes": str(row.get("notes", "")),
        }

        if sample["lux_map"] is not None:
            lux_summary = summarize_lux_map(sample["lux_map"].numpy())
            if sample["avg_lux"] is None:
                sample["avg_lux"] = lux_summary["avg_lux"]
            if sample["low_lux_p5"] is None:
                sample["low_lux_p5"] = lux_summary["low_lux_p5"]
            if sample["high_lux_p95"] is None:
                sample["high_lux_p95"] = lux_summary["high_lux_p95"]

        point_targets_json = row.get("point_targets_json")
        if point_targets_json is None or (isinstance(point_targets_json, float) and np.isnan(point_targets_json)):
            sample["point_lux"] = None
        else:
            point_targets_path = str(point_targets_json).strip()
            sample["point_lux"] = load_point_target_values(point_targets_path) if point_targets_path else None

        return sample


def _collate_optional_dense(samples: Sequence[dict[str, Any]], key: str) -> tuple[torch.Tensor | None, torch.Tensor | None]:
    """Stacks optional dense targets and returns a batch-valid mask."""

    available = [sample[key] for sample in samples if sample.get(key) is not None]
    if not available:
        return None, None
    template = available[0]
    stacked: list[torch.Tensor] = []
    valid_mask: list[bool] = []
    for sample in samples:
        value = sample.get(key)
        if value is None:
            stacked.append(torch.zeros_like(template))
            valid_mask.append(False)
        else:
            stacked.append(value)
            valid_mask.append(True)
    return torch.stack(stacked, dim=0), torch.tensor(valid_mask, dtype=torch.bool)


def _collate_optional_scalar(samples: Sequence[dict[str, Any]], key: str) -> tuple[torch.Tensor | None, torch.Tensor | None]:
    """Stacks optional scalar targets and returns a batch-valid mask."""

    available = [sample[key] for sample in samples if sample.get(key) is not None]
    if not available:
        return None, None
    values: list[float] = []
    valid_mask: list[bool] = []
    for sample in samples:
        value = sample.get(key)
        if value is None:
            values.append(0.0)
            valid_mask.append(False)
        else:
            values.append(float(value))
            valid_mask.append(True)
    tensor = torch.tensor(values, dtype=torch.float32).unsqueeze(1)
    return tensor, torch.tensor(valid_mask, dtype=torch.bool)


def _collate_point_targets(
    samples: Sequence[dict[str, Any]],
) -> tuple[dict[str, torch.Tensor] | None, dict[str, torch.Tensor] | None]:
    """Stacks optional point-wise lux targets."""

    available = [sample["point_lux"] for sample in samples if sample.get("point_lux") is not None]
    if not available:
        return None, None

    point_names = sorted({name for point_dict in available for name in point_dict.keys()})
    point_values: dict[str, torch.Tensor] = {}
    point_masks: dict[str, torch.Tensor] = {}

    for point_name in point_names:
        values: list[float] = []
        valid: list[bool] = []
        for sample in samples:
            point_dict = sample.get("point_lux")
            if point_dict is None or point_name not in point_dict:
                values.append(0.0)
                valid.append(False)
            else:
                values.append(float(point_dict[point_name]))
                valid.append(True)
        point_values[point_name] = torch.tensor(values, dtype=torch.float32)
        point_masks[point_name] = torch.tensor(valid, dtype=torch.bool)

    return point_values, point_masks


def multitask_collate_fn(samples: Sequence[dict[str, Any]]) -> dict[str, Any]:
    """Collates manifest samples into a batch with availability masks."""

    batch: dict[str, Any] = {
        "sample_id": [sample["sample_id"] for sample in samples],
        "dataset_name": [sample["dataset_name"] for sample in samples],
        "split": [sample["split"] for sample in samples],
        "image_path": [sample["image_path"] for sample in samples],
        "notes": [sample["notes"] for sample in samples],
        "image": torch.stack([sample["image"] for sample in samples], dim=0),
    }

    for key in ("floor_mask", "lux_map", "albedo", "gloss"):
        values, mask = _collate_optional_dense(samples, key)
        batch[key] = values
        batch[f"{key}_valid_mask"] = mask

    for key in ("avg_lux", "low_lux_p5", "high_lux_p95", "measured_power_w", "interval_hours"):
        values, mask = _collate_optional_scalar(samples, key)
        batch[key] = values
        batch[f"{key}_valid_mask"] = mask

    point_values, point_masks = _collate_point_targets(samples)
    batch["point_lux"] = point_values
    batch["point_lux_valid_mask"] = point_masks
    return batch


def combine_manifests_for_split(
    manifests: Mapping[str, pd.DataFrame],
    split: str,
) -> pd.DataFrame:
    """Combines all manifest rows matching a split name."""

    rows: list[pd.DataFrame] = []
    for manifest in manifests.values():
        split_rows = manifest.loc[manifest["split"].fillna("").str.lower() == split.lower()].copy()
        if not split_rows.empty:
            rows.append(split_rows)
    if not rows:
        return pd.DataFrame()
    return pd.concat(rows, ignore_index=True)


def build_dataloaders(
    manifests: Mapping[str, pd.DataFrame],
    batch_size: int,
    num_workers: int,
    image_size: tuple[int, int],
    seed: int,
) -> dict[str, DataLoader | None]:
    """Creates train/val/test dataloaders from normalized manifests."""

    generator = make_torch_generator(seed)
    dataloaders: dict[str, DataLoader | None] = {}

    for split_name in ("train", "val", "test"):
        split_manifest = combine_manifests_for_split(manifests, split_name)
        if split_manifest.empty:
            dataloaders[split_name] = None
            continue

        dataset = ManifestMultitaskDataset(split_manifest, image_size=image_size)
        dataloaders[split_name] = DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=split_name == "train",
            num_workers=num_workers,
            pin_memory=True,
            collate_fn=multitask_collate_fn,
            worker_init_fn=seed_worker,
            generator=generator,
        )

    return dataloaders


def _move_to_device(value: Any, device: torch.device) -> Any:
    """Moves nested tensors to a device."""

    if value is None:
        return None
    if isinstance(value, torch.Tensor):
        return value.to(device)
    if isinstance(value, dict):
        return {key: _move_to_device(item, device) for key, item in value.items()}
    return value


def _dataset_task_mask(dataset_names: Sequence[str], task_name: str, device: torch.device) -> torch.Tensor:
    """Builds a per-sample boolean mask for a supervision task."""

    flags = [task_name in DATASET_SUPERVISION_RULES.get(name, set()) for name in dataset_names]
    return torch.tensor(flags, dtype=torch.bool, device=device)


def _combine_sample_masks(
    availability_mask: torch.Tensor | None,
    dataset_mask: torch.Tensor,
) -> torch.Tensor | None:
    """Combines sample-availability and dataset-routing masks."""

    if availability_mask is None:
        return None
    return availability_mask.to(dataset_mask.device) & dataset_mask


def _combine_point_masks(
    point_masks: dict[str, torch.Tensor] | None,
    dataset_mask: torch.Tensor,
) -> dict[str, torch.Tensor] | None:
    """Combines point-level availability masks with dataset-routing masks."""

    if point_masks is None:
        return None
    return {name: mask.to(dataset_mask.device) & dataset_mask for name, mask in point_masks.items()}


def compute_multitask_loss(
    outputs: Mapping[str, Any],
    batch: Mapping[str, Any],
    loss_weights: Mapping[str, float],
    carbon_config: Mapping[str, Any],
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    """Computes routed multitask losses for a single batch."""

    device = outputs["lux_map"].device
    dataset_names = batch["dataset_name"]

    floor_dataset_mask = _dataset_task_mask(dataset_names, "floor", device)
    lux_dataset_mask = _dataset_task_mask(dataset_names, "lux", device)
    stats_dataset_mask = _dataset_task_mask(dataset_names, "stats", device)
    point_dataset_mask = _dataset_task_mask(dataset_names, "point", device)
    albedo_dataset_mask = _dataset_task_mask(dataset_names, "albedo", device)
    gloss_dataset_mask = _dataset_task_mask(dataset_names, "gloss", device)
    power_dataset_mask = _dataset_task_mask(dataset_names, "power", device)
    carbon_dataset_mask = _dataset_task_mask(dataset_names, "carbon", device)

    loss_breakdown = {
        "lux_map": log_lux_smooth_l1_loss(
            outputs["lux_map"],
            batch["lux_map"],
            valid_mask=_combine_sample_masks(batch["lux_map_valid_mask"], lux_dataset_mask),
        ),
        "avg_lux": avg_lux_loss(
            outputs["avg_lux"],
            batch["avg_lux"],
            valid_mask=_combine_sample_masks(batch["avg_lux_valid_mask"], stats_dataset_mask),
        ),
        "p5_lux": p5_lux_loss(
            outputs["low_lux_p5"],
            batch["low_lux_p5"],
            valid_mask=_combine_sample_masks(batch["low_lux_p5_valid_mask"], stats_dataset_mask),
        ),
        "p95_lux": p95_lux_loss(
            outputs["high_lux_p95"],
            batch["high_lux_p95"],
            valid_mask=_combine_sample_masks(batch["high_lux_p95_valid_mask"], stats_dataset_mask),
        ),
        "point_lux": pointwise_lux_loss(
            outputs["point_lux"],
            batch["point_lux"],
            valid_mask=_combine_point_masks(batch["point_lux_valid_mask"], point_dataset_mask),
        ),
        "floor_mask": segmentation_loss(
            outputs["floor_logits"],
            batch["floor_mask"],
            valid_mask=_combine_sample_masks(batch["floor_mask_valid_mask"], floor_dataset_mask),
        ),
        "albedo": albedo_regression_loss(
            outputs["albedo_pred"],
            batch["albedo"],
            valid_mask=_combine_sample_masks(batch["albedo_valid_mask"], albedo_dataset_mask),
        ),
        "gloss": gloss_regression_loss(
            outputs["gloss_pred"],
            batch["gloss"],
            valid_mask=_combine_sample_masks(batch["gloss_valid_mask"], gloss_dataset_mask),
        ),
        "uncertainty": heteroscedastic_l1_loss(
            outputs["lux_map"],
            batch["lux_map"],
            outputs["uncertainty_map"],
            valid_mask=_combine_sample_masks(batch["lux_map_valid_mask"], lux_dataset_mask),
        )
        + uncertainty_regularization_loss(
            outputs["uncertainty_map"],
            valid_mask=_combine_sample_masks(batch["lux_map_valid_mask"], lux_dataset_mask),
            weight=0.05,
        ),
        "power": power_regression_loss(
            outputs["estimated_power_w"],
            batch["measured_power_w"],
            valid_mask=_combine_sample_masks(batch["measured_power_w_valid_mask"], power_dataset_mask),
        ),
    }

    carbon_factor = float(carbon_config["carbon"]["default_grid_carbon_factor_kg_per_kwh"])
    default_interval_hours = float(carbon_config["carbon"]["default_interval_hours"])
    if batch["measured_power_w"] is not None:
        interval_hours = (
            batch["interval_hours"]
            if batch["interval_hours"] is not None
            else torch.full_like(batch["measured_power_w"], default_interval_hours)
        )
        target_carbon = batch["measured_power_w"] / 1000.0 * interval_hours * carbon_factor
    else:
        target_carbon = None
    carbon_mask = None
    if batch["measured_power_w_valid_mask"] is not None:
        interval_mask = (
            batch["interval_hours_valid_mask"]
            if batch["interval_hours_valid_mask"] is not None
            else batch["measured_power_w_valid_mask"]
        )
        carbon_mask = _combine_sample_masks(
            batch["measured_power_w_valid_mask"] & interval_mask,
            carbon_dataset_mask,
        )

    loss_breakdown["carbon"] = carbon_interval_loss(
        outputs["estimated_power_w"],
        target_carbon,
        carbon_factor_kg_per_kwh=carbon_factor,
        interval_hours=interval_hours if batch["measured_power_w"] is not None else default_interval_hours,
        valid_mask=carbon_mask,
    )

    total_loss = outputs["lux_map"].new_zeros(())
    for loss_name, loss_value in loss_breakdown.items():
        total_loss = total_loss + float(loss_weights.get(loss_name, 1.0)) * loss_value

    return total_loss, loss_breakdown


def _tensor_batch_to_cpu_dict(point_dict: Mapping[str, torch.Tensor], index: int) -> dict[str, float]:
    """Extracts one sample from a batched point dictionary."""

    return {name: float(values[index].detach().cpu().item()) for name, values in point_dict.items()}


@dataclass
class EpochResult:
    """Aggregated result for one train/val/test epoch."""

    summary: dict[str, float]
    visual_examples: list[dict[str, Any]]
    point_reports: list[dict[str, Any]]


def run_epoch(
    model: nn.Module,
    dataloader: DataLoader | None,
    device: torch.device,
    loss_weights: Mapping[str, float],
    carbon_config: Mapping[str, Any],
    optimizer: torch.optim.Optimizer | None = None,
    scaler: torch.cuda.amp.GradScaler | None = None,
    amp_enabled: bool = False,
    max_visualization_examples: int = 4,
    gradient_clip_norm: float = 1.0,
) -> EpochResult:
    """Runs one train, validation, or test epoch."""

    if dataloader is None:
        return EpochResult(summary={}, visual_examples=[], point_reports=[])

    is_training = optimizer is not None
    model.train(is_training)
    device_type = "cuda" if device.type == "cuda" else "cpu"

    loss_sums: dict[str, float] = {"total_loss": 0.0}
    metric_values: dict[str, list[float]] = {}
    visual_examples: list[dict[str, Any]] = []
    point_reports: list[dict[str, Any]] = []

    for batch_index, batch in enumerate(dataloader):
        batch = {key: _move_to_device(value, device) for key, value in batch.items()}
        image_tensor = batch["image"]

        autocast_enabled = amp_enabled and device_type == "cuda"
        with torch.set_grad_enabled(is_training):
            with torch.autocast(device_type=device_type, enabled=autocast_enabled):
                outputs = model(image_tensor)
                total_loss, loss_breakdown = compute_multitask_loss(
                    outputs=outputs,
                    batch=batch,
                    loss_weights=loss_weights,
                    carbon_config=carbon_config,
                )

        if is_training:
            optimizer.zero_grad(set_to_none=True)
            if scaler is not None and autocast_enabled:
                scaler.scale(total_loss).backward()
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=gradient_clip_norm)
                scaler.step(optimizer)
                scaler.update()
            else:
                total_loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=gradient_clip_norm)
                optimizer.step()

        loss_sums["total_loss"] += float(total_loss.detach().cpu())
        for loss_name, loss_value in loss_breakdown.items():
            loss_sums.setdefault(loss_name, 0.0)
            loss_sums[loss_name] += float(loss_value.detach().cpu())

        batch_metrics = multitask_lux_metrics(
            outputs={
                "lux_map": outputs["lux_map"].detach(),
                "avg_lux": outputs["avg_lux"].detach(),
                "low_lux_p5": outputs["low_lux_p5"].detach(),
                "high_lux_p95": outputs["high_lux_p95"].detach(),
                "point_lux": outputs["point_lux"],
            },
            targets={
                "lux_map": batch.get("lux_map"),
                "lux_map_valid_mask": batch.get("lux_map_valid_mask"),
                "avg_lux": batch.get("avg_lux"),
                "avg_lux_valid_mask": batch.get("avg_lux_valid_mask"),
                "low_lux_p5": batch.get("low_lux_p5"),
                "low_lux_p5_valid_mask": batch.get("low_lux_p5_valid_mask"),
                "high_lux_p95": batch.get("high_lux_p95"),
                "high_lux_p95_valid_mask": batch.get("high_lux_p95_valid_mask"),
                "point_lux": batch.get("point_lux"),
                "point_lux_valid_mask": batch.get("point_lux_valid_mask"),
            },
        )
        for metric_name, metric_value in batch_metrics.items():
            metric_values.setdefault(metric_name, []).append(metric_value)

        if len(visual_examples) < max_visualization_examples:
            sample_index = 0
            visual_examples.append(
                {
                    "sample_id": batch["sample_id"][sample_index],
                    "dataset_name": batch["dataset_name"][sample_index],
                    "image": batch["image"][sample_index].detach().cpu(),
                    "lux_map_pred": outputs["lux_map"][sample_index].detach().cpu(),
                    "floor_mask_pred": outputs["floor_mask_pred"][sample_index].detach().cpu(),
                    "albedo_pred": outputs["albedo_pred"][sample_index].detach().cpu(),
                    "gloss_pred": outputs["gloss_pred"][sample_index].detach().cpu(),
                    "point_targets": outputs["point_targets"],
                    "point_lux_pred": _tensor_batch_to_cpu_dict(outputs["point_lux"], sample_index),
                    "point_lux_target": (
                        _tensor_batch_to_cpu_dict(batch["point_lux"], sample_index)
                        if batch.get("point_lux") is not None
                        else None
                    ),
                }
            )

        if batch.get("point_lux") is not None and len(point_reports) < max_visualization_examples:
            sample_index = 0
            point_reports.append(
                {
                    "sample_id": batch["sample_id"][sample_index],
                    "dataset_name": batch["dataset_name"][sample_index],
                    "predicted_points": _tensor_batch_to_cpu_dict(outputs["point_lux"], sample_index),
                    "target_points": _tensor_batch_to_cpu_dict(batch["point_lux"], sample_index),
                }
            )

    batch_count = max(len(dataloader), 1)
    summary = {
        loss_name: value / batch_count
        for loss_name, value in loss_sums.items()
    }
    for metric_name, values in metric_values.items():
        summary[metric_name] = float(np.mean(values)) if values else 0.0

    return EpochResult(summary=summary, visual_examples=visual_examples, point_reports=point_reports)
