"""Metrics and summary helpers for hallway lighting outputs."""

from __future__ import annotations

from typing import Any, Mapping

import numpy as np
import torch


def _to_numpy(values: torch.Tensor | np.ndarray) -> np.ndarray:
    """Converts tensors or arrays to NumPy."""

    if isinstance(values, torch.Tensor):
        return values.detach().cpu().numpy()
    return np.asarray(values)


def _masked_error_array(
    prediction: torch.Tensor | np.ndarray,
    target: torch.Tensor | np.ndarray,
    valid_mask: torch.Tensor | np.ndarray | None = None,
) -> np.ndarray:
    """Builds a 1D error array after optional masking."""

    prediction_np = _to_numpy(prediction).astype(np.float64)
    target_np = _to_numpy(target).astype(np.float64)
    error = (prediction_np - target_np).reshape(-1)

    if valid_mask is None:
        return error

    mask_np = _to_numpy(valid_mask).astype(bool).reshape(-1)
    if mask_np.size != error.size:
        mask_np = np.broadcast_to(mask_np, prediction_np.shape).reshape(-1)
    return error[mask_np]


def mae(
    prediction: torch.Tensor | np.ndarray,
    target: torch.Tensor | np.ndarray,
    valid_mask: torch.Tensor | np.ndarray | None = None,
) -> float:
    """Computes mean absolute error."""

    error = _masked_error_array(prediction, target, valid_mask=valid_mask)
    if error.size == 0:
        return 0.0
    return float(np.mean(np.abs(error)))


def rmse(
    prediction: torch.Tensor | np.ndarray,
    target: torch.Tensor | np.ndarray,
    valid_mask: torch.Tensor | np.ndarray | None = None,
) -> float:
    """Computes root mean squared error."""

    error = _masked_error_array(prediction, target, valid_mask=valid_mask)
    if error.size == 0:
        return 0.0
    return float(np.sqrt(np.mean(np.square(error))))


def summarize_lux_map(
    lux_map: torch.Tensor | np.ndarray,
    floor_mask: torch.Tensor | np.ndarray | None = None,
) -> dict[str, float]:
    """Summarizes average, p5, and p95 lux values."""

    lux_np = _to_numpy(lux_map).astype(np.float64)
    if floor_mask is not None:
        mask_np = _to_numpy(floor_mask).astype(bool)
        if mask_np.shape != lux_np.shape:
            mask_np = np.broadcast_to(mask_np, lux_np.shape)
        lux_values = lux_np[mask_np]
    else:
        lux_values = lux_np.reshape(-1)

    if lux_values.size == 0:
        return {"avg_lux": 0.0, "low_lux_p5": 0.0, "high_lux_p95": 0.0}

    return {
        "avg_lux": float(np.mean(lux_values)),
        "low_lux_p5": float(np.percentile(lux_values, 5)),
        "high_lux_p95": float(np.percentile(lux_values, 95)),
    }


def avg_lux_error(
    prediction: torch.Tensor | np.ndarray,
    target: torch.Tensor | np.ndarray,
    valid_mask: torch.Tensor | np.ndarray | None = None,
) -> float:
    """Computes absolute error for average lux."""

    return mae(prediction, target, valid_mask=valid_mask)


def p5_error(
    prediction: torch.Tensor | np.ndarray,
    target: torch.Tensor | np.ndarray,
    valid_mask: torch.Tensor | np.ndarray | None = None,
) -> float:
    """Computes absolute error for p5 lux."""

    return mae(prediction, target, valid_mask=valid_mask)


def p95_error(
    prediction: torch.Tensor | np.ndarray,
    target: torch.Tensor | np.ndarray,
    valid_mask: torch.Tensor | np.ndarray | None = None,
) -> float:
    """Computes absolute error for p95 lux."""

    return mae(prediction, target, valid_mask=valid_mask)


def pointwise_lux_error(
    prediction: Mapping[str, torch.Tensor | np.ndarray | float],
    target: Mapping[str, torch.Tensor | np.ndarray | float] | None,
    valid_mask: Mapping[str, torch.Tensor | np.ndarray] | None = None,
) -> float | None:
    """Computes mean absolute point-wise lux error across shared point names."""

    if target is None:
        return None
    shared_names = [name for name in prediction.keys() if name in target]
    if not shared_names:
        return None

    point_errors: list[float] = []
    for point_name in shared_names:
        prediction_value = prediction[point_name]
        target_value = target[point_name]
        point_errors.append(
            mae(
                prediction_value,
                target_value,
                valid_mask=None if valid_mask is None else valid_mask.get(point_name),
            )
        )
    return float(np.mean(point_errors))


def regression_metrics(
    prediction: torch.Tensor | np.ndarray,
    target: torch.Tensor | np.ndarray,
    valid_mask: torch.Tensor | np.ndarray | None = None,
) -> dict[str, float]:
    """Computes generic regression metrics."""

    prediction_np = _to_numpy(prediction).astype(np.float64)
    target_np = _to_numpy(target).astype(np.float64)
    error = _masked_error_array(prediction_np, target_np, valid_mask=valid_mask)
    if error.size == 0:
        return {"mae": 0.0, "rmse": 0.0, "mape_percent": 0.0}

    filtered_target = target_np.reshape(-1)
    if valid_mask is not None:
        mask_np = _to_numpy(valid_mask).astype(bool).reshape(-1)
        if mask_np.size != filtered_target.size:
            mask_np = np.broadcast_to(mask_np, prediction_np.shape).reshape(-1)
        filtered_target = filtered_target[mask_np]

    denom = np.clip(np.abs(filtered_target), a_min=1e-6, a_max=None)
    mape = float(np.mean(np.abs(error) / denom) * 100.0)
    return {
        "mae": float(np.mean(np.abs(error))),
        "rmse": float(np.sqrt(np.mean(np.square(error)))),
        "mape_percent": mape,
    }


def multitask_lux_metrics(
    outputs: Mapping[str, Any],
    targets: Mapping[str, Any],
) -> dict[str, float]:
    """Aggregates reusable hallway lighting metrics from model outputs."""

    metrics: dict[str, float] = {}

    if "lux_map" in outputs and "lux_map" in targets and targets["lux_map"] is not None:
        valid_mask = targets.get("lux_map_valid_mask")
        metrics["lux_mae"] = mae(outputs["lux_map"], targets["lux_map"], valid_mask=valid_mask)
        metrics["lux_rmse"] = rmse(outputs["lux_map"], targets["lux_map"], valid_mask=valid_mask)

    if "avg_lux" in outputs and "avg_lux" in targets and targets["avg_lux"] is not None:
        metrics["avg_lux_error"] = avg_lux_error(
            outputs["avg_lux"],
            targets["avg_lux"],
            valid_mask=targets.get("avg_lux_valid_mask"),
        )

    if "low_lux_p5" in outputs and "low_lux_p5" in targets and targets["low_lux_p5"] is not None:
        metrics["p5_error"] = p5_error(
            outputs["low_lux_p5"],
            targets["low_lux_p5"],
            valid_mask=targets.get("low_lux_p5_valid_mask"),
        )

    if "high_lux_p95" in outputs and "high_lux_p95" in targets and targets["high_lux_p95"] is not None:
        metrics["p95_error"] = p95_error(
            outputs["high_lux_p95"],
            targets["high_lux_p95"],
            valid_mask=targets.get("high_lux_p95_valid_mask"),
        )

    point_error = pointwise_lux_error(
        outputs.get("point_lux", {}),
        targets.get("point_lux"),
        valid_mask=targets.get("point_lux_valid_mask"),
    )
    if point_error is not None:
        metrics["pointwise_lux_error"] = point_error

    return metrics


def format_point_report(point_values: Mapping[str, float]) -> list[dict[str, Any]]:
    """Converts point lux values to a notebook-friendly table."""

    return [{"point_name": name, "lux": value} for name, value in point_values.items()]
