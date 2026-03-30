"""Metrics and summary helpers."""

from __future__ import annotations

from typing import Any

import numpy as np
import torch


def _to_numpy(values: torch.Tensor | np.ndarray) -> np.ndarray:
    """Converts a tensor-like input to a NumPy array."""

    if isinstance(values, torch.Tensor):
        return values.detach().cpu().numpy()
    return np.asarray(values)


def regression_metrics(
    prediction: torch.Tensor | np.ndarray,
    target: torch.Tensor | np.ndarray,
) -> dict[str, float]:
    """Computes simple regression metrics for reporting in notebooks."""

    prediction_np = _to_numpy(prediction).astype(np.float64)
    target_np = _to_numpy(target).astype(np.float64)
    error = prediction_np - target_np

    mae = float(np.mean(np.abs(error)))
    rmse = float(np.sqrt(np.mean(np.square(error))))
    denom = np.clip(np.abs(target_np), a_min=1e-6, a_max=None)
    mape = float(np.mean(np.abs(error) / denom) * 100.0)
    return {"mae": mae, "rmse": rmse, "mape_percent": mape}


def summarize_lux_map(lux_map: torch.Tensor | np.ndarray) -> dict[str, float]:
    """Summarizes average and distributional illuminance statistics."""

    lux_np = _to_numpy(lux_map).astype(np.float64).reshape(-1)
    return {
        "avg_lux": float(np.mean(lux_np)),
        "low_lux_p5": float(np.percentile(lux_np, 5)),
        "high_lux_p95": float(np.percentile(lux_np, 95)),
    }


def format_point_report(point_values: dict[str, float]) -> list[dict[str, Any]]:
    """Converts sampled point lux values to a notebook-friendly table format."""

    return [{"point_name": name, "lux": value} for name, value in point_values.items()]
