"""Shared helper code embedded into the local ONNX notebook."""

from __future__ import annotations

from io import BytesIO
import json
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
from matplotlib.path import Path as MplPath
from matplotlib.patches import Polygon
import numpy as np
from PIL import Image

MODEL_FILENAME = "hallway_multitask_unet_drive_prototype.onnx"
COLAB_PROJECT_ROOT = Path("/content/APS112-DL-Lighting-Camera/hallway_lighting")
IMAGENET_MEAN = np.asarray([0.485, 0.456, 0.406], dtype=np.float32).reshape(1, 1, 3)
IMAGENET_STD = np.asarray([0.229, 0.224, 0.225], dtype=np.float32).reshape(1, 1, 3)
DARK_FRAME_MEAN_THRESHOLD = 0.03
DARK_FRAME_P95_THRESHOLD = 0.08
EXPECTED_ONNX_OUTPUT_NAMES = [
    "lux_map",
    "avg_lux",
    "low_lux_p5",
    "high_lux_p95",
    "floor_mask_pred",
    "albedo_pred",
    "gloss_pred",
    "uncertainty_map",
    "estimated_power_w",
]


def load_onnx_session(onnx_path: Path):
    """Loads the ONNX runtime session on CPU."""

    try:
        import onnxruntime as ort
    except ImportError as exc:
        raise ImportError(
            "onnxruntime is not installed in this notebook environment. "
            "Run the install cell, restart the kernel, and rerun the notebook."
        ) from exc

    if not onnx_path.exists():
        raise FileNotFoundError(f"ONNX model not found: {onnx_path}")
    return ort.InferenceSession(str(onnx_path), providers=["CPUExecutionProvider"])


def extract_single_map(value: np.ndarray | float | int | None) -> np.ndarray | None:
    """Returns a single 2D or HxWxC array from an ONNX output value."""

    if value is None or isinstance(value, (float, int)):
        return None
    array = np.asarray(value)
    if array.ndim == 4:
        array = array[0]
    if array.ndim == 3 and array.shape[0] in {1, 3}:
        array = np.transpose(array, (1, 2, 0))
    if array.ndim == 3 and array.shape[-1] == 1:
        array = array[..., 0]
    return array.astype(np.float32, copy=False)


def extract_scalar(value: np.ndarray | float | int | None, fallback: float = 0.0) -> float:
    """Extracts a scalar from a tensor-like output."""

    if value is None:
        return fallback
    if isinstance(value, (float, int)):
        return float(value)
    array = np.asarray(value).reshape(-1)
    return fallback if array.size == 0 else float(array[0])


def assess_image_quality(display_rgb: np.ndarray) -> dict[str, float | bool]:
    """Computes simple brightness metrics and a dark-frame gate."""

    rgb = np.clip(np.asarray(display_rgb, dtype=np.float32), 0.0, 1.0)
    luminance = 0.2126 * rgb[..., 0] + 0.7152 * rgb[..., 1] + 0.0722 * rgb[..., 2]
    bottom_start = min(rgb.shape[0] - 1, max(0, int(round(rgb.shape[0] * 0.55))))
    bottom_band = luminance[bottom_start:, :] if bottom_start < rgb.shape[0] else luminance

    mean_luminance = float(np.mean(luminance))
    p95_luminance = float(np.percentile(luminance, 95))
    max_luminance = float(np.max(luminance))
    bottom_mean_luminance = float(np.mean(bottom_band))
    is_dark_frame = (
        mean_luminance < DARK_FRAME_MEAN_THRESHOLD
        and p95_luminance < DARK_FRAME_P95_THRESHOLD
    )
    return {
        "mean_luminance": mean_luminance,
        "p95_luminance": p95_luminance,
        "max_luminance": max_luminance,
        "bottom_mean_luminance": bottom_mean_luminance,
        "is_dark_frame": bool(is_dark_frame),
    }


def preprocess_uploaded_image(
    image_bytes: bytes,
    input_height: int,
    input_width: int,
) -> tuple[np.ndarray, np.ndarray, dict[str, float | bool]]:
    """Resizes image, keeps a display copy, and applies training-time normalization."""

    image = Image.open(BytesIO(image_bytes)).convert("RGB")
    resized = image.resize((input_width, input_height), resample=Image.BILINEAR)
    display_rgb = np.asarray(resized).astype(np.float32) / 255.0
    quality = assess_image_quality(display_rgb)
    normalized_rgb = (display_rgb - IMAGENET_MEAN) / IMAGENET_STD
    model_input = np.transpose(normalized_rgb, (2, 0, 1))[None, ...].astype(np.float32)
    return model_input, display_rgb, quality


def summarize_lux_map(lux_map: np.ndarray, floor_mask: np.ndarray | None = None) -> dict[str, float]:
    """Computes basic lux summary statistics."""

    values = np.asarray(lux_map, dtype=np.float32)
    if floor_mask is not None:
        mask = np.asarray(floor_mask).astype(bool)
        if mask.shape == values.shape and float(mask.mean()) > 0.001:
            values = values[mask]
        else:
            values = values.reshape(-1)
    else:
        values = values.reshape(-1)

    if values.size == 0:
        return {"avg_lux": 0.0, "low_lux_p5": 0.0, "high_lux_p95": 0.0}
    return {
        "avg_lux": float(np.mean(values)),
        "low_lux_p5": float(np.percentile(values, 5)),
        "high_lux_p95": float(np.percentile(values, 95)),
    }


def _sample_mask_mean(lux_map: np.ndarray, mask: np.ndarray) -> float:
    mask_bool = np.asarray(mask, dtype=bool)
    if mask_bool.shape != lux_map.shape:
        raise ValueError("Sampling mask must match lux map shape.")
    if not np.any(mask_bool):
        return float("nan")
    return float(np.mean(np.asarray(lux_map, dtype=np.float32)[mask_bool]))


def _build_disc_mask(
    height: int,
    width: int,
    center_x: float,
    center_y: float,
    radius_x: float,
    radius_y: float,
) -> np.ndarray:
    y_coords, x_coords = np.ogrid[:height, :width]
    norm_x = (x_coords - center_x) / max(radius_x, 1.0)
    norm_y = (y_coords - center_y) / max(radius_y, 1.0)
    return (norm_x * norm_x + norm_y * norm_y) <= 1.0


def _polygon_mask(
    polygon_points: list[tuple[float, float]],
    height: int,
    width: int,
) -> np.ndarray:
    if not polygon_points:
        return np.zeros((height, width), dtype=bool)
    polygon_pixels = np.asarray(
        [
            (float(x_value) * (width - 1), float(y_value) * (height - 1))
            for x_value, y_value in polygon_points
        ],
        dtype=np.float32,
    )
    grid_x, grid_y = np.meshgrid(np.arange(width), np.arange(height))
    points = np.stack([grid_x.ravel(), grid_y.ravel()], axis=-1)
    mask = MplPath(polygon_pixels).contains_points(points, radius=0.5)
    return mask.reshape(height, width)


def _build_under_fixture_mask(
    point: dict[str, Any],
    floor_mask: np.ndarray | None,
    height: int,
    width: int,
) -> np.ndarray:
    center_x = float(point["x"]) * (width - 1)
    center_y = float(point["y"]) * (height - 1)

    if floor_mask is not None:
        row_index = int(np.clip(round(center_y), 0, height - 1))
        floor_columns = np.where(floor_mask[row_index])[0]
        floor_width = float(floor_columns[-1] - floor_columns[0]) if floor_columns.size > 1 else width * 0.2
        radius_x = max(4.0, floor_width * 0.10)
    else:
        radius_x = max(4.0, width * 0.06)
    radius_y = max(3.0, height * 0.02)

    mask = _build_disc_mask(height, width, center_x, center_y, radius_x, radius_y)
    if floor_mask is not None:
        mask &= floor_mask
    return mask


def compute_floor_measurements(
    lux_map: np.ndarray,
    floor_mask: np.ndarray | None,
    fixture_analysis: dict[str, Any] | None,
) -> tuple[dict[str, float], dict[str, float], dict[str, np.ndarray]]:
    """Computes floor-area lux summaries under and between fixtures."""

    if fixture_analysis is None:
        return {}, {}, {}

    height, width = lux_map.shape
    point_targets = fixture_analysis.get("point_targets") or []
    between_regions = fixture_analysis.get("between_regions") or []

    measurement_masks: dict[str, np.ndarray] = {}
    under_fixture_lux: dict[str, float] = {}
    between_fixture_lux: dict[str, float] = {}

    for point in point_targets:
        point_name = str(point["name"])
        if point.get("group") != "under_fixture":
            continue
        mask = _build_under_fixture_mask(point, floor_mask=floor_mask, height=height, width=width)
        measurement_masks[point_name] = mask
        under_fixture_lux[point_name] = _sample_mask_mean(lux_map, mask)

    for region in between_regions:
        region_name = str(region["name"])
        mask = _polygon_mask(region.get("polygon") or [], height=height, width=width)
        if floor_mask is not None:
            mask &= floor_mask
        measurement_masks[region_name] = mask
        between_fixture_lux[region_name] = _sample_mask_mean(lux_map, mask)

    return under_fixture_lux, between_fixture_lux, measurement_masks


def _find_measurement_value(
    point_name: str,
    under_fixture_lux: dict[str, float],
    between_fixture_lux: dict[str, float],
) -> float | None:
    if point_name in under_fixture_lux:
        return float(under_fixture_lux[point_name])
    if point_name in between_fixture_lux:
        return float(between_fixture_lux[point_name])
    return None


def build_overlay_figure(
    display_rgb: np.ndarray,
    lux_map: np.ndarray,
    fixture_analysis: dict[str, Any] | None,
    under_fixture_lux: dict[str, float],
    between_fixture_lux: dict[str, float],
    measurement_masks: dict[str, np.ndarray] | None = None,
    warning_texts: list[str] | None = None,
):
    """Builds the inline/saveable notebook overlay."""

    figure, axis = plt.subplots(1, 1, figsize=(8, 5))
    axis.imshow(display_rgb)
    if float(np.max(lux_map, initial=0.0)) > 1e-6:
        axis.imshow(lux_map, cmap="inferno", alpha=0.38)

    height, width = display_rgb.shape[:2]
    warning_texts = warning_texts or []
    measurement_masks = measurement_masks or {}

    if fixture_analysis is not None:
        for region in fixture_analysis.get("between_regions", []):
            polygon_points = region.get("polygon") or []
            if polygon_points:
                polygon_pixels = [
                    (float(x_value) * (width - 1), float(y_value) * (height - 1))
                    for x_value, y_value in polygon_points
                ]
                axis.add_patch(
                    Polygon(
                        polygon_pixels,
                        closed=True,
                        facecolor="#80cbc4",
                        edgecolor="#004d40",
                        linewidth=1.2,
                        alpha=0.14,
                    )
                )

        for fixture in fixture_analysis.get("fixtures", []):
            x_pixel = float(fixture["x"]) * (width - 1)
            y_pixel = float(fixture["y"]) * (height - 1)
            axis.scatter(
                x_pixel,
                y_pixel,
                s=120,
                c="#ffee58",
                edgecolors="black",
                linewidths=0.9,
                zorder=3,
            )
            axis.text(
                x_pixel + 6,
                y_pixel - 6,
                str(fixture["name"]),
                color="white",
                fontsize=8,
                bbox={"facecolor": "black", "alpha": 0.45, "pad": 2},
            )

        for point in fixture_analysis.get("point_targets", []):
            point_name = str(point["name"])
            measurement_value = _find_measurement_value(
                point_name,
                under_fixture_lux=under_fixture_lux,
                between_fixture_lux=between_fixture_lux,
            )
            if measurement_value is None or not np.isfinite(measurement_value):
                continue
            x_pixel = float(point["x"]) * (width - 1)
            y_pixel = float(point["y"]) * (height - 1)
            point_color = "#4fc3f7" if point_name.startswith("between_") else "#ffffff"

            mask = measurement_masks.get(point_name)
            if mask is not None and np.any(mask):
                mask_alpha = np.zeros((height, width, 4), dtype=np.float32)
                rgba = np.array([0.20, 0.80, 1.00, 0.18], dtype=np.float32)
                if point_name.startswith("under_"):
                    rgba = np.array([1.00, 1.00, 1.00, 0.16], dtype=np.float32)
                mask_alpha[mask] = rgba
                axis.imshow(mask_alpha)

            axis.scatter(
                x_pixel,
                y_pixel,
                s=48,
                c=point_color,
                edgecolors="black",
                linewidths=0.8,
                zorder=4,
            )
            axis.text(
                x_pixel + 4,
                y_pixel + 10,
                f"floor {point_name}\n{measurement_value:.1f} lux",
                color="white",
                fontsize=7,
                bbox={"facecolor": "black", "alpha": 0.5, "pad": 2},
                zorder=5,
            )

    if warning_texts:
        axis.text(
            0.01,
            0.99,
            "\n".join(warning_texts),
            transform=axis.transAxes,
            ha="left",
            va="top",
            color="white",
            fontsize=8,
            bbox={"facecolor": "#5d4037", "alpha": 0.72, "pad": 4},
            zorder=6,
        )

    axis.set_title("Fixture-aware Floor Lux Overlay")
    axis.axis("off")
    figure.tight_layout()
    return figure


def save_result_artifacts(
    output_dir: Path,
    image_name: str,
    display_rgb: np.ndarray,
    lux_map: np.ndarray,
    fixture_analysis: dict[str, Any] | None,
    under_fixture_lux: dict[str, float],
    between_fixture_lux: dict[str, float],
    measurement_masks: dict[str, np.ndarray],
    result: dict[str, Any],
    warning_texts: list[str],
):
    """Saves the overlay and JSON summary for a notebook run."""

    figure = build_overlay_figure(
        display_rgb=display_rgb,
        lux_map=lux_map,
        fixture_analysis=fixture_analysis,
        under_fixture_lux=under_fixture_lux,
        between_fixture_lux=between_fixture_lux,
        measurement_masks=measurement_masks,
        warning_texts=warning_texts,
    )
    output_dir.mkdir(parents=True, exist_ok=True)
    image_stem = Path(image_name).stem
    overlay_path = output_dir / f"{image_stem}_fixture_lux_overlay.png"
    summary_path = output_dir / f"{image_stem}_fixture_lux_summary.json"
    figure.savefig(overlay_path, bbox_inches="tight")
    result["overlay_image"] = str(overlay_path)
    result["summary_json"] = str(summary_path)
    with summary_path.open("w", encoding="utf-8") as handle:
        json.dump(result, handle, indent=2)
    return figure


def _run_named_outputs(session, model_input: np.ndarray) -> dict[str, Any]:
    raw_outputs = session.run(None, {session.get_inputs()[0].name: model_input})
    session_output_names = [output.name for output in session.get_outputs()]
    if session_output_names and len(session_output_names) == len(raw_outputs):
        return {name: value for name, value in zip(session_output_names, raw_outputs)}
    return {name: value for name, value in zip(EXPECTED_ONNX_OUTPUT_NAMES, raw_outputs)}


def run_uploaded_photo(
    image_bytes: bytes,
    image_name: str,
    onnx_path: Path,
    output_dir: Path,
    max_fixture_count: int = 8,
    floor_area_m2: float = 12.0,
):
    """Runs ONNX inference plus fixture-aware post-processing for the notebook."""

    session = load_onnx_session(onnx_path)
    input_shape = session.get_inputs()[0].shape
    input_height = int(input_shape[2])
    input_width = int(input_shape[3])
    model_input, display_rgb, image_quality = preprocess_uploaded_image(
        image_bytes=image_bytes,
        input_height=input_height,
        input_width=input_width,
    )

    warning_texts: list[str] = []
    if bool(image_quality["is_dark_frame"]):
        warning_texts.append(
            "Frame is too dark for reliable lux inference. Returning 0 lux and no fixtures."
        )
        result = {
            "image_name": image_name,
            "onnx_path": str(onnx_path),
            "fixture_count": 0,
            "avg_lux": 0.0,
            "low_lux_p5": 0.0,
            "high_lux_p95": 0.0,
            "lux_map_summary": {"avg_lux": 0.0, "low_lux_p5": 0.0, "high_lux_p95": 0.0},
            "under_fixture_lux": {},
            "between_fixture_lux": {},
            "point_lux": {},
            "fixture_analysis": None,
            "warnings": warning_texts,
            "image_quality": image_quality,
            "inference_skipped": True,
        }
        zero_lux_map = np.zeros((input_height, input_width), dtype=np.float32)
        figure = save_result_artifacts(
            output_dir=output_dir,
            image_name=image_name,
            display_rgb=display_rgb,
            lux_map=zero_lux_map,
            fixture_analysis=None,
            under_fixture_lux={},
            between_fixture_lux={},
            measurement_masks={},
            result=result,
            warning_texts=warning_texts,
        )
        return result, figure

    named_outputs = _run_named_outputs(session, model_input=model_input)

    lux_map = extract_single_map(named_outputs.get("lux_map"))
    if lux_map is None or lux_map.ndim != 2:
        raise ValueError("Expected a single-channel lux map from the ONNX output.")

    floor_mask = extract_single_map(named_outputs.get("floor_mask_pred"))
    floor_mask_binary = None if floor_mask is None else (np.asarray(floor_mask) > 0.5)

    floor_mask_ratio = None
    if floor_mask_binary is not None:
        floor_mask_ratio = float(np.mean(floor_mask_binary))
        if floor_mask_ratio < 0.03:
            warning_texts.append(
                "Predicted floor area is very small in this frame, so floor measurements may be unreliable."
            )

    fixture_layout = infer_fixture_layout(
        image=display_rgb,
        floor_mask=floor_mask_binary,
        max_fixture_count=max_fixture_count,
        floor_area_m2=floor_area_m2,
    )
    fixture_analysis = None if fixture_layout is None else fixture_layout.to_summary_dict()

    under_fixture_lux, between_fixture_lux, measurement_masks = compute_floor_measurements(
        lux_map=lux_map,
        floor_mask=floor_mask_binary,
        fixture_analysis=fixture_analysis,
    )
    point_lux = {**under_fixture_lux, **between_fixture_lux}

    if fixture_analysis is None or int(fixture_analysis.get("inferred_fixture_count", 0)) == 0:
        warning_texts.append(
            "No fixtures were found in this frame. Use an end-to-end hallway image with visible floor "
            "and increase Max fixtures if needed."
        )
    elif int(fixture_analysis.get("inferred_fixture_count", 0)) >= int(max_fixture_count):
        warning_texts.append(
            "Detected fixture count reached the Max fixtures cap. Increase that value if your hallway has more fixtures."
        )

    lux_summary = summarize_lux_map(lux_map=lux_map, floor_mask=floor_mask_binary)
    result = {
        "image_name": image_name,
        "onnx_path": str(onnx_path),
        "fixture_count": 0 if fixture_analysis is None else int(fixture_analysis.get("inferred_fixture_count", 0)),
        "avg_lux": extract_scalar(named_outputs.get("avg_lux"), fallback=lux_summary["avg_lux"]),
        "low_lux_p5": extract_scalar(named_outputs.get("low_lux_p5"), fallback=lux_summary["low_lux_p5"]),
        "high_lux_p95": extract_scalar(
            named_outputs.get("high_lux_p95"),
            fallback=lux_summary["high_lux_p95"],
        ),
        "lux_map_summary": lux_summary,
        "under_fixture_lux": under_fixture_lux,
        "between_fixture_lux": between_fixture_lux,
        "point_lux": point_lux,
        "fixture_analysis": fixture_analysis,
        "warnings": warning_texts,
        "image_quality": image_quality,
        "inference_skipped": False,
        "floor_mask_ratio": floor_mask_ratio,
    }
    figure = save_result_artifacts(
        output_dir=output_dir,
        image_name=image_name,
        display_rgb=display_rgb,
        lux_map=lux_map,
        fixture_analysis=fixture_analysis,
        under_fixture_lux=under_fixture_lux,
        between_fixture_lux=between_fixture_lux,
        measurement_masks=measurement_masks,
        result=result,
        warning_texts=warning_texts,
    )
    return result, figure

