"""Inference and ONNX export helpers for hallway lighting estimation."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
from PIL import Image
import torch

from hallway_lighting.data.point_sampling import (
    PointTarget,
    default_hallway_points,
    load_point_targets,
    sample_values_at_points,
)
from hallway_lighting.data.transforms import build_image_transform
from hallway_lighting.models.hallway_multitask_unet import HallwayMultitaskUNet
from hallway_lighting.utils.carbon import (
    estimate_interval_carbon_kg,
    estimate_interval_energy_kwh,
    estimate_power_from_lux,
)
from hallway_lighting.utils.fixture_detection import infer_fixture_layout
from hallway_lighting.utils.io import ensure_dir, save_json
from hallway_lighting.utils.metrics import summarize_lux_map
from hallway_lighting.utils.visualization import (
    save_fixture_layout_visualization,
    save_heatmap_image,
    save_overlay_visualization,
    save_point_annotation_visualization,
    save_prediction_figure,
)

ONNX_OUTPUT_NAMES = [
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


@dataclass
class InferenceBatch:
    """Packaged single-image input tensors plus display metadata."""

    image_tensor: torch.Tensor
    image_path: str
    resized_rgb: np.ndarray


@dataclass
class InferenceArtifacts:
    """Paths to saved single-image inference artifacts."""

    output_dir: str
    summary_json: str | None = None
    lux_heatmap_image: str | None = None
    overlay_image: str | None = None
    point_overlay_image: str | None = None
    fixture_layout_image: str | None = None
    prediction_overview_image: str | None = None


@dataclass
class InferenceOutput:
    """User-facing single-image inference result."""

    backend: str
    image_path: str
    avg_lux: float
    low_lux_p5: float
    high_lux_p95: float
    point_lux: dict[str, float]
    estimated_power_w: float
    interval_energy_kwh: float
    interval_carbon_kg: float
    lux_map_summary: dict[str, float]
    floor_mask_available: bool
    albedo_available: bool
    gloss_available: bool
    uncertainty_available: bool
    artifacts: InferenceArtifacts
    point_targets: list[dict[str, Any]]
    fixture_analysis: dict[str, Any] | None
    raw_outputs: dict[str, np.ndarray | float | None]

    def to_summary_dict(self) -> dict[str, Any]:
        """Returns a JSON-serializable summary of the inference result."""

        return {
            "backend": self.backend,
            "image_path": self.image_path,
            "avg_lux": self.avg_lux,
            "low_lux_p5": self.low_lux_p5,
            "high_lux_p95": self.high_lux_p95,
            "point_lux": self.point_lux,
            "estimated_power_w": self.estimated_power_w,
            "interval_energy_kwh": self.interval_energy_kwh,
            "interval_carbon_kg": self.interval_carbon_kg,
            "lux_map_summary": self.lux_map_summary,
            "floor_mask_available": self.floor_mask_available,
            "albedo_available": self.albedo_available,
            "gloss_available": self.gloss_available,
            "uncertainty_available": self.uncertainty_available,
            "artifacts": asdict(self.artifacts),
            "point_targets": self.point_targets,
            "fixture_analysis": self.fixture_analysis,
        }


class OnnxExportWrapper(torch.nn.Module):
    """Tensor-only wrapper used for ONNX export."""

    def __init__(self, wrapped_model: torch.nn.Module) -> None:
        super().__init__()
        self.wrapped_model = wrapped_model

    def forward(self, image: torch.Tensor) -> tuple[torch.Tensor, ...]:
        """Returns a fixed tuple of tensor outputs for ONNX export."""

        outputs = self.wrapped_model(image)
        return tuple(outputs[name] for name in ONNX_OUTPUT_NAMES)


def _resolve_device(device: str | torch.device) -> torch.device:
    """Returns a valid device, falling back to CPU when CUDA is unavailable."""

    device = torch.device(device)
    if device.type == "cuda" and not torch.cuda.is_available():
        return torch.device("cpu")
    return device


def _to_numpy(value: torch.Tensor | np.ndarray | float | int | None) -> np.ndarray | float | None:
    """Converts tensors to NumPy while preserving scalars."""

    if value is None:
        return None
    if isinstance(value, torch.Tensor):
        value = value.detach().cpu().numpy()
    if isinstance(value, np.ndarray):
        if value.size == 1:
            return float(value.reshape(-1)[0])
        return value
    if isinstance(value, (float, int)):
        return float(value)
    raise TypeError(f"Unsupported output type: {type(value)!r}")


def _extract_single_map(value: torch.Tensor | np.ndarray | float | int | None) -> np.ndarray | None:
    """Converts a model output to a single-image HxW or HxWxC NumPy array."""

    array = _to_numpy(value)
    if array is None or isinstance(array, float):
        return None

    array = np.asarray(array)
    if array.ndim == 4:
        array = array[0]
    if array.ndim == 3 and array.shape[0] in {1, 3}:
        array = np.transpose(array, (1, 2, 0))
    if array.ndim == 3 and array.shape[-1] == 1:
        array = array[..., 0]
    if array.ndim not in {2, 3}:
        raise ValueError(f"Expected a 2D or 3D prediction map, got shape {tuple(array.shape)}")
    return array.astype(np.float32, copy=False)


def _extract_scalar(value: torch.Tensor | np.ndarray | float | int | None) -> float | None:
    """Converts a model output to a Python float when possible."""

    scalar = _to_numpy(value)
    if scalar is None:
        return None
    if isinstance(scalar, np.ndarray):
        return float(np.asarray(scalar).reshape(-1)[0])
    return float(scalar)


def _resolve_point_targets(
    point_targets: Sequence[PointTarget] | None,
    point_targets_path: str | Path | None,
    fixture_count: int,
) -> list[PointTarget]:
    """Resolves coordinate-based point definitions for hallway reporting."""

    if point_targets is not None:
        return list(point_targets)
    if point_targets_path:
        return load_point_targets(point_targets_path)
    return default_hallway_points(fixture_count=fixture_count)


def preprocess_single_image(
    image_path: str | Path,
    image_size: tuple[int, int] = (256, 256),
) -> InferenceBatch:
    """Loads and preprocesses a single RGB image for model inference."""

    image_path = Path(image_path)
    if not image_path.exists():
        raise FileNotFoundError(f"Image does not exist: {image_path}")

    transform = build_image_transform(image_size=image_size)
    image = Image.open(image_path).convert("RGB")
    resized = image.resize((image_size[1], image_size[0]), resample=Image.BILINEAR)
    image_tensor = transform(image).unsqueeze(0)
    return InferenceBatch(
        image_tensor=image_tensor,
        image_path=str(image_path),
        resized_rgb=np.asarray(resized).astype(np.float32) / 255.0,
    )


def build_model_from_config(model_config: Mapping[str, Any]) -> HallwayMultitaskUNet:
    """Instantiates the hallway multitask model from a config mapping."""

    if "model" in model_config and isinstance(model_config["model"], Mapping):
        model_config = model_config["model"]
    return HallwayMultitaskUNet(dict(model_config))


def load_model_from_checkpoint(
    checkpoint_path: str | Path,
    model_config: Mapping[str, Any],
    device: str | torch.device = "cpu",
) -> HallwayMultitaskUNet:
    """Loads a trained PyTorch model checkpoint for inference or export."""

    checkpoint_path = Path(checkpoint_path)
    if not checkpoint_path.exists():
        raise FileNotFoundError(f"Checkpoint does not exist: {checkpoint_path}")

    device_obj = _resolve_device(device)
    model = build_model_from_config(model_config).to(device_obj)
    checkpoint = torch.load(checkpoint_path, map_location=device_obj)
    if "model_state_dict" not in checkpoint:
        raise KeyError(f"Checkpoint is missing 'model_state_dict': {checkpoint_path}")
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()
    return model


def load_onnx_session(
    onnx_path: str | Path,
    providers: Sequence[str] | None = None,
):
    """Builds an ONNX Runtime session for later deployment-oriented inference."""

    try:
        import onnxruntime as ort
    except ImportError as exc:
        raise ImportError(
            "onnxruntime is required for ONNX inference. Install requirements.txt first."
        ) from exc

    onnx_path = Path(onnx_path)
    if not onnx_path.exists():
        raise FileNotFoundError(f"ONNX model does not exist: {onnx_path}")

    runtime_providers = list(providers or ["CPUExecutionProvider"])
    return ort.InferenceSession(str(onnx_path), providers=runtime_providers)


def export_model_to_onnx(
    model: torch.nn.Module,
    output_path: str | Path,
    image_size: tuple[int, int] = (256, 256),
    device: str | torch.device = "cpu",
    opset_version: int = 17,
) -> Path:
    """Exports the multitask model to ONNX for later edge deployment."""

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    device_obj = _resolve_device(device)
    model = model.to(device_obj)
    model.eval()
    dummy_input = torch.randn(1, 3, image_size[0], image_size[1], device=device_obj)

    torch.onnx.export(
        OnnxExportWrapper(model),
        dummy_input,
        output_path,
        input_names=["image"],
        output_names=ONNX_OUTPUT_NAMES,
        opset_version=opset_version,
        dynamic_axes={
            "image": {0: "batch"},
            **{name: {0: "batch"} for name in ONNX_OUTPUT_NAMES},
        },
        do_constant_folding=True,
    )
    return output_path


def _run_pytorch_forward(
    model: torch.nn.Module,
    image_tensor: torch.Tensor,
    device: str | torch.device,
) -> dict[str, np.ndarray | float]:
    """Runs one PyTorch forward pass and normalizes outputs to NumPy or float."""

    device_obj = _resolve_device(device)
    model = model.to(device_obj)
    model.eval()
    with torch.no_grad():
        outputs = model(image_tensor.to(device_obj))

    normalized_outputs: dict[str, np.ndarray | float] = {}
    for key, value in outputs.items():
        if key == "point_targets":
            continue
        normalized = _to_numpy(value)
        if normalized is not None:
            normalized_outputs[key] = normalized
    return normalized_outputs


def _run_onnx_forward(
    session: Any,
    image_tensor: torch.Tensor,
) -> dict[str, np.ndarray | float]:
    """Runs one ONNX Runtime forward pass and returns named outputs."""

    input_name = session.get_inputs()[0].name
    raw_outputs = session.run(None, {input_name: image_tensor.detach().cpu().numpy()})
    return {
        output_name: output_value
        for output_name, output_value in zip(ONNX_OUTPUT_NAMES, raw_outputs)
    }


def _choose_floor_mask_for_summary(
    floor_mask_pred: np.ndarray | None,
) -> np.ndarray | None:
    """Returns a usable binary floor mask when one is available."""

    if floor_mask_pred is None:
        return None
    binary_mask = floor_mask_pred > 0.5
    if float(binary_mask.mean()) <= 0.001:
        return None
    return binary_mask


def _build_inference_output(
    backend: str,
    batch: InferenceBatch,
    raw_outputs: Mapping[str, np.ndarray | float],
    point_targets: Sequence[PointTarget],
    auto_detect_fixtures: bool,
    manual_point_targets_supplied: bool,
    max_fixture_count: int,
    output_dir: str | Path | None,
    save_outputs: bool,
    save_point_visualization: bool,
    floor_area_m2: float,
    watts_per_lux_m2: float,
    carbon_factor_kg_per_kwh: float,
    interval_hours: float,
) -> InferenceOutput:
    """Builds user-facing summaries and saves visual artifacts when requested."""

    lux_map = _extract_single_map(raw_outputs.get("lux_map"))
    if lux_map is None:
        raise KeyError("Inference outputs are missing 'lux_map'.")

    floor_mask_pred = _extract_single_map(raw_outputs.get("floor_mask_pred"))
    floor_mask_for_summary = _choose_floor_mask_for_summary(floor_mask_pred)
    lux_map_summary = summarize_lux_map(lux_map, floor_mask=floor_mask_for_summary)

    avg_lux = _extract_scalar(raw_outputs.get("avg_lux"))
    if avg_lux is None:
        avg_lux = lux_map_summary["avg_lux"]
    low_lux_p5 = _extract_scalar(raw_outputs.get("low_lux_p5"))
    if low_lux_p5 is None:
        low_lux_p5 = lux_map_summary["low_lux_p5"]
    high_lux_p95 = _extract_scalar(raw_outputs.get("high_lux_p95"))
    if high_lux_p95 is None:
        high_lux_p95 = lux_map_summary["high_lux_p95"]
    estimated_power_w = _extract_scalar(raw_outputs.get("estimated_power_w"))
    if estimated_power_w is None:
        estimated_power_w = estimate_power_from_lux(
            avg_lux=avg_lux,
            floor_area_m2=floor_area_m2,
            watts_per_lux_m2=watts_per_lux_m2,
        )

    fixture_analysis: dict[str, Any] | None = None
    effective_point_targets = list(point_targets)
    if auto_detect_fixtures:
        detected_layout = infer_fixture_layout(
            image=batch.resized_rgb,
            floor_mask=floor_mask_pred,
            max_fixture_count=max_fixture_count,
            floor_area_m2=floor_area_m2,
        )
        if detected_layout is not None:
            fixture_analysis = detected_layout.to_summary_dict()
            fixture_analysis["used_for_point_sampling"] = not manual_point_targets_supplied
            if not manual_point_targets_supplied and detected_layout.point_targets:
                effective_point_targets = list(detected_layout.point_targets)

    point_lux = sample_values_at_points(lux_map, effective_point_targets)
    interval_energy_kwh = estimate_interval_energy_kwh(
        power_w=estimated_power_w,
        interval_hours=interval_hours,
    )
    interval_carbon_kg = estimate_interval_carbon_kg(
        energy_kwh=interval_energy_kwh,
        carbon_factor_kg_per_kwh=carbon_factor_kg_per_kwh,
    )

    albedo_pred = _extract_single_map(raw_outputs.get("albedo_pred"))
    gloss_pred = _extract_single_map(raw_outputs.get("gloss_pred"))
    uncertainty_map = _extract_single_map(raw_outputs.get("uncertainty_map"))

    artifact_payload = InferenceArtifacts(output_dir=str(output_dir) if output_dir else "")
    point_targets_payload = [
        {"name": point.name, "x": point.x, "y": point.y, "group": point.group}
        for point in effective_point_targets
    ]

    result = InferenceOutput(
        backend=backend,
        image_path=batch.image_path,
        avg_lux=avg_lux,
        low_lux_p5=low_lux_p5,
        high_lux_p95=high_lux_p95,
        point_lux=point_lux,
        estimated_power_w=estimated_power_w,
        interval_energy_kwh=interval_energy_kwh,
        interval_carbon_kg=interval_carbon_kg,
        lux_map_summary=lux_map_summary,
        floor_mask_available=floor_mask_pred is not None,
        albedo_available=albedo_pred is not None,
        gloss_available=gloss_pred is not None,
        uncertainty_available=uncertainty_map is not None,
        artifacts=artifact_payload,
        point_targets=point_targets_payload,
        fixture_analysis=fixture_analysis,
        raw_outputs={
            "lux_map": lux_map,
            "floor_mask_pred": floor_mask_pred if floor_mask_pred is not None else None,
            "albedo_pred": albedo_pred if albedo_pred is not None else None,
            "gloss_pred": gloss_pred if gloss_pred is not None else None,
            "uncertainty_map": uncertainty_map if uncertainty_map is not None else None,
            "avg_lux": avg_lux,
            "low_lux_p5": low_lux_p5,
            "high_lux_p95": high_lux_p95,
            "estimated_power_w": estimated_power_w,
        },
    )

    if save_outputs and output_dir is not None:
        output_dir = ensure_dir(output_dir)
        image_stem = Path(batch.image_path).stem

        heatmap_path = output_dir / f"{image_stem}_lux_heatmap.png"
        overlay_path = output_dir / f"{image_stem}_lux_overlay.png"
        point_overlay_path = output_dir / f"{image_stem}_point_overlay.png"
        fixture_layout_path = output_dir / f"{image_stem}_fixture_layout.png"
        overview_path = output_dir / f"{image_stem}_prediction_overview.png"
        summary_path = output_dir / f"{image_stem}_summary.json"

        save_heatmap_image(heatmap_path, lux_map)
        save_overlay_visualization(overlay_path, batch.resized_rgb, lux_map)
        if save_point_visualization:
            save_point_annotation_visualization(
                point_overlay_path,
                lux_map,
                effective_point_targets,
                point_values=point_lux,
                title="Hallway Point-wise Lux",
            )
        if fixture_analysis is not None:
            save_fixture_layout_visualization(
                fixture_layout_path,
                batch.resized_rgb,
                fixtures=fixture_analysis.get("fixtures", []),
                between_regions=fixture_analysis.get("between_regions", []),
                title="Detected Fixture Layout",
            )
        save_prediction_figure(
            overview_path,
            image=batch.resized_rgb,
            lux_map=lux_map,
            floor_mask_pred=floor_mask_pred,
            albedo_pred=albedo_pred,
            gloss_pred=gloss_pred,
            points=effective_point_targets,
            point_values=point_lux,
            title="Single-image Inference Overview",
        )

        result.artifacts.lux_heatmap_image = str(heatmap_path)
        result.artifacts.overlay_image = str(overlay_path)
        result.artifacts.prediction_overview_image = str(overview_path)
        if save_point_visualization:
            result.artifacts.point_overlay_image = str(point_overlay_path)
        if fixture_analysis is not None:
            result.artifacts.fixture_layout_image = str(fixture_layout_path)
        result.artifacts.summary_json = str(summary_path)
        save_json(result.to_summary_dict(), summary_path)

    return result


def run_single_image_inference(
    image_path: str | Path,
    model: torch.nn.Module | None = None,
    checkpoint_path: str | Path | None = None,
    model_config: Mapping[str, Any] | None = None,
    onnx_path: str | Path | None = None,
    onnx_session: Any | None = None,
    device: str | torch.device = "cpu",
    image_size: tuple[int, int] = (256, 256),
    point_targets: Sequence[PointTarget] | None = None,
    point_targets_path: str | Path | None = None,
    fixture_count: int = 3,
    auto_detect_fixtures: bool = True,
    max_fixture_count: int = 8,
    output_dir: str | Path | None = None,
    save_outputs: bool = True,
    save_point_visualization: bool = True,
    floor_area_m2: float = 12.0,
    watts_per_lux_m2: float = 0.015,
    carbon_factor_kg_per_kwh: float = 0.35,
    interval_hours: float = 1.0,
) -> InferenceOutput:
    """Runs single-image inference with a PyTorch checkpoint or ONNX model.

    Notes:
    - `point_targets` expects coordinate definitions for sampling.
    - `point_targets_path` should point to a coordinate-based JSON file understood
      by `load_point_targets(...)`.
    - When `auto_detect_fixtures` is enabled and no manual point targets are
      supplied, the helper will attempt to infer fixture count and locations
      directly from the image and use those detected floor projections.
    - The custom hallway dataset's flat point-value JSON is a different file type
      used for supervision, not inference-time sampling coordinates.
    """

    batch = preprocess_single_image(image_path=image_path, image_size=image_size)
    manual_point_targets_supplied = point_targets is not None or bool(point_targets_path)
    resolved_points = _resolve_point_targets(
        point_targets=point_targets,
        point_targets_path=point_targets_path,
        fixture_count=fixture_count,
    )

    backend = "pytorch"
    if onnx_session is not None or onnx_path is not None:
        backend = "onnx"

    if backend == "onnx":
        session = onnx_session or load_onnx_session(onnx_path)
        raw_outputs = _run_onnx_forward(session, batch.image_tensor)
    else:
        if model is None:
            if checkpoint_path is None or model_config is None:
                raise ValueError(
                    "PyTorch inference requires either a loaded model or both checkpoint_path and model_config."
                )
            model = load_model_from_checkpoint(
                checkpoint_path=checkpoint_path,
                model_config=model_config,
                device=device,
            )
        raw_outputs = _run_pytorch_forward(model, batch.image_tensor, device=device)

    return _build_inference_output(
        backend=backend,
        batch=batch,
        raw_outputs=raw_outputs,
        point_targets=resolved_points,
        auto_detect_fixtures=auto_detect_fixtures,
        manual_point_targets_supplied=manual_point_targets_supplied,
        max_fixture_count=max_fixture_count,
        output_dir=output_dir,
        save_outputs=save_outputs,
        save_point_visualization=save_point_visualization,
        floor_area_m2=floor_area_m2,
        watts_per_lux_m2=watts_per_lux_m2,
        carbon_factor_kg_per_kwh=carbon_factor_kg_per_kwh,
        interval_hours=interval_hours,
    )
