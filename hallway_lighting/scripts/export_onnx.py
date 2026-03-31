"""Export a trained hallway lighting checkpoint to ONNX."""

from __future__ import annotations

import argparse
from pathlib import Path

from hallway_lighting.infer import ONNX_OUTPUT_NAMES, export_model_to_onnx, load_model_from_checkpoint
from hallway_lighting.utils.io import save_json, load_yaml


def parse_args() -> argparse.Namespace:
    """Builds the command-line interface for ONNX export."""

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--base-config",
        default="configs/base.yaml",
        help="Path to the base YAML config containing the model section.",
    )
    parser.add_argument(
        "--infer-config",
        default="configs/infer.yaml",
        help="Path to the inference YAML config.",
    )
    parser.add_argument(
        "--checkpoint",
        default="",
        help="Trained checkpoint path. Defaults to inference.checkpoint_path in infer config.",
    )
    parser.add_argument(
        "--output",
        default="",
        help="ONNX output path. Defaults to inference.export_onnx_path in infer config.",
    )
    parser.add_argument(
        "--device",
        default="cpu",
        help="Export device. Use cpu for deployment-oriented exports unless CUDA is required.",
    )
    parser.add_argument(
        "--height",
        type=int,
        default=0,
        help="Override input height. Defaults to infer config.",
    )
    parser.add_argument(
        "--width",
        type=int,
        default=0,
        help="Override input width. Defaults to infer config.",
    )
    parser.add_argument(
        "--opset",
        type=int,
        default=0,
        help="Override ONNX opset version. Defaults to infer config or 17.",
    )
    return parser.parse_args()


def main() -> None:
    """Loads configs, restores a checkpoint, and exports an ONNX model."""

    args = parse_args()
    project_root = Path(__file__).resolve().parents[1]

    base_config = load_yaml(project_root / args.base_config)
    infer_config = load_yaml(project_root / args.infer_config)
    inference_settings = infer_config.get("inference", {})

    checkpoint_path = Path(args.checkpoint or inference_settings.get("checkpoint_path", ""))
    if not checkpoint_path.is_absolute():
        checkpoint_path = project_root / checkpoint_path
    if not checkpoint_path.exists():
        raise FileNotFoundError(
            "Checkpoint path is required for export. Set --checkpoint or inference.checkpoint_path."
        )

    output_path = Path(args.output or inference_settings.get("export_onnx_path", ""))
    if not output_path.is_absolute():
        output_path = project_root / output_path
    if not str(output_path):
        raise ValueError("Output path is required. Set --output or inference.export_onnx_path.")

    height = int(args.height or inference_settings.get("image_size", {}).get("height", 256))
    width = int(args.width or inference_settings.get("image_size", {}).get("width", 256))
    opset = int(args.opset or inference_settings.get("opset_version", 17))

    model = load_model_from_checkpoint(
        checkpoint_path=checkpoint_path,
        model_config=base_config["model"],
        device=args.device,
    )
    exported_path = export_model_to_onnx(
        model=model,
        output_path=output_path,
        image_size=(height, width),
        device=args.device,
        opset_version=opset,
    )

    metadata_path = output_path.with_suffix(".metadata.json")
    save_json(
        {
            "checkpoint_path": str(checkpoint_path),
            "onnx_path": str(exported_path),
            "input_shape": [1, 3, height, width],
            "input_names": ["image"],
            "output_names": ONNX_OUTPUT_NAMES,
            "opset_version": opset,
            "device_used_for_export": args.device,
        },
        metadata_path,
    )

    print(f"Exported ONNX model: {exported_path}")
    print(f"Saved export metadata: {metadata_path}")


if __name__ == "__main__":
    main()
