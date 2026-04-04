#!/usr/bin/env python3
"""
Raspberry Pi inference script for hallway lighting model.
Run this on your Raspberry Pi with a camera attached.
"""

import cv2
import numpy as np
import onnxruntime as ort
from pathlib import Path
import sys

REPO_ROOT = Path(__file__).resolve().parent
PACKAGE_ROOT = REPO_ROOT / "hallway_lighting"
if PACKAGE_ROOT.exists():
    sys.path.insert(0, str(PACKAGE_ROOT))

try:
    from hallway_lighting.utils.fixture_detection import infer_fixture_layout
except Exception:
    infer_fixture_layout = None

IMAGENET_MEAN = np.asarray([0.485, 0.456, 0.406], dtype=np.float32).reshape(1, 1, 3)
IMAGENET_STD = np.asarray([0.229, 0.224, 0.225], dtype=np.float32).reshape(1, 1, 3)
DARK_FRAME_MEAN_THRESHOLD = 0.03
DARK_FRAME_P95_THRESHOLD = 0.08

# Load the ONNX model
model_path = '/home/jonah/models/hallway_multitask_unet_drive_prototype.onnx'
session = ort.InferenceSession(model_path)
input_shape = session.get_inputs()[0].shape
MODEL_HEIGHT = int(input_shape[2])
MODEL_WIDTH = int(input_shape[3])

# Camera setup (adjust for your camera)
cap = cv2.VideoCapture(0)  # 0 for default camera

def assess_frame_quality(display_rgb: np.ndarray) -> dict[str, float | bool]:
    """Computes simple brightness checks before trusting model outputs."""

    luminance = 0.2126 * display_rgb[..., 0] + 0.7152 * display_rgb[..., 1] + 0.0722 * display_rgb[..., 2]
    mean_luminance = float(np.mean(luminance))
    p95_luminance = float(np.percentile(luminance, 95))
    return {
        "mean_luminance": mean_luminance,
        "p95_luminance": p95_luminance,
        "is_dark_frame": bool(
            mean_luminance < DARK_FRAME_MEAN_THRESHOLD
            and p95_luminance < DARK_FRAME_P95_THRESHOLD
        ),
    }

def preprocess_image(image: np.ndarray, target_size=(256, 256)):
    """Preprocess image for model input."""
    # Convert BGR to RGB
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    # Resize
    image = cv2.resize(image, target_size)
    display_rgb = image.astype(np.float32) / 255.0
    # Match training-time ImageNet normalization.
    image = (display_rgb - IMAGENET_MEAN) / IMAGENET_STD
    # Transpose to CHW
    image = np.transpose(image, (2, 0, 1))
    # Add batch dimension
    image = np.expand_dims(image, 0)
    return image, display_rgb

def run_inference(image: np.ndarray):
    """Run model inference."""
    inputs = {session.get_inputs()[0].name: image}
    outputs = session.run(None, inputs)
    return {
        'lux_map': outputs[0],
        'avg_lux': outputs[1],
        'low_lux_p5': outputs[2],
        'high_lux_p95': outputs[3],
        'floor_mask_pred': outputs[4],
        'albedo_pred': outputs[5],
        'gloss_pred': outputs[6],
        'uncertainty_map': outputs[7],
        'estimated_power_w': outputs[8],
    }

# Main loop
while True:
    ret, frame = cap.read()
    if not ret:
        break

    # Preprocess
    target_size = (MODEL_WIDTH, MODEL_HEIGHT)
    processed, display_rgb = preprocess_image(frame, target_size)
    quality = assess_frame_quality(display_rgb)

    if quality["is_dark_frame"]:
        print(
            "Average Lux: 0.00 | Frame too dark for reliable inference "
            f"(mean={quality['mean_luminance']:.3f}, p95={quality['p95_luminance']:.3f})"
        )
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
        continue

    # Run inference
    results = run_inference(processed)

    # Display results (example: average lux)
    avg_lux = float(results['avg_lux'].flatten()[0])
    low_lux_p5 = float(results['low_lux_p5'].flatten()[0])
    high_lux_p95 = float(results['high_lux_p95'].flatten()[0])
    line = f"Average Lux: {avg_lux:.2f} | Low Lux P5: {low_lux_p5:.2f} | High Lux P95: {high_lux_p95:.2f}"

    if infer_fixture_layout is not None:
        floor_mask = results['floor_mask_pred']
        if isinstance(floor_mask, np.ndarray) and floor_mask.ndim >= 4:
            floor_mask = floor_mask[0, 0]
        layout = infer_fixture_layout(
            image=display_rgb,
            floor_mask=floor_mask,
            max_fixture_count=8,
        )
        if layout is not None:
            fixture_locations = ", ".join(
                f"{fixture.name}@({fixture.x:.2f},{fixture.y:.2f})"
                for fixture in layout.fixtures
            )
            line += f" | Fixtures: {len(layout.fixtures)} [{fixture_locations}]"
    print(line)

    # Optional: Save results or display visualizations
    # ...

    # Break on 'q' key
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
