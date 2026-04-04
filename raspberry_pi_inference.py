#!/usr/bin/env python3
"""
Raspberry Pi inference script for hallway lighting model.
Run this on your Raspberry Pi with a camera attached.
"""

import cv2
import numpy as np
import onnxruntime as ort
from PIL import Image
import json
from pathlib import Path

# Load the ONNX model
model_path = '/home/jonah/models/hallway_multitask_unet_drive_prototype.onnx'
session = ort.InferenceSession(model_path)

# Camera setup (adjust for your camera)
cap = cv2.VideoCapture(0)  # 0 for default camera

def preprocess_image(image: np.ndarray, target_size=(256, 256)):
    """Preprocess image for model input."""
    # Convert BGR to RGB
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    # Resize
    image = cv2.resize(image, target_size)
    # Normalize to [0,1]
    image = image.astype(np.float32) / 255.0
    # Transpose to CHW
    image = np.transpose(image, (2, 0, 1))
    # Add batch dimension
    image = np.expand_dims(image, 0)
    return image

def run_inference(image: np.ndarray):
    """Run model inference."""
    inputs = {session.get_inputs()[0].name: image}
    outputs = session.run(None, inputs)
    return {
        'lux_map': outputs[0],
        'avg_lux': outputs[1],
        'low_lux_p5': outputs[2],
        'high_lux_p5': outputs[3],
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
    processed = preprocess_image(frame, (192,192))

    # Run inference
    results = run_inference(processed)

    # Display results (example: average lux)
    avg_lux = float(results['avg_lux'][0])
    print(f"Average Lux: {avg_lux:.2f}")

    # Optional: Save results or display visualizations
    # ...

    # Break on 'q' key
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()