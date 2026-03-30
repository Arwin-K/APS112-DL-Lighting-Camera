# Hallway Lighting

Colab-first PyTorch project for hallway floor-plane illuminance estimation with auxiliary appearance, intrinsic-image, segmentation, point-wise reporting, and carbon estimation outputs.

## What This Project Does

This repository is designed around a single notebook-first workflow for training and evaluating a hallway lighting model that can eventually estimate:

- average floor-plane illuminance (`avg_lux`)
- lower-end illuminance distribution (`low_lux_p5`)
- upper-end illuminance distribution (`high_lux_p95`)
- point-wise illuminance under fixtures
- point-wise illuminance between fixtures
- estimated lighting power
- interval energy use
- interval carbon emissions

The main learning target is hallway illuminance. Public indoor datasets act as auxiliary priors for geometry, intrinsic decomposition, and lighting appearance. A custom hallway dataset is optional, but it is the expected path to the best hallway-specific performance.

## Supported Datasets

The default workflow only assumes these public datasets:

- NYU Depth V2
- MIT Intrinsic Images
- MID Intrinsics
- Fast Spatially-Varying Indoor Lighting Estimation dataset
- optional custom hallway dataset

No unavailable or gated dataset is part of the default project structure.

## Notebook-First Workflow

The main user interface is:

- [`notebooks/hallway_illuminance_train_eval_all_in_one.ipynb`](/Users/ArwinKarir/Desktop/APS112-DL-Lighting-Camera/hallway_lighting/notebooks/hallway_illuminance_train_eval_all_in_one.ipynb)

The notebook is intended to be run in Google Colab. The workflow is:

1. Install dependencies from `requirements.txt`.
2. Mount Google Drive.
3. Point the configs or notebook variables at dataset locations in Drive.
4. Extract archives if needed.
5. Generate or validate manifests.
6. Visualize sample data.
7. Configure the model and training loop.
8. Train, validate, and evaluate.
9. Report point-wise lux and carbon-related outputs.
10. Save checkpoints and export ONNX for deployment experiments.

## Google Drive Path Strategy

Dataset paths are deliberately not hardcoded.

The intended Colab usage is:

- keep the repository in Drive or clone it into the Colab runtime
- place datasets anywhere under Drive
- set dataset roots through YAML configs or notebook variables
- save outputs under `runs/`

Example Drive-style paths the user will eventually provide:

```python
DATASET_ROOTS = {
    "nyu_depth_v2": "/content/drive/MyDrive/datasets/nyu_depth_v2",
    "mit_intrinsic_images": "/content/drive/MyDrive/datasets/mit_intrinsic_images",
    "mid_intrinsics": "/content/drive/MyDrive/datasets/mid_intrinsics",
    "fast_sv_indoor_lighting": "/content/drive/MyDrive/datasets/fast_sv_indoor_lighting",
    "custom_hallway": "/content/drive/MyDrive/datasets/custom_hallway",
}
```

## Repository Layout

```text
hallway_lighting/
  README.md
  requirements.txt
  pyproject.toml
  AGENTS.md
  configs/
  hallway_lighting/
  notebooks/
  templates/
  runs/
```

## Current Status

This initial scaffold includes:

- a real importable Python package
- path-driven YAML configs
- dataset registry and manifest helpers
- model, losses, metrics, and inference scaffolding
- a non-empty Colab-friendly notebook with the required sections
- templates for custom hallway manifests and point-wise targets

Several modules intentionally include targeted TODO sections where dataset-specific training behavior still needs to be implemented. Those TODOs state the intended next behavior rather than leaving empty placeholders.

## Intended Outputs

The project is structured to eventually report:

- `avg_lux`
- `low_lux_p5`
- `high_lux_p95`
- `point_lux_under_fixture`
- `point_lux_between_fixtures`
- `estimated_power_w`
- `interval_energy_kwh`
- `interval_carbon_kg`

## Quick Start

From Colab or a local environment:

```bash
pip install -r requirements.txt
```

Then open the notebook and follow the sections in order.

## Implementation Notes

- Code uses Python type hints.
- Modules include docstrings and student-readable comments.
- Carbon is derived from estimated or measured electricity use, not directly sensed.
- Public datasets are auxiliary priors. Custom hallway supervision is optional but expected to improve hallway-specific performance.
