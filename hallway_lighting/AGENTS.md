# Repository Rules

## Core Rules

- Never hardcode dataset paths.
- Keep the notebook under `notebooks/` as the main user interface.
- Do not invent labels, targets, or annotations that are not present in source data.
- Prefer readable, student-friendly code over clever abstractions.
- Save outputs, checkpoints, visualizations, inference artifacts, and exports under `runs/`.
- Carbon must be derived from electricity use, not directly sensed.
- Public datasets are auxiliary priors. Custom hallway labels are optional but are the main route to stronger hallway-specific performance.

## Notebook-First Workflow

- The primary user flow must remain runnable from the notebook.
- New helpers should make notebook cells shorter and clearer, not replace the notebook workflow.
- Inference, checkpoint loading, and ONNX export should be callable from both notebook cells and importable Python modules.
- Keep notebook cells understandable for a student user running Colab with Google Drive paths.

## Engineering Expectations

- Keep configuration path-driven through YAML files and notebook variables.
- Add type hints and docstrings to new Python code.
- Fail loudly on malformed inputs instead of silently guessing.
- Preserve importable package boundaries so notebook cells can call package modules directly.
- Prefer one shared implementation path for notebook inference and script-based export.

## Data Rules

- Default public sources are NYU Depth V2, MIT Intrinsic Images, MID Intrinsics, and the Fast Spatially-Varying Indoor Lighting Estimation dataset.
- Optional custom hallway data should use manifests and point-target files from `templates/`.
- Do not assume every dataset has every target.
- Leave missing manifest fields empty rather than fabricating values.

## Deployment Rules

- Save ONNX exports under `runs/exports/`.
- Save single-image inference artifacts under `runs/inference/`.
- Treat the ONNX export plus metadata JSON as the main later-deployment artifact for Raspberry Pi work.
