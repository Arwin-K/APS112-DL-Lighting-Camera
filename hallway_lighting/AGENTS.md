# Repository Rules

## Core Rules

- Never hardcode dataset paths.
- Keep the notebook under `notebooks/` as the main user interface.
- Do not invent labels, targets, or annotations that are not present in source data.
- Prefer readable, student-friendly code over clever abstractions.
- Save outputs, checkpoints, plots, and exports under `runs/`.
- Carbon must be derived from electricity use, not directly sensed.
- Public datasets are auxiliary priors; custom hallway labels are optional but improve performance.

## Engineering Expectations

- Keep configuration path-driven through YAML files and notebook variables.
- Add type hints and docstrings to new Python code.
- Use clear TODO blocks when functionality depends on future dataset-specific work.
- Keep notebook code understandable for Colab users.
- Preserve importable package boundaries so notebook cells can call package modules directly.

## Data Rules

- NYU Depth V2, MIT Intrinsic Images, MID Intrinsics, and the Fast Spatially-Varying Indoor Lighting Estimation dataset are the default public sources.
- Optional custom hallway data should use manifests and point-target files from `templates/`.
- Do not assume every dataset has every target.
- Treat carbon outputs as post-processing derived from power and energy estimates.
