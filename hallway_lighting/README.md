# Hallway Lighting

Colab-first PyTorch project for hallway floor-plane illuminance estimation with auxiliary indoor priors from public datasets and optional hallway-specific supervision.

## What The Project Does

The main target is hallway lighting estimation. The intended outputs are:

- `avg_lux`
- `low_lux_p5`
- `high_lux_p95`
- point-wise lux under fixtures
- point-wise lux between fixtures
- `estimated_power_w`
- `interval_energy_kwh`
- `interval_carbon_kg`

Carbon is always derived from lighting electricity use, not directly sensed.

## Main User Interface

The primary interface is the Colab-friendly notebook:

- [`notebooks/hallway_illuminance_train_eval_all_in_one.ipynb`](/Users/ArwinKarir/Desktop/APS112-DL-Lighting-Camera/hallway_lighting/notebooks/hallway_illuminance_train_eval_all_in_one.ipynb)

The expected workflow is:

1. Put official dataset files or extracted folders in Google Drive.
2. Open the notebook in Colab.
3. Set dataset input paths in the notebook.
4. Run dataset preparation and manifest cells.
5. Preview rows and sample images.
6. Create dataloaders from the saved manifests.
7. Initialize the model and optionally resume from a checkpoint.
8. Run training, validation, and test evaluation from the notebook cells.
9. Inspect saved checkpoints, visualizations, point reports, and history under `runs/notebook_run/`.

## Supported Datasets

Only these datasets are part of the default workflow:

| Dataset | Input Types | What It Contributes | Real Labels Ingested |
| --- | --- | --- | --- |
| NYU Depth V2 | extracted folder, `.mat`, `.zip`, `.tar`, `.tar.gz`, `.tgz` | indoor geometry priors | RGB image path, depth path |
| MIT Intrinsic Images | extracted folder, `.zip`, `.tar`, `.tar.gz`, `.tgz` | reflectance and shading priors | RGB image path, reflectance path, shading path |
| MID Intrinsics | extracted folder, `.zip`, `.tar`, `.tar.gz`, `.tgz` | intrinsic appearance priors | RGB image path, albedo/reflectance path, shading path, optional gloss/specular path if present |
| Fast Spatially-Varying Indoor Lighting Estimation dataset | extracted folder, `.zip`, `.tar`, `.tar.gz`, `.tgz` | lighting and appearance priors | RGB image path, optional albedo/gloss references when official files are present |
| custom hallway dataset | extracted folder containing a manifest CSV, or a direct manifest CSV path | hallway-specific lux, floor, point, and power supervision | only fields actually present in the user CSV |

No unavailable or gated dataset is assumed.

## Google Drive Dataset Setup

Place datasets anywhere in Drive. Paths must be user-provided in the notebook or configs; nothing is hardcoded.

Example layout:

```text
/content/drive/MyDrive/datasets/
  nyu_depth_v2/
    nyu_depth_v2_labeled.mat
  mit_intrinsic_images/
    ...official extracted files...
  mid_intrinsics/
    ...official extracted files...
  fast_sv_indoor_lighting/
    ...official extracted files...
  custom_hallway/
    custom_hallway_manifest.csv
    point_targets/
    images/
    lux_maps/
    floor_masks/
```

Accepted input paths in the notebook:

- extracted dataset folder
- `.zip`
- `.tar`
- `.tar.gz`
- `.tgz`
- `.mat` where applicable, especially NYU Depth V2
- direct custom hallway manifest CSV for the custom dataset only

The preparation utilities detect the input type, extract archives into a local working directory under `runs/`, stage `.mat` files safely, and return prepared dataset roots that the manifest builders consume.

## Manifest Generation

Every supported dataset is normalized into the same manifest schema so the training notebook can consume a single table format.

Core normalized fields:

- `sample_id`
- `dataset_name`
- `split`
- `image_path`
- `depth_path`
- `floor_mask_path`
- `lux_map_path`
- `avg_lux`
- `low_lux_p5`
- `high_lux_p95`
- `point_targets_json`
- `material_label`
- `floor_finish_label`
- `albedo_path`
- `reflectance_path`
- `shading_path`
- `gloss_path`
- `measured_power_w`
- `interval_hours`
- `notes`

If a source dataset does not provide a field, the manifest leaves it empty. The code does not invent labels that are absent from the source data.

## Notebook Training Flow

The notebook now supports the practical end-to-end execution flow:

1. Load configs and resolve Google Drive dataset paths.
2. Prepare extracted roots and build normalized manifests.
3. Load manifests and inspect split counts.
4. Create train/val/test dataloaders.
5. Initialize the model, optimizer, scheduler, and AMP scaler.
6. Optionally resume from a checkpoint.
7. Run the training loop from the training section.
8. Run validation or test evaluation on demand.
9. Visualize predictions and hallway point overlays.
10. Inspect `best_model.pt`, `last_model.pt`, training history, config snapshots, and saved figures.

The notebook is still the main interface. Helper code in the package exists to keep notebook cells readable, but the intended user flow remains notebook-first.

## Multi-Dataset Supervision Routing

The training helper routes losses by dataset name and available labels:

- NYU Depth V2: floor-related supervision only when a floor target is available.
- MIT Intrinsic Images: albedo / reflectance-style supervision.
- MID Intrinsics: albedo plus gloss/specularity supervision when available.
- Fast Indoor Lighting: appearance-related supervision when official targets are present in the manifest.
- custom hallway dataset: lux map, scalar lux values, point-wise lux, floor mask, optional albedo/gloss, and optional power-derived carbon supervision.

If a dataset row does not provide a label for a given head, that loss is skipped.

## Custom Hallway Data

Public datasets do not directly provide hallway lux supervision at under-fixture and between-fixture points. For that reason, custom hallway data is the main route to real hallway-specific performance.

The custom hallway adapter expects a CSV manifest. A template is provided here:

- [`templates/custom_hallway_manifest_template.csv`](/Users/ArwinKarir/Desktop/APS112-DL-Lighting-Camera/hallway_lighting/templates/custom_hallway_manifest_template.csv)

Optional point-wise lux labels can be provided through JSON files referenced by `point_targets_json`. The supported format is:

```json
{
  "under_fixture_1": 0.0,
  "under_fixture_2": 0.0,
  "between_fixture_1_2": 0.0
}
```

A template is provided here:

- [`templates/point_targets_template.json`](/Users/ArwinKarir/Desktop/APS112-DL-Lighting-Camera/hallway_lighting/templates/point_targets_template.json)

## Real Labels Vs Optional Labels

Real public labels used by the data layer:

- NYU Depth V2: RGB and depth
- MIT Intrinsic Images: RGB, reflectance, shading
- MID Intrinsics: RGB, albedo/reflectance, shading, optional gloss/specular if the official files expose it
- Fast Indoor Lighting: RGB, plus optional appearance-related files when present

Optional hallway labels:

- floor mask
- lux map
- average lux statistics
- point target JSON
- material label
- floor finish label
- albedo
- gloss
- measured power
- interval hours

These hallway fields are only consumed when you provide them.

## Repository Layout

```text
hallway_lighting/
  README.md
  AGENTS.md
  requirements.txt
  pyproject.toml
  configs/
  hallway_lighting/
  notebooks/
  templates/
  runs/
```

## Installation

```bash
pip install -r requirements.txt
```

## Running The Notebook

Typical Colab usage:

1. Mount Drive.
2. Fill `DATASET_INPUTS` with Drive paths.
3. Run the manifest-building cells.
4. Confirm the split counts in the manifest loading section.
5. Run the dataloader and model initialization sections.
6. Set `RUN_TRAINING = True` in the training section when ready.
7. Set `RUN_VALIDATION = True` or `RUN_TEST = True` for standalone evaluation passes.
8. Inspect saved outputs under:
   - `runs/notebook_run/checkpoints/`
   - `runs/notebook_run/visualizations/`
   - `runs/notebook_run/training_history.json`
   - `runs/notebook_run/config_snapshot/`

## Current Status

Ready now:

- dataset input preparation from Drive paths
- archive extraction for `.zip`, `.tar`, `.tar.gz`, and `.tgz`
- `.mat` staging for official-file workflows
- dataset-specific manifest builders for all supported datasets
- custom hallway manifest and point-target JSON validation
- notebook cells that prepare datasets, build manifests, preview rows, and visualize example images
- notebook-integrated dataloader creation, training, validation, testing, visualization, and checkpointing flow

Not added yet:

- smoke tests
- overfit tests
- automated test infrastructure
- finalized multi-dataset `Dataset`/`DataLoader` training pipeline

## Implementation Notes

- Code uses Python type hints and docstrings.
- Parsing logic is intentionally readable and fails loudly on malformed input.
- Outputs and prepared artifacts should be stored under `runs/`.
- Public datasets are auxiliary priors. Custom hallway labels are optional but expected to improve hallway-specific performance.
