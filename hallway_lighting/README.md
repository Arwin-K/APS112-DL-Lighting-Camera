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
6. Continue to model setup, training, evaluation, point-wise reporting, and carbon estimation.

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

## Current Status

Ready now:

- dataset input preparation from Drive paths
- archive extraction for `.zip`, `.tar`, `.tar.gz`, and `.tgz`
- `.mat` staging for official-file workflows
- dataset-specific manifest builders for all supported datasets
- custom hallway manifest and point-target JSON validation
- notebook cells that prepare datasets, build manifests, preview rows, and visualize example images

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
