# Hallway Lighting

Colab-first PyTorch project for hallway floor-plane illuminance estimation, point-wise illuminance reporting, and power-to-carbon reporting.

## Main Interface

The primary user workflow is the notebook:

- [`notebooks/hallway_illuminance_train_eval_all_in_one.ipynb`](/Users/ArwinKarir/Desktop/APS112-DL-Lighting-Camera/hallway_lighting/notebooks/hallway_illuminance_train_eval_all_in_one.ipynb)

The intended flow is:

1. Put official datasets or extracted folders in Google Drive.
2. Open the notebook in Google Colab.
3. Set Drive paths in the notebook config cell.
4. Run dataset preparation and manifest generation cells.
5. Train, validate, and test from the notebook.
6. Run single-image inference and ONNX export from the notebook after checkpoints exist.

## What The Project Produces

The model and notebook report:

- `avg_lux`
- `low_lux_p5`
- `high_lux_p95`
- `point_lux` under fixtures
- `point_lux` between fixtures
- automatic fixture count and approximate fixture locations when enabled at inference time
- approximate floor area between adjacent detected fixtures
- optional floor mask prediction
- optional albedo proxy
- optional gloss/specularity proxy
- `estimated_power_w`
- `interval_energy_kwh`
- `interval_carbon_kg`

Carbon is always derived from electricity use, not directly sensed.

## Supported Datasets

Only these public datasets are part of the default workflow:

| Dataset | Accepted Input | What It Contributes | Real Source Labels Used |
| --- | --- | --- | --- |
| NYU Depth V2 | extracted folder, `.mat`, `.zip`, `.tar`, `.tar.gz`, `.tgz` | geometry and floor priors | RGB and depth |
| MIT Intrinsic Images | extracted folder, `.zip`, `.tar`, `.tar.gz`, `.tgz` | reflectance / shading priors | RGB, reflectance, shading |
| MID Intrinsics | extracted folder, `.zip`, `.tar`, `.tar.gz`, `.tgz` | illumination-vs-material priors | RGB, albedo/reflectance, shading, optional gloss/specular if present |
| Fast Spatially-Varying Indoor Lighting Estimation | extracted folder, `.zip`, `.tar`, `.tar.gz`, `.tgz` | local indoor lighting priors | RGB, plus optional official appearance-related files when present |
| optional custom hallway dataset | extracted folder with manifest CSV, or direct CSV path | hallway-specific lux supervision | only the fields actually supplied by the user |

The project does not assume unavailable or gated datasets.

## Google Drive Placement

Nothing is hardcoded. Put datasets anywhere in Drive and point the notebook to them.

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
    images/
    lux_maps/
    floor_masks/
    point_targets/
```

Accepted notebook inputs:

- extracted folder
- `.zip`
- `.tar`
- `.tar.gz`
- `.tgz`
- `.mat` where applicable
- direct custom hallway manifest CSV

Archives are extracted under `runs/notebook_run/prepared_datasets/`. `.mat` inputs are staged safely, and NYU `.mat` inputs are materialized into manifest-ready content before training.

## Manifest System

All datasets are normalized into one manifest schema so notebook code can use one loader path.

Important normalized fields:

- `sample_id`
- `dataset_name`
- `split`
- `image_path`
- `floor_mask_path`
- `lux_map_path`
- `avg_lux`
- `low_lux_p5`
- `high_lux_p95`
- `point_targets_json`
- `material_label`
- `floor_finish_label`
- `albedo_path`
- `gloss_path`
- `measured_power_w`
- `interval_hours`
- `notes`

If a source dataset does not provide a field, the manifest leaves it empty. The code does not invent labels.

## Model Architecture

The model is a shared encoder-decoder network:

- ResNet-18 encoder
- U-Net-style decoder
- floor segmentation head
- dense lux-map head
- scalar avg-lux / p5 / p95 heads
- albedo proxy head
- gloss/specularity proxy head
- uncertainty head
- power regression head for downstream carbon reporting

Point-wise lux is not predicted as a separate image branch. It is sampled from the predicted lux map at canonical hallway floor positions or, during inference, at automatically detected fixture projections when auto-detection is enabled:

- `under_fixture_1 ... under_fixture_N`
- `between_fixture_1_2 ... between_fixture_(N-1)_N`

The material and gloss heads exist because hallway appearance affects how lighting reads in RGB images. The model uses those auxiliary priors to better separate illumination from surface appearance.

The project reports `p5` and `p95` instead of raw min/max because percentile summaries are much more stable than single-pixel extremes.

## Training Supervision Routing

The notebook training helper routes losses by dataset and available labels:

- NYU Depth V2 supervises geometry / floor-related outputs only.
- MIT Intrinsic Images supervises reflectance-style priors.
- MID Intrinsics supervises illumination-vs-material disentanglement.
- Fast Indoor Lighting supervises local-lighting appearance priors when official files are present.
- Custom hallway data supervises lux maps, lux scalars, point targets, optional floor masks, and optional power-derived carbon targets.

If a manifest row does not have a target for a head, that loss is skipped.

## Custom Hallway Data

Public datasets do not directly provide real hallway lux labels at under-fixture and between-fixture locations. For actual hallway performance, custom hallway labels are the most important data source.

Templates:

- [`templates/custom_hallway_manifest_template.csv`](/Users/ArwinKarir/Desktop/APS112-DL-Lighting-Camera/hallway_lighting/templates/custom_hallway_manifest_template.csv)
- [`templates/point_targets_template.json`](/Users/ArwinKarir/Desktop/APS112-DL-Lighting-Camera/hallway_lighting/templates/point_targets_template.json)

Supported point-target JSON format:

```json
{
  "under_fixture_1": 0.0,
  "under_fixture_2": 0.0,
  "between_fixture_1_2": 0.0
}
```

Useful optional custom hallway labels:

- floor mask
- lux map
- avg lux
- `low_lux_p5`
- `high_lux_p95`
- point target JSON
- measured power
- interval hours
- optional albedo and gloss references

## Notebook Usage

Typical Colab workflow:

1. Install dependencies.
2. Mount Google Drive.
3. Edit `DATASET_INPUTS` in the configuration cell.
4. Run archive extraction and manifest-building cells.
5. Inspect manifest previews and coverage diagnostics.
6. Create dataloaders and inspect one batch.
7. Initialize the model and optionally load `best` or `last` checkpoint.
8. Run training.
9. Run validation or testing.
10. Inspect saved figures, point reports, checkpoints, and history.
11. Export ONNX.
12. Run single-image inference on a hallway image.

Primary outputs are written under `runs/notebook_run/`:

- `checkpoints/`
- `visualizations/`
- `manifests/`
- `training_history.json`
- `config_snapshot/`

Single-image inference outputs are written under `runs/inference/`:

- `*_summary.json`
- `*_lux_heatmap.png`
- `*_lux_overlay.png`
- `*_point_overlay.png`
- `*_prediction_overview.png`

## Validation, Testing, And Inspection

The notebook evaluation sections report:

- average lux metrics
- p5 metrics
- p95 metrics
- point-wise lux outputs
- saved visual examples
- carbon summaries derived from estimated or measured power

## Inference

The package provides a shared inference helper in [`hallway_lighting/infer.py`](/Users/ArwinKarir/Desktop/APS112-DL-Lighting-Camera/hallway_lighting/hallway_lighting/infer.py).

It supports:

- loading a PyTorch checkpoint
- loading an ONNX model
- single-image inference
- JSON summary export
- lux heatmap export
- overlay visualization export
- point-annotation export
- automatic fixture-layout analysis with inferred count, approximate locations, and between-fixture floor regions

For a quick local laptop test against an exported ONNX, use:

- [`notebooks/local_photo_onnx_fixture_test.ipynb`](/Users/ArwinKarir/Desktop/APS112-DL-Lighting-Camera/hallway_lighting/notebooks/local_photo_onnx_fixture_test.ipynb)

The notebook lets you upload one photo from your laptop, runs the exported ONNX, detects fixtures, reports:

- lux directly under each detected fixture
- lux at the midpoint between adjacent fixtures
- inferred fixture count

It also saves:

- an annotated overlay image showing detected fixtures, under-fixture lux points, and between-fixture lux points
- a JSON summary with `point_lux` and `fixture_analysis`

The notebook uses the same helper that later deployment scripts can use.

## ONNX Export

You can export from:

- the notebook ONNX section
- [`scripts/export_onnx.py`](/Users/ArwinKarir/Desktop/APS112-DL-Lighting-Camera/hallway_lighting/scripts/export_onnx.py)

Example:

```bash
cd hallway_lighting
python scripts/export_onnx.py \
  --checkpoint runs/notebook_run/checkpoints/best_model.pt \
  --output runs/exports/hallway_multitask_unet.onnx \
  --device cpu
```

The export script also saves a `.metadata.json` file with model input shape and output names. That ONNX file is the main deployment-oriented artifact for later Raspberry Pi use.

## Installation

```bash
pip install -r requirements.txt
```

## What The User Still Must Provide

The repository is ready for practical use once you provide:

- Google Drive dataset paths
- official dataset files or extracted dataset folders
- optional custom hallway manifest CSV
- optional point-target JSON files
- at least one hallway image for final single-image inference

For strong hallway-specific results, custom hallway lux supervision still matters most.