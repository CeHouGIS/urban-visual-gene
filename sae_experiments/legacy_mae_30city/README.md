# Replacement Feature-MAE cohort: BatchTopK SAE

This experiment reuses the old Feature-MAE task design, but the archived
original JPEG/TAR and patch-token inputs are absent. The user approved a new
NAS image batch. Do not describe this cohort as the old 378,818-image
Feature-MAE data or compare metrics as paired observations. The new cohort
is recorded in the experiment manifests/batch.json file.

## Sample and method

- 30 selected cities, sampled at pano level from GoogleSV NAS with metadata and
  all four images at headings 0, 90, 180 and 270 degrees.
- Target 3,200 panoramas per city; cities with fewer matching four-view panos
  use all eligible candidates. The manifest contains 370,768 images before QC.
- The existing small CPU classifier screens each four-view panorama before SAE
  training. Black or tunnel panos are excluded from training and inference.
  Raw image and DINO token caches remain intact; the quality parquet controls
  which rows enter the SAE experiment. The current batch has 370,768 manifest
  images and 363,700 quality-passed images.
- Each direction image is resized to 224 by 224 and encoded separately by
  frozen DINOv3 ViT-B/16. The 196 patch vectors are L2-normalized and 768-D.
- The SAE uses BatchTopK, dictionary width 4096, and K=32. Training samples
  patches from each city, uses panorama-disjoint validation, and saves a
  resumable checkpoint. Decoder vectors are clustered into 32 and 64 groups.
- Image features pool the 20 highest activations over 196 patches. Figures and
  tables are recomputed from this SAE. The figure inventory names any archived
  Feature-MAE figure that requires additional analysis instead of copying it.

## Run or resume

Run the module sae_experiments.legacy_mae_30city.run_nas_experiment from the
repository root. If four extraction jobs are already running, add the option
--wait-existing-extraction to avoid duplicate GPU work. The runner writes
runner_status.json and per-stage logs under its experiment output directory.
It resumes partial token arrays and SAE checkpoints, then runs quality
screening, model training, clustering, inference, and all figure modules:
`figures`, `extra_figures`, `visual_figures`, `pair_figures`, and
`spatial_figures`.

The DINOv3-B weights are loaded from the ModelScope cache. The default model
and NAS paths are in nas_batch.py.

## Outputs

Principal outputs are model_best.pt, checkpoint_last.pt, history.csv,
training_report.json, image_activations.f16.npy, inference_image_index.parquet,
results/*.csv, and figures/*.png. See figure_inventory.json for each archived
figure's status, and data_audit.json for screened train/validation counts.
The 64-cluster coactivation examples require both clusters to win at least four
patches in one directional image. City-enrichment uses the log2 ratio of the
city's joint-image rate to the other cities' rate. Spatial plots use 1 km
Web Mercator grids with at least five panoramas per grid; the comparison plots
exclude cities with fewer than 20 adjacent grid pairs. Full metrics, including
excluded cities, remain in results/city_spatial_structure_sae.csv.

`recomputed_from_SAE` in the inventory means a new SAE analogue, not identical
MAE methodology. The inventory's `reason` explains changed metrics. Old
graph-archetype, image-layout and Mask2Former segmentation figures still
require separate analyses and have not been copied or relabeled.
