# Mask2Former Swin-L Mapillary Vistas pilot

## Purpose

This pilot tests whether an off-the-shelf supervised semantic segmentation
model can provide a stable semantic reference for interpreting the learned
Feature-MAE dimensions. It does not use semantic labels to train or modify
Feature-MAE.

## Model and sampling

- Checkpoint: `facebook/mask2former-swin-large-mapillary-vistas-semantic`
- Task: semantic segmentation with the Mapillary Vistas 65-class taxonomy
- Sampling seed: 42
- Sample: 100 complete panorama locations from 30 cities
- Views: four independent cardinal views per location (0°, 90°, 180°, 270°)
- Total predictions: 400 direction images
- Source resolution: 640 × 640 for all 400 images
- Model processor resolution: 384 × 384
- Output masks: restored to the original 640 × 640 resolution

The 100 locations are balanced across cities: ten cities contribute four
locations and the remaining twenty cities contribute three locations. The
four directions of a location are always retained together.

## Outputs

For each direction image, the pipeline stores:

1. A full-resolution `uint8` semantic class-ID mask.
2. A `14 × 14 × 65` tensor containing the within-patch fraction of every
   semantic class.
3. Image and prediction metadata.

The four direction tensors are also concatenated in heading order to form a
`14 × 56 × 65` semantic panorama tensor. This is aligned with the spatial
shape of the rectangular-panorama Feature-MAE result and is the recommended
input for later semantic-alignment analysis.

Large generated arrays and masks are stored outside Git at:

```text
/workplace/urban_visual_gene/outputs/experiments/dinov3_multicity/
  mask2former_swin_l_mapillary_pilot/
    masks/                                            # 400 PNG class-ID masks
    semantic_patch_fractions_65.f16.npy              # (400, 14, 14, 65)
    panorama_semantic_fractions_14x56x65.f16.npy     # (100, 14, 56, 65)
    run_complete.json
```

Small tables and the run report are under:

```text
paper/data/semantic_alignment/mask2former_swin_l_pilot/
```

The qualitative comparison is:

```text
paper/figures/supplementary/Fig_Mask2Former_Mapillary_Pilot.png
```

## Run result and QA

- Status: complete
- Failed images: 0
- Peak allocated CUDA memory: 0.978 GiB
- Wall time: 36.69 seconds, including model loading and output generation
- NaN values: 0
- Predicted mask IDs: within the valid range 0–64
- Every panorama has exactly four headings
- Patch probability-sum maximum error: 0.000488 (expected float16 rounding)

The ten classes with the largest pixel shares in this balanced pilot are:

| Rank | Class | Pixel share | Image presence |
|---:|---|---:|---:|
| 1 | Sky | 24.42% | 97.25% |
| 2 | Road | 21.71% | 97.25% |
| 3 | Vegetation | 17.70% | 96.00% |
| 4 | Building | 14.74% | 95.50% |
| 5 | Terrain | 3.92% | 67.00% |
| 6 | Sidewalk | 3.02% | 76.75% |
| 7 | Fence | 2.21% | 75.25% |
| 8 | Car | 2.03% | 72.75% |
| 9 | Wall | 1.59% | 70.50% |
| 10 | Curb | 1.13% | 84.00% |

These percentages describe this pilot sample and should not be interpreted as
population-level city estimates.

## Reproduction

Run from the repository root. The CPU affinity explicitly excludes CPUs 8 and
9, and inference uses batch size one internally.

```bash
HF_HOME=/workplace/models/huggingface \
HF_HUB_DISABLE_TELEMETRY=1 \
TOKENIZERS_PARALLELISM=false \
taskset -c 0-7,10-15 \
python3 -m scripts.multicity.run_mask2former_mapillary_pilot \
  --panoramas 100 \
  --qa-images 10
```

Existing valid masks are reused, making the run resumable. CUDA out-of-memory
errors are recorded in `progress.json` and terminate the run instead of
silently changing the model or input resolution.

## Interpretation boundary

This run establishes that the model can produce usable semantic reference
masks and patch-level class mixtures at the Feature-MAE grid scale. It does
not yet measure which of the 512 Feature-MAE dimensions align with which
semantic classes. That alignment should be computed as a separate experiment
using the saved `14 × 56 × 65` semantic fractions and the corresponding
`14 × 56 × 512` Feature-MAE activations.
