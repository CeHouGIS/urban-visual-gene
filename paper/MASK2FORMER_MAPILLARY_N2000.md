# Mask2Former Swin-L large semantic-segmentation run

## Scope

The production-scale semantic reference run is complete. It uses the frozen
`facebook/mask2former-swin-large-mapillary-vistas-semantic` checkpoint and the
65-class Mapillary Vistas taxonomy.

- Cities: 30
- Complete panorama locations per city: 2,000
- Panorama locations: 60,000
- Cardinal direction images: 240,000
- Directions retained per location: 0°, 90°, 180°, and 270°
- Sampling seed: 42
- Feature-MAE-aligned semantic patches: 47,040,000

Semantic labels were not used to train or change Feature-MAE. They serve as an
external supervised reference for interpreting the learned dimensions.

## Result

- Status: complete
- Failed images: 0
- Runtime: 18,550.75 seconds (5 hours, 9 minutes, 11 seconds)
- Throughput: 12.94 direction images per second
- Peak allocated CUDA memory: 0.978 GiB
- Output footprint: approximately 7.4 GiB
- Source sizes: 239,968 images at 640 × 640 and 32 images at 400 × 400
- Model processor size: 384 × 384
- Prediction masks were restored to each source image's original resolution

## Primary data products

Large generated outputs are deliberately kept outside Git:

```text
/workplace/urban_visual_gene/outputs/experiments/dinov3_multicity/
  mask2former_swin_l_mapillary_n2000_per_city/
    masks/                                            # 240,000 uint8 PNG masks
    panorama_semantic_fractions_14x56x65.f16.npy     # (60,000, 14, 56, 65)
    progress.json
    run_complete.json
    run.log
```

The panorama tensor is a 6,115,200,128-byte disk-backed NumPy array. Every
location is stored in heading order as a `14 × 56 × 65` grid, matching the
spatial grid of the rectangular-panorama Feature-MAE output.

Full local metadata are stored at:

```text
/workplace/urban_visual_gene/paper/data/semantic_alignment/
  mask2former_swin_l_mapillary_n2000_per_city/
    sample_manifest.csv          # 240,000 source-image records
    prediction_metadata.csv      # 240,000 prediction records
```

The large metadata tables are not committed to Git. The run report, 65-class
index, class prevalence table, and qualitative figure are committed under the
same paper data and figure directories.

## Integrity checks

- The semantic tensor has shape `(60000, 14, 56, 65)` and dtype `float16`.
- All 60,000 locations contain exactly four direction images.
- Every city contributes exactly 2,000 unique locations.
- Exactly 240,000 mask files are present.
- A deterministic sample of 1,000 panoramas contains no NaN or Inf values.
- Sampled semantic fractions are within `[0, 1]`.
- Maximum sampled class-sum error is 0.000416, consistent with float16
  rounding.
- The run log contains no error, traceback, failure, or out-of-memory marker.

## Most prevalent classes

| Rank | Class | Pixel share | Image presence |
|---:|---|---:|---:|
| 1 | Sky | 25.35% | 96.25% |
| 2 | Road | 22.04% | 96.15% |
| 3 | Vegetation | 16.58% | 95.57% |
| 4 | Building | 14.86% | 94.19% |
| 5 | Terrain | 3.38% | 68.02% |
| 6 | Sidewalk | 3.10% | 77.49% |
| 7 | Car | 2.48% | 72.73% |
| 8 | Fence | 2.25% | 75.04% |
| 9 | Wall | 1.73% | 71.18% |
| 10 | Curb | 1.05% | 82.76% |

These are balanced-sample descriptive statistics rather than population
weights for the 30 cities.

## Next analysis input

The semantic tensor can be paired directly with the corresponding
`60,000 × 14 × 56 × 512` frozen Feature-MAE activation tensor. The next
experiment should quantify, for each of the 512 dimensions, its association
with each of the 65 semantic classes while retaining mixed-class patch
fractions rather than reducing every patch to a single winning semantic label.
