# Urban Visual Gene experiment outputs

These tables are derived from the 378,818 quality-controlled directional
street-view images. The frozen visual vocabulary is the 512-dimensional
Feature-MAE hierarchy built from **E + D + P** only:

- E: encoder direction;
- D: linearized decoder direction;
- P: 14 × 14 spatial activation profile.

City identity (C), activation context/co-occurrence (Q), and Qwen semantic
labels do not define the main hierarchy. C and Q are used only in ablations or
downstream analyses. Qwen labels are attached after the hierarchy is frozen.

## Primary settings

- Visual vocabulary: 64 fine elements nested within 32 coarse families.
- Element occurrence: top eight fine elements by per-image patch-winner count.
- PPMI support: at least `max(100 images, 0.05% of all images)` globally.
- Spatial analysis: 1 km grid, at least five unique panoramas per retained cell.
- Spatial sensitivity: 500 m, 1 km, and 2 km grids with 5/10/20/30-panorama
  support thresholds.
- Moran's I: queen-free rook adjacency and 999 permutations.
- Random seed: 42 unless a seed experiment explicitly states otherwise.

## All-city UMAP

The all-city UMAP contains all 378,818 training images. It is fitted on a
60,000-image subset with exactly 2,000 images per city. Each image is described
by its standardized 512-dimensional Feature-MAE activation profile, where each
dimension's image score is the mean of its 20 strongest spatial patches. The
fit subset includes at least 100 high-response representatives from a
2,000-image candidate pool for every one of the 512 dimensions. City labels are
used only for visualization, never for fitting PCA or UMAP. Exact coverage is
reported in `umap_sampling_audit.csv`, and the full coordinates are available
as both CSV and Parquet files.

Exact inputs, software versions, checkpoint, and run timestamp are recorded in
`analysis_provenance.json`. The per-city PPMI matrices are in
`city_cooccurrence/`.

## Validation status

The internal-data analyses are populated for clustering views/resolution,
visual vocabulary, city composition and similarity, global/city-specific
co-occurrence, within-city spatial structure, sampling, spatial resolution,
nuisance sensitivity, and Qwen semantic interpretation.

The controlled Feature-MAE sensitivity and seed suite writes
`feature_mae_sensitivity.csv`, `seed_stability.csv`, and
`seed_dimension_matching.csv` after its resumable GPU jobs finish. Independent
human or second-model semantic validation remains a separate study. OSM road
and building validation is intentionally postponed until the internal pipeline
is frozen, following Rule 8 of the experiment specification.
