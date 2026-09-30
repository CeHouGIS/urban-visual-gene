# Compact spectral image-graph archetypes — implementation report

## Outcome

The compact graph experiment completed successfully on all **378,818 directional street-view images** from **30 cities**. The original 2,080-dimensional graph descriptor remains unchanged and is used only for interpretable prototype summaries. Clustering now uses a separate **201-dimensional** encoding.

## Encoding implemented

For image \(s\), the new descriptor is

\[
z_s = [\sqrt{a_s},\rho_s,\operatorname{vech}(U^\top R_sU)] \in \mathbb{R}^{201}.
\]

- **64D composition:** square-root node areas \(\sqrt{a_s}\).
- **1D boundary density:** the fraction \(\rho_s\) of the 364 grid adjacencies that cross between different F categories.
- **136D topology:** the upper triangle of a symmetric \(16\times16\) spectral projection.

The edge residual compares the observed cross-category contact distribution with the independent-mixing expectation implied by the same image's node composition. Contacts are smoothed toward that image-specific null with \(\alpha=20\), so an element absent from an image cannot acquire a false adjacency merely from a global prior.

The fixed basis \(U\) consists of the first 16 non-constant eigenvectors of the normalized Laplacian of the training-set mean graph. It is learned once from the city-balanced training sample and reused for every image.

## Execution

The required staged run completed in this order:

1. 500-image encoding and numerical QA;
2. 5,000-image complete smoke test;
3. full 378,818-image feature build, PCA, K=8 clustering, assignment, prototypes, representatives, and figures.

All numerical stages ran in isolated processes with logical CPUs 8 and 9 excluded. Full encoding used a memory-mapped float32 array and 1,024-image batches.

## Full-data QA and model

| Item | Result |
|---|---:|
| Compact feature shape | 378,818 × 201 |
| Training images | 60,000 |
| Cities | 30 |
| Maximum node normalization error | 1.19e-7 |
| Maximum boundary-density reconstruction error | 1.19e-7 |
| Random 20-image exact rebuild error | 0 |
| NaN / Inf | 0 / 0 |
| Global Laplacian zero modes | 1 |
| PCA components retained | 112 |
| PCA explained variance | 90.0376% |
| K | 8 |

The retained PCA loading energy is **38.74% node composition**, **0.14% boundary density**, and **61.12% spectral topology**. In the original 2,080D experiment, approximately 96% of PCA loading energy was associated with node features. The new representation therefore fixes the main diagnostic problem: topology now materially participates in the clustering.

## Archetype sizes

| Archetype | Images | Share | Top nodes | Top edges |
|---|---:|---:|---|---|
| A01 | 63,130 | 16.66% | F057, F054, F058 | F036–F054, F007–F013, F007–F057 |
| A02 | 54,961 | 14.51% | F058, F054, F057 | F036–F054, F013–F058, F001–F013 |
| A03 | 49,794 | 13.14% | F058, F029, F013 | F013–F058, F036–F054, F027–F058 |
| A04 | 46,644 | 12.31% | F036, F013, F007 | F036–F054, F007–F013, F001–F013 |
| A05 | 46,445 | 12.26% | F028, F054, F021 | F010–F013, F033–F054, F021–F028 |
| A06 | 42,898 | 11.32% | F054, F036, F013 | F036–F054, F007–F013, F001–F013 |
| A07 | 39,879 | 10.53% | F054, F028, F058 | F036–F054, F010–F013, F021–F054 |
| A08 | 35,067 | 9.26% | F054, F010, F021 | F010–F013, F036–F054, F027–F035 |

## Interpretation and limitation

Visual inspection of the representative-image atlas shows coherent recurring within-image layouts, and the representatives span multiple cities. City labels and semantic annotations were not used by PCA or clustering.

The new descriptor is better aligned with the graph question, but the clusters should still be interpreted as archetypal regions in a continuous visual-composition space, not as sharply separated natural classes. On a fixed random sample of 5,000 images, mean silhouette is **0.0167** and 22.18% of samples have negative silhouette. This is weaker geometric separation than the node-dominated V1 result (0.0543), so the gain is representational relevance and compactness—not stronger evidence for eight discrete classes.

## Outputs

- Data and models: `paper/data/image_graph_compact_archetypes/`
- Atlas: `paper/figures/main/Fig_Compact_Graph_Archetype_Atlas.pdf` (the larger PNG is generated locally but not versioned)
- City heatmap: `paper/figures/main/Fig_Compact_Graph_City_Heatmap.png`
- Encoding QA: `paper/figures/main/Fig_Compact_Graph_QA.png`
- Detailed machine-readable reports: `compact_feature_report.json`, `training_assignment_report.json`, and `cluster_quality_report.json`
- Reproducible entry point: `run_image_graph_compact_archetypes.py`

The original V1 files under `paper/data/image_graph_archetypes/` were not overwritten.

## Ten image-level examples

An additional diagnostic atlas shows ten centroid-near images, their 14×14 F
maps, image-specific node/edge graphs, complete 512D standardized activation
profiles, and 512D top-1 winner-patch counts. See
`compact_graph_10_examples_guide.md` and
`paper/figures/supplementary/compact_graph_examples/`.
