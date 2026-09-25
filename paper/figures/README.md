# Figure organization

All publication figures live in this directory. The project-root `figures/`
directory contains compatibility symlinks only, so analysis scripts can keep
their existing output paths without creating duplicate figure files.

## Main figures

The `main/` directory contains figures that directly support the paper's core
claims:

- `Fig_MAE_Hierarchy_EDP_32_64`: multiscale visual vocabulary.
- `Fig_City_Visual_Composition_Heatmap`: cross-city element composition.
- `Fig_City_Similarity_Heatmap`, `Fig_City_Similarity_Embedding`, and
  `Fig_City_Hierarchical_Clustering`: inter-city visual similarity.
- `Fig_Global_Cooccurrence_Heatmap` and `Fig_Global_Visual_Element_Network`:
  global element combinations.
- `Fig_Composition_vs_Cooccurrence`: comparison of two city-similarity views.
- `Fig_Unexpected_Visual_Cooccurrence_Global`: global observed-versus-expected
  association against frozen E+D+P visual distance.
- `Fig_City_Specific_Atypical_Combinations`: city-versus-rest atypical
  combination scores for the strongest element pairs.
- `Fig_Image_Dimension_Coexistence`: within-image prevalence, spatial share,
  and supported co-occurrence of the 512 latent dimensions.
- `Fig_Homogeneity_vs_Local_Continuity` and
  `Fig_Within_City_Visual_Structure`: within-city spatial organization.

## Supplementary figures

The `supplementary/` directory contains embedding diagnostics, alternative
metrics, sensitivity results, and detailed variants:

- `Fig_All_City_Samples_UMAP`, `Fig_All_City_UMAP_Visual_Families`, and
  `Fig_All_City_UMAP_512_Dimensions`.
- `Fig_City_Similarity_JS_Heatmap` and
  `Fig_City_Cooccurrence_Similarity`.
- `Fig_Cluster_Resolution_Metrics` and `Fig_Cluster_Size_Distribution`.
- `Fig_Image_Dimension_Threshold_Sensitivity` and
  `Fig_Image_Dimension_PPMI_512`.
- `Fig_Prevalent_Dimension_Combinations`: the most frequent within-image
  dimension pairs and triples, with their conditional spatial proportions.
- `Fig_Global_Visual_Element_Network_Detailed`.

Interactive HTML figures are stored in `interactive/`. Superseded figures are
stored under `paper/archive/` rather than mixed with current results.
