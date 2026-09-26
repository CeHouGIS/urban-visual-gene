# Feature-MAE dimension–semantic alignment

## Research question

Do the 512 frozen Feature-MAE bottleneck dimensions consistently respond to
recognizable street-scene semantics, or are their activations primarily mixed,
position-dependent, or caused by panorama artifacts?

The representation evaluated here is the dense 512D Feature-MAE bottleneck.
It is not the earlier non-negative TopK SAE. Consequently, each dimension must
be studied through its own high activations rather than assigning every patch
to its single highest-response dimension.

## Data

- 60,000 panorama locations sampled evenly from 30 cities
- 2,000 locations per city
- 59,992 locations with valid Feature-MAE and semantic outputs
- 8 excluded Vienna locations whose original Feature-MAE panorama construction
  had failed because their source views were 400 × 400
- Feature-MAE activation grid: `14 × 56 × 512`
- Mask2Former semantic grid: `14 × 56 × 65`

Both tensors refer to the same panorama locations and spatial cells.

Each of the 512 dimensions is additionally annotated with the previously
frozen `D000–D511 → F000–F063` E+D+P hierarchy. This mapping is descriptive and
is not recomputed from the Mask2Former results. The F category and its existing
English and Chinese labels therefore provide an internal visual-element name,
while the 65 Mask2Former columns remain an independent supervised semantic
check.

## Per-dimension method

No winner map or cross-dimension argmax is used in the semantic calculation.
For every dimension `D000`–`D511` independently:

1. Rank all 59,992 panoramas by that dimension's existing image-level score,
   defined as the mean of its 20 strongest patches.
2. Retain the 50 strongest panoramas for that dimension.
3. Read that dimension's complete `14 × 56` activation map in those panoramas.
4. Retain the dimension's 1,000 strongest patches.
5. Average the 65-class fractional Mask2Former composition of those patches.

The resulting vector estimates:

```text
P(Mapillary semantic class | dimension has a very high activation)
```

Every dimension therefore has the same semantic-profile sample size. The
profile describes what a dimension prefers, not how frequently it activates in
the full dataset.

## Twenty visual case studies

The 20 displayed locations were clustered using both the 512D image activation
profile and the 65D image semantic composition. The closest point to each of 20
cluster centroids was selected, with at most one point per city. The final set
contains 20 distinct cities.

Each detailed case-study figure contains:

- the original four-direction panorama;
- the full-resolution Mask2Former semantic overlay;
- six separate Feature-MAE activation heatmaps;
- the three most prevalent semantic classes in each dimension's global
  high-activation profile.

These heatmaps show each dimension separately. They do not display a winner
dimension map.

## Quantitative result

| Assessment | Dimensions | Share |
|---|---:|---:|
| Clear single Mapillary semantic | 291 | 56.84% |
| Coherent semantic mixture | 119 | 23.24% |
| Low Mapillary specificity | 72 | 14.06% |
| Spatial/artifact candidate | 30 | 5.86% |

Thus, 410 of 512 dimensions (80.08%) show either a strong single semantic or a
stable semantic mixture in their highest activations.

Examples of highly specific dimensions include:

| Dimension | Dominant semantic | Share in top patches |
|---|---|---:|
| D250 | Sky | 99.99% |
| D008 | Vegetation | 99.68% |
| D427 | Building | 98.34% |
| D361 | Road | 98.05% |

Among the 291 clear dimensions, the largest groups are vegetation (76),
building (62), road (62), and sky (59). This indicates substantial semantic
redundancy: the representation contains multiple dimensions for different
appearances or subtypes of the same broad semantic category.

## What counts as an artifact candidate

A dimension is only marked as a spatial/artifact candidate when all of the
following hold:

- it is neither a clear single semantic nor a coherent semantic mixture;
- its largest semantic class accounts for less than 35% of high activations;
- and its high activations are strongly concentrated at cardinal-view seams
  or at a narrow set of grid rows/columns.

Of the 30 candidates, 7 exceed the explicit seam-lift threshold and 26 show a
strong fixed-position preference; these groups overlap. Examples with high seam
lift include D279, D191, D304, D326, D455, D249, and D442.

The word *candidate* is essential. Low agreement with Mapillary labels is not
proof of a model artifact. A Feature-MAE dimension can legitimately represent
material, texture, lighting, shadow, perspective, façade style, or another
concept absent from the 65-class taxonomy. Mask2Former errors can also distort
the profile; for example, entrance-like dark regions may be predicted as
`Tunnel`.

## Output files

Exact 512 × 65 profiles and diagnostics:

```text
paper/data/semantic_alignment/feature_mae_mapillary_alignment_n60000/
  alignment_summary.json
  dimension_semantic_profiles.csv
  dimension_semantic_distribution_65.csv
  dimension_f64_mapping.csv
  dimension_top_patch_exemplars.csv
  heatmap_dimension_order.csv
  heatmap_semantic_order.csv
  selected_20_points.csv
  selected_point_top_dimensions.csv
```

Figures:

```text
paper/figures/supplementary/feature_mae_semantic_alignment_20/
  Fig_512D_Semantic_Profile_Heatmap.png
  Fig_Dimension_Semantic_Clarity.png
  Fig_FeatureMAE_Semantic_Alignment_20_Atlas.jpg
  P01_semantic_feature_alignment.jpg
  ...
  P20_semantic_feature_alignment.jpg
```

The 512-row figure is an ordinary heatmap of every dimension's complete
65-class semantic profile; it does not apply hierarchical clustering. Rows are
grouped by the frozen visual-element mapping `F000`-`F063`, and dimensions
within each F group are ordered by ascending D ID. Each displayed row label
therefore describes one F group and reports its number of constituent
dimensions. The adjacent `F64` strip makes these fixed groups explicit. Columns
use a stable, human-readable order by nature, ground/road, built structure,
people/riders, street-object, and vehicle roles. The coloured `type` side strip
identifies clear, mixed, low-specificity, and artifact-candidate dimensions.
The exact fixed row and column orders are recorded in
`heatmap_dimension_order.csv` and `heatmap_semantic_order.csv`; the former also
contains each D dimension's rank within its F group. These CSV tables should be
used when exact dimension IDs and percentages are required. The full D-to-F
mapping, including the prior English and Chinese F labels, is stored in
`dimension_f64_mapping.csv`.

## Conclusion

The experiment supports the feasibility of semantic interpretation: most
Feature-MAE dimensions have meaningful semantic concentration, and only a
small minority are currently suspicious for spatial artifacts. The result does
not imply that every low-specificity dimension is useless. Those dimensions
need exemplar inspection against texture, material, illumination, geometry,
and seam position before any dimensions are removed.
