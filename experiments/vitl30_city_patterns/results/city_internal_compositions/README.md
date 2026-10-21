# City-internal landscape composition screening

Run from the repository root:

```bash
PYTHONPATH=. python -m experiments.vitl30_city_patterns.code.city_internal_compositions
```

This analysis reuses the frozen 30-city panorama assignment in
`followup/stages/composition_patterns_screened/assignment.npy`. It does not
extract features, retrain the SAE, or infer patch geometry. The semantic-unit
thresholds were calibrated once in the earlier analysis and are held fixed
across cities. Mobile objects and excluded temporary objects remain excluded.

Each city is mined separately for combinations of two to four distinct semantic
families. A candidate must cover at least 0.5% of the equal-cell-weighted
sample, 20 panoramas and eight cells in the discovery 5-km blocks. Confirmation
requires the same rate, ten panoramas and four cells in held-out blocks. One
subtype can represent each family per panorama. This prevents counting parts of
one family as separate members, but it also prevents detecting two separate
trees or buildings in the same panorama.

Files:

- `city_patterns.csv`: all within-city discovery candidates, including holdout
  outcomes and the count of cities in which each exact combination replicated.
- `city_summary.csv`: replicated pair, triple and quadruple counts per city.
- `pattern_city_support.csv`: exact combinations and their number of confirming
  cities; the city list is included.
- `single_city_candidates.csv`: combinations confirmed in just one city, with
  the highest observed rate in any other city. This is a retrieval shortlist,
  not proof that the combination occurs nowhere else.
- `all_city_pattern_rates.csv`: rate of each retained exact combination in all
  30 cities, including below-threshold responses.
- `city_overlap_jaccard.csv`: pairwise Jaccard overlap of confirmed exact
  subtype combinations.
- `city_parent_overlap_jaccard.csv`: the same calculation after replacing
  subtype combinations with semantic-family signatures.
- `city_pattern_overlap.png` and `.pdf`: counts and exact-overlap matrix.
- `city_sampling.csv` and `summary.json`: denominators, split and run metadata.

All matches are panorama-level SAE co-responses pooled across four views.
Neither Qwen labels nor spatial arrangements have been independently validated.
The overlap and city-specificity comparisons are exploratory; they are not
significance tests and cannot establish genuinely exclusive landscape types.

## Rio three-scope figure

`rio_three_scopes/Fig_Rio_Three_Scopes.png` and `.pdf` illustrate one
Rio de Janeiro combination at each scope: road + trees (confirmed in 29/30
cities), wall + tile-roof building + sidewalk + wires (Rio and Brasilia), and
wall + tile-roof building + hillside (only Rio passes the within-city cutoff).
The first category requires detection in most cities, not all cities. The
last two are sample-specific candidates, not proven regional or city-exclusive
types. Gray map cells contain sampled panoramas; colored cells contain at least
one match. Four-view overlays show relative frozen SAE response, not object
segmentation or a verified arrangement. `rio_three_scopes/manifest.json`
records the exact pattern IDs, support cities, counts and illustrated panoids.

Reproduce with a GPU and the frozen main-model inputs:

```bash
python -m experiments.vitl30_city_patterns.code.render_city_three_scopes --case rio --device cuda:0
```

The comparable Dhaka case is in `dhaka_three_scopes/`. It uses road + trees
(confirmed in 29/30 cities), fence + older roadside structures + trees
(Dhaka and Yogyakarta), and older roadside structures + signs/posters + trees
(only Dhaka passes the within-city cutoff). Its manually selected third-ranked
example shows the signage response more clearly than the highest-strength
example; the rank and panoid are recorded in its manifest. These candidate
labels are provisional Qwen interpretations, and the selected examples include
peri-urban scenes rather than representing every part of Dhaka. Reproduce with:

```bash
python -m experiments.vitl30_city_patterns.code.render_city_three_scopes --case dhaka --device cuda:0
```

## Dhaka local grid and example atlas

`dhaka_local_atlas/` maps all 55 combinations that replicate within Dhaka and
do not meet the same cutoff in another study city. They are not proven absent
elsewhere. The study region includes peripheral and rural street views; the
maps cover all 9,587 observed 1-km cells rather than the administrative urban
core alone. At least one candidate matches in 7,458 cells and 21,561 of the
93,400 sampled panoramas. The 55 candidates include nested pairs, triples and
quadruples, so counts of candidate types per cell are not counts of independent
landscape objects.

- `Dhaka_Local_Overview.png` and `.pdf`: all observed cells, any-match cells and
  the number of matching candidate types per cell.
- `Dhaka_Local_Combination_Atlas.pdf`: one grid map and one four-view original
  plus SAE overlay for each of the 55 candidates. Each page also exists as
  `patterns/<pattern_id>/map_and_example.png`.
- `Dhaka_Local_Examples_P01.png` through `P05.png`: one original four-view
  example per candidate, arranged for quick visual review.
- `Dhaka_Local_Grid_Summary.csv` and `Dhaka_Local_Grid_Pattern_Counts.csv`:
  complete cell-level counts including zero-match cells in the first file.
- `manifest.json`: pattern IDs, raw and display labels, counts, panoids,
  response ranks, and image paths.

Examples are selected from strong calibrated SAE responses, with distinct
panoids across candidates. Their labels remain provisional Qwen interpretations;
the colored pixels are not segmentation masks, and an example is not a random
sample or a validation of label accuracy.

```bash
python -m experiments.vitl30_city_patterns.code.render_dhaka_local_atlas --device cuda:0
```
