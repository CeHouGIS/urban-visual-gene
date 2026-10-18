Continue only from verified recurring-combination and roadside evidence for
the 30-city, 1-km/ten-panorama sample. The goal is a city-pattern analysis,
not a gallery of selected exemplars.

Task: specify a reproducible detector for each retained candidate relation
and apply it to the complete quality-screened sampled corpus. If the saved
vectors cannot preserve the required spatial distinction, run frozen
patch-level inference on the necessary panoramas. Keep same-side, opposite-
side, and undetermined detections separate. Never infer roadside identity
from a heading index or a city-wide group average. Include negative and
ambiguous results, with source identities and detector thresholds.

For each prototype, report observed panoramas, occupied 1-km cells and cities.
Compute city prevalence by first estimating within-cell prevalence and then
weighting valid cells equally; include uncertainty using spatial blocks or a
defensible equivalent. Distinguish sampled-area prevalence from a claim about
all streets in a city. Compare against prevalence-preserving and spatial
nulls where relevant. Implement code and focused tests.

Produce protocol.json, summary.json, city_prototype_distribution.csv with
explicit city and prototype_id columns covering all 30 cities, and
cell_prototype_distribution.parquet with cell IDs, denominators and counts.
Save inspectable four-view examples and limitations. A scientifically valid
negative finding is preferable to assigning an unsupported prototype label.
