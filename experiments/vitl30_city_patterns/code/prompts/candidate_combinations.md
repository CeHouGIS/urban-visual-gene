Continue the 30-city Urban Scene Primitives main experiment. The frozen
DINOv3 ViT-L panorama-aware W4096/K32 SAE is already trained; do not retrain it.
The completed extraction contains at most ten four-direction panoramas per
1-km city cell. Its panorama vectors and city/cell group-mass screening are
descriptive inputs, not verified prototypes.

Task: discover and screen recurring combinations of distinct landscape
elements. Read the existing Qwen hierarchy and structured four-view motif
work before choosing the method. Merge dimensions that name parts of one
object; do not count a wall plus a window as two independent objects. Keep
uncertain and low-evidence labels separate. Use whole panoramas as the
observation unit, spatial blocks and cities as repetition units. Compare
against null models that preserve element marginal prevalence and test
whether layout adds evidence beyond co-presence. Split discovery and
confirmation spatially or by city. Do not call panorama-level co-presence a
roadside relation or a validated landscape prototype.

Implement reproducible code and focused tests under
experiments/vitl30_city_patterns. Produce protocol.json with the exact source
identity, thresholds, splits and null models; candidates.csv with one row per
candidate and support from panoramas, cells and cities; and summary.json
including cities=30 and candidate_count. Negative findings are valid if the
specified tests find no candidates. Document limitations and code paths.
