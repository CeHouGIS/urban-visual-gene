Continue only from the verified candidate-combination stage and the frozen
30-city ViT-L W4096/K32 main SAE. The saved panorama means and four 64-group
heading vectors do not encode exact object positions or roadside identity.

Task: for candidate landscape combinations, re-run the frozen image encoder
and SAE on representative complete four-direction panoramas to recover patch
activations. Use an established image segmentation model or independently
checked road geometry to estimate the local road axis. Calibrate the four
camera bearings and assign element-mask regions to the signed sides of that
axis. Confirm 'same side' only with two confident element assignments to one
side, and 'opposite sides' only with different confident assignments and
visible road separation. Distinct camera directions alone are insufficient.
Mark intersections, curves, occlusions, uncertain masks, and ambiguous road
axes as undetermined. Preserve fine labels, object-family consolidation, and
the source provenance. Audit representative original panoramas with overlays
and record uncertainty; AI-only audit is not independent human validation.

Implement reproducible code and focused tests under
experiments/vitl30_city_patterns. Produce protocol.json, side_examples.csv
(panoid, city, grid, candidate, road-axis confidence, side label, evidence
paths), and summary.json with integer same_side, opposite_side, undetermined
counts. Include enough original four-view/masked evidence to inspect both
positive and rejected examples. Report city and cell support, not just image
count. Do not force a side classification for low-quality geometry.
