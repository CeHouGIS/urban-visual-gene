# 30-city spatially balanced SAE panorama sample

This experiment samples the original 30 main-experiment cities from source
metadata. Within each city, metadata coordinates are projected to the local
UTM zone and binned into fixed 1,000 m by 1,000 m cells. A fixed-seed stable
hash selects at most 10 distinct four-view panoids per occupied cell. Cells
with fewer than 10 eligible panoids retain all of them. Each selected panoid
is processed with all four directions; image completeness and quality are
checked when the images are read. A rejected panoid is recorded but is not
replaced, so the final usable count may be below 10 for a cell.

The frozen main DINOv3 ViT-L/16 encodes each direction at 448 px into a 28x28
patch grid. The main panorama-aware context construction mixes the four
headings before the frozen W4096/K32 BatchTopK SAE. BatchTopK is applied over
the 3,136 patches of each complete panorama. The output keeps the panoid,
coordinates, 1-km grid cell, 4,096-dimensional mean positive SAE activation,
and four 64-group heading vectors. It does not store dense patch activations,
so later local mask inspection needs another frozen-model pass.

## Current Chapter 5 results: screened semantic analysis

The current thesis figures use the frozen DINOv3-L/SAE responses of **681,645
complete panoramas, 72,625 observed 1-km cells, and all 30 cities**. The previous
decoder-based 64-group vocabulary and seven-pair figures have been retired from
the thesis. Their server-side caches are historical products, not current
semantic results. No backbone or SAE is retrained for this revision.

`results/current_semantic_hierarchy.json` freezes the current dimension screening
and 17 spatial-encoding candidate overrides. It corrects an older city-analysis
snapshot that still admitted 18 unsupported and seven spatially flagged
dimensions. The complete displayed taxonomy accounts for all 4,096 dimensions
in 22 parents and 190 subclasses, including uncertainty. The narrower landscape
analysis retains **1,345 eligible dimensions in 69 semantic units and 13 families**.
Synonyms and same-object parts are consolidated before combination mining.
People, vehicles, containers, mobile machinery and the explicitly listed
temporary categories are excluded; the complete exclusion list is in the run
protocol. Exclusions do not delete SAE dimensions. They cannot re-enter through
parent fallbacks, and all 4,096 positive responses remain in the denominator.

The screened stage is `followup/stages/composition_patterns_screened/` under the
NAS experiment root. Saved dimension-level means are re-aggregated there,
without repeating full-corpus image inference. Relative-response thresholds use
positive 60th percentiles in 22 discovery cities, balancing cities, observed
cells and within-cell panoramas. Child/parent mass floors are 0.1%/0.5%.
A responsive parent contributes at most one normalized winning child or an
eligible fallback. Apriori searches orders 2–4, requiring at least 0.5% weighted
support, 30 panoramas, eight cells and three cities in both city partitions.
An extension retaining 90% of support prunes a near-redundant subset.

The screened run retains **184 combinations: 125 pairs and 59 triples**.
At least one matches **85.03%** of panoramas, or **85.69%** with cities and
observed cells weighted equally. These are panorama retrieval rates, not patch
explanation, object area, object recall, or the 80% patch-coverage target.
At quantiles 50/60/70 the same frozen patterns yield weighted coverage
93.64/85.69/68.54%; 181/184/119 patterns repeat in evaluation cities. Those
cities were examined during previous revisions, so these are exploratory
response patterns, not new confirmatory results or validated landscape
prototypes. Object counts, spatial relations and road sides remain unverified.

### Reproduce the current figures

```bash
python -m experiments.vitl30_city_patterns.code.composition_patterns \
  --hierarchy experiments/vitl30_city_patterns/results/current_semantic_hierarchy.json \
  --output /host/root/mnt/nas/huangyj/GoogleSV/urban_visual_gene/vitl30_city_patterns_1km10/followup/stages/composition_patterns_screened \
  --run-identity 6d6519a1cb0b78d2fbc384722030a3a3fb929764724630c016cf2ddb3c34e617
python -m experiments.vitl30_city_patterns.code.semantic_city_evidence --device cuda:3
# Inspect the refreshed maps and record their statuses in stage/evidence_review.json.
# Re-running evidence generation resets its review; no unchecked image is plotted.
python -m experiments.vitl30_city_patterns.code.semantic_city_figures
```

Evidence inference is GPU-only and refreshes the actual screened unit membership.
Fourteen candidate panoramas were visually inspected by the assistant; two with
semantic mismatches were excluded from illustration selection. This bounded
inspection is not independent human validation or an accuracy estimate. The
record is `results/semantic_city_evidence_review.json`. No Qwen/API calls are
needed by these commands, and there is no CPU model-inference fallback.

`results/figures/` provides PDF/PNG replacements for the pipeline, city shares
and excess, city similarity, cell PPMI, adjacent-cell structure, a 3,000-cell
UMAP, composition heatmap/intervals and refreshed four-view response examples.
The 12 selected combinations have full 30-city 1-km distribution maps, including
zero-match cells. Each city also has its own response-share and top-eight
composition chart in `figures/cities/`; all 30 pages are collected in
`Fig_ViTL30_City_Composition_Atlas.pdf`. Composition matches may overlap.
`semantic_city_composition.csv`, `all_composition_key.csv`,
`selected_composition_key.csv` and `selected_composition_city_intervals.csv`
retain values and stable pattern identities. `semantic_figure_manifest.json`
records input/output hashes, weighting and plotting parameters.

The current thesis source is `/root/code/HKUST-Thesis/05_research3.tex` plus its
three Chapter 5 figure includes and `A_appendix_results.tex`. Original SAE
training diagnostics are retained. The old `code/chapter5_figures.py` installer
is guarded against overwriting this semantic revision with 64-group results.
The original queue completion and all audits documented below refer to their
historical input snapshots; their numerical results have not been recomputed
with the screened vocabulary and must not be presented as current findings.

### Feature breadth and category heatmaps

`code/render_feature_city_breadth.py` renders the 30-city tree-feature heatmap,
the semantic-class breadth boxplot, and 18 category-specific heatmaps from the
saved city-feature response tables. Run it from the repository root with:

```bash
python -m experiments.vitl30_city_patterns.code.render_feature_city_breadth
```

Outputs are in `results/chapter5_revision/feature_city_breadth/`. The 18
category heatmaps are also collected in
`category_heatmaps/Fig_ViTL30_All_Class_Heatmaps.pdf`. They retain all 2,626
dimensions assigned to those categories; only 1,264 unresolved, 17 spatial
encoding, 46 people, and 143 vehicle dimensions are excluded. Category
membership comes from the saved assignments, without status-based filtering.
All heatmaps share a 30-city column order and a muted color scale. The tree
heatmap reproduces the original panel A row and column order; the category
heatmaps order rows by response-profile similarity. The boxplot's effective
city count measures evenness across sampled cities, not the number of cities
with any positive response. Labels remain provisional.

## Historical local composition audit (older semantic snapshot)

`code/local_composition_audit.py` retrieves the 185 compositions from the
681,645-panorama assignment and inspects six distinct-city matching panoramas
per composition. The full-corpus vectors contain panorama-level activation
summaries, not patch positions. The audit therefore uses saved roadside patch
maps where complete and recomputes frozen DINOv3-L/SAE maps on GPU for the
remaining 55 pairs and 49 triplets. It does not repeat full-corpus inference.

```bash
python -m experiments.vitl30_city_patterns.code.local_composition_audit \
  --augment-pairs --augment-triplets --device cuda:0
```

A provisional local match requires all two or three units to have connected
responses of at least eight patches in one heading, each at least 35% of that
unit's panorama peak, with component gaps no greater than five patches and a
joint box covering no more than half of one 28x28 view. Results and four-view
highlights are under the follow-up stage `local_composition_audit/` on NAS.
Of 1,110 selected combination instances, 531 pass this geometric rule: 446/786
pairs and 85/324 triplets. At least two sampled cities pass for 113/131 pairs
and 23/54 triplets. These are conditional retrieval-audit counts, not local
prevalence estimates for the full corpus. Inspect the images before assigning
landscape meaning: some Qwen labels and broad sky/road responses are visibly
incorrect or too coarse, and compact SAE regions are not object boundaries.

## Historical roadside composition audit (older semantic snapshot)

`code/composition_roadside_audit.py` samples up to six distinct-city panoramas
for each current composition with at least two side-eligible units. Road surfaces,
crosswalks, bridges/rail/tunnels and crossing power lines are not treated as
objects confined to a single side. Frozen DINOv3/SAE responses and an independent
Cityscapes road segmenter are recomputed on GPU; each panorama is processed only
once. A provisional same-side or opposite-sides label requires stable road-axis
and unit-side decisions over 75, 90 and 105 degree FOV assumptions. It cannot
establish object instances or citywide roadside prevalence. All final labels
remain undetermined pending acquisition calibration and semantic mask review.

```bash
python -u -m experiments.vitl30_city_patterns.code.composition_roadside_audit \
  --device cuda:0 --per-pattern 6
```

The stage writes a source identity, per-pattern audit CSV, summary, selected
four-view images and activation/road segmentation masks. The website publishes
the statistics and bounded evidence at `road-sides/`, leaving the whole sampled
image corpus on the analysis server.

Completed audit: 118 of 185 compositions have at least two side-eligible units;
67 are not applicable. Six distinct-city retrievals per eligible composition
produce 708 pattern instances (707 distinct panoramas across 30 cities). The
automatic geometry yields 21 same-side hypotheses, 21 opposite-sides hypotheses
and 666 undetermined cases. At the composition level, 17 have at least one
same-side hypothesis, 18 have at least one opposite-sides hypothesis, two have
both, and 85 have no side hypothesis in the six-example audit. All 708 final
labels remain undetermined until calibration and semantic review. Do not divide
21 by 708 and present it as a citywide roadside rate.

## Select panoramas

```bash
PYTHONPATH=. python -u -m experiments.vitl30_city_patterns.code.sample_grid
```

The selection manifest and per-city Parquet files are written to
`/host/root/mnt/nas/huangyj/GoogleSV/urban_visual_gene/vitl30_city_patterns_1km10/selection`.
The command resumes completed cities after verifying file hashes. The manifest
reports eligible metadata rows, occupied cells, cells with fewer than 10
candidates, and selected panoids per city. The metadata availability flag and
recorded four-image count gate candidates; actual JPEG availability is checked
during inference.

The completed selection contains 688,174 candidate panoramas in 72,630
occupied cells across 30 cities, drawn from 24,602,382 metadata rows. Of these
cells, 7,299 had fewer than 10 eligible panoids. Reprojecting every selected
coordinate independently reproduced its saved grid ID, and no cell exceeds
the 10-panorama cap.

## Extract SAE features

```bash
bash experiments/vitl30_city_patterns/code/launch_full.sh
```

This starts one shard on each of four GPUs. Each worker reads the same
selection manifest, writes atomic result chunks, and resumes verified chunks.
Results, logs, and progress files are under
`/host/root/mnt/nas/huangyj/GoogleSV/urban_visual_gene/vitl30_city_patterns_1km10/extraction`.
The quality gate rejects panoramas with any black direction or at least two
tunnel-like directions; blur is retained. The selection and extraction roots
are separate from all older pilots and full-census code.

After all shards complete, city-pattern analysis must account for the grid
sampling design and compare cities using per-cell summaries rather than raw
panorama totals. A completed shard alone is not a complete city sample.

## Automatic handoff after extraction

```bash
bash experiments/vitl30_city_patterns/code/launch_followup.sh
```

This detached monitor checks the four shard status files every five minutes.
It starts the next stage only after all four report `complete`, the selected
count matches, every expected chunk has a matching run identity and accounting,
and each processed panoid/grid pair matches its selection record. The monitor
is idempotent and writes `followup/status.json` and `followup/followup.log`.
It also writes `followup/task_queue.json` immediately, showing completed,
running, waiting and blocked stages with their dependencies and selected-point
progress.
After validation it creates `followup/cells.parquet`, `followup/cities.csv`,
`followup/city_group_mass_preview.png` and `followup/next_tasks.json`.
The city/cell shares are normalized **group activation mass**, not dominant
patch prevalence or semantically validated landscape prototypes. They are a
screening product and must not replace thesis figures without re-analysis.

The task manifest records later candidate-combination search, roadside
classification, prototype distributions, and Chapter 5 figure replacement as
dependencies. The separate sequential runner starts these stages only after
city-cell screening has completed. Its artifact checks provide a reproducible
handoff, but semantic labels, roadside decisions and scientific claims still
need evidence review.

## Historical automatic research task queue

```bash
bash experiments/vitl30_city_patterns/code/launch_auto_continue.sh
```

The detached runner waits for `followup/status.json`, `complete.json` and the
queue to agree that screening finished. It then executes the four remaining
stages in dependency order by running the local Python module for each stage.
There is no Codex subprocess or model API in the queue supervisor. Each
stage writes its results, evidence and completion marker under
`followup/stages/<stage>/`. The runner checks the extraction identity and
required products before marking a stage complete and starting the next one.
An interrupted runner can be relaunched; verified completed stages are skipped.
Research or artifact-check failures are retried at most three times and then
left as `error`. If a stage itself needs a remote service, API rate limits
(HTTP 429) retry every 15 minutes, including when the runner resumes an
existing wait. Pure local stages do not wait for the former Codex rate limit.
Temporary server failures and
connection interruptions use exponential backoff and jitter, capped at 30
minutes. Service interruptions do not consume the three research attempts. The
attempt numbers continue across restarts so prior logs are preserved.
The active status includes the next retry time
while a service outage persists. It does not skip a failed dependency.

The implemented `candidate_combinations.py` reads the existing 4,096-dimensional
panorama means, caches eligible family mass by city, and tests family pairs in
held-out cities against marginal-preserving city and 5-km-block nulls. It does
not extract images, retrain models, call Qwen, or establish roadside geometry.
Qwen family labels remain provisional. Its completion means the co-presence
screen finished, not that landscape prototypes were validated.

All four scripts are implemented and run as local modules:

- `road_sides.py`: on GPU, recompute frozen panorama-aware SAE attribution and
  independent SegFormer road probabilities for 12 distinct-cell examples per
  retained pair, with at most three per city. Preserve original and overlaid
  four-view images. Perspective bearings and three assumed fields of view give
  tentative geometry only. Final side labels remain undetermined until source
  camera calibration and human mask/road review are available.
- `prototype_city_distribution.py`: apply the frozen provisional family
  co-presence rule to all 681,645 retained panoramas. Average within-cell
  positive fractions equally across cells, with 999 five-kilometre-block
  bootstrap draws. This is a distribution of co-presence candidates, not
  validated prototypes. Corpus-wide roadside rates remain missing.
- `chapter5_figures.py`: the completed historical export stage. Its installer
  now refuses to overwrite the thesis when the current semantic snapshot is
  present. Use `semantic_city_figures.py` for the current revision.

The computational queue can finish while scientific review remains pending.
No script trains a new SAE, calls Qwen or invokes a model API. GPU inference
has no CPU fallback; statistical aggregation and plotting use the CPU.

Read `followup/task_queue.json` for task states and
`followup/auto_continue_status.json` for the active stage. Logs are in
`followup/auto_continue.log` and `followup/stages/<stage>/attempt_N.log`.

## Analysis protocol and remaining scientific review

1. Audit every shard against the selection manifest. Report retained and
   rejected panoramas by city and cell, and check that four headings belong to
   the same panoid. Do not silently replace rejected panoramas.
2. Summarize SAE dimensions, reviewed element families, and candidate
   compositions within each 1-km cell. Give cells comparable weight within a
   city, and report spatial support as well as panorama counts. Keep uncertain
   Qwen labels separate from human-reviewed elements.
3. Retrieve recurring combinations using the pooled vectors as a first-pass
   index. The saved 4,096-dimensional panorama means and four heading/64-group
   vectors cannot establish exact patch layout or which side of a road holds an
   object. Re-run frozen DINOv3 and SAE on candidate panoramas to obtain patch
   activations and masks before making a spatial-relation claim.
4. Estimate the local road axis at each panorama from a verified road mask and
   direction (with matched road geometry where reliable). Map element-mask
   centroids through the calibrated 0/90/180/270-degree views to panoramic
   bearings. Assign each element to the signed left or right half-plane of the
   road axis, retaining uncertainty near the road axis or image seams. Label a
   pair **same roadside** only when both belong to the same confidently
   estimated side; label it **opposite roadsides** only when they belong to
   different sides and the road region separates them. The fact that two
   elements occur in different viewing directions is not a side label.
   Intersections, bends, multiple roads, occluded road surfaces and ambiguous
   axes remain **undetermined** until checked; they are excluded from side
   comparisons rather than forced into either class.
5. Discover and validate same-side and opposite-side combinations separately.
   Merge SAE dimensions that describe parts of one object before counting
   distinct elements. Report panorama, cell and city support; compare matched
   side-shuffled and spatial-block controls; check representative four-view
   masks by hand. Only then describe a candidate as a recurring landscape
   prototype. Re-apply validated detectors to the complete sampled corpus for
   city distributions; a gallery of retrieved examples alone is insufficient.

## Historical Chapter 5 audit plan (superseded by current semantic revision)

- Reconcile the methods, sample table and study-city map with the 30-city
  selection, actual quality-screened counts and Moscow's verified source-image
  mapping. Preserve the separate 89,031-panorama SAE training cohort in model
  training descriptions; do not relabel it as the new spatial analysis sample.
- Regenerate all data-dependent Chapter 5 figures from the finished analysis:
  shared/distinctive features, city composition and similarity, co-occurrence,
  1-km spatial structure, panorama UMAP, activation examples and VLM evidence.
  Check the pipeline and panorama-construction diagrams, training-loss curve
  and decoder hierarchy against their original training inputs; re-export them
  where needed, without implying the SAE was retrained on the larger sample.
- Add dedicated figures for each validated prototype: representative four-view
  original images and activation masks, same-side versus opposite-side layout,
  per-city prevalence with cell-level uncertainty, and within-city 1-km maps.
  Include underdetermined-roadside and low-support counts. Equalize cell
  contributions and distinguish raw sample counts from normalized prevalence.
- Update every affected caption, denominator, statistical claim and discussion
  in `/root/code/HKUST-Thesis/05_research3.tex`,
  `chapter5_vitl30_figures.tex` and `chapter5_vitl30_semantics.tex`; replace
  superseded assets only after checking each against the new run identity.
  Compile the thesis and compare its figures and captions with the published
  site. These are downstream tasks, separate from feature extraction; completion is recorded in the verified stage artifacts.

Current Chapter 5 asset inventory for that audit:

| Source figures | Required check |
| --- | --- |
| `figure/pipeline.jpg`, `figure/panorama-aware.jpg` | Confirm the diagrams describe the frozen model and the new downstream city analysis separately. |
| `Fig_ViTL30_Study_Cities.pdf` | Update all 30 city locations and mapping provenance. |
| `Fig_ViTL30_Training_Loss.pdf`, `Fig_ViTL30_Decoder_Hierarchy.pdf` | Retain the original SAE training provenance; regenerate only if their underlying training result or presentation changes. |
| `Fig_ViTL30_Shared_Distinctive.pdf`, `Fig_ViTL30_City_Composition.pdf`, `Fig_ViTL30_City_Similarity.pdf` | Recalculate with grid-balanced city summaries and revised denominators. |
| `Fig_ViTL30_Cooccurrence.pdf`, `Fig_ViTL30_Spatial_Structure.pdf`, `Fig_ViTL30_Panorama_UMAP.pdf` | Recalculate from the new sampled corpus and state the test/sampling unit. |
| `Fig_ViTL30_Activation_Examples.pdf`, `Fig_ViTL30_VLM_Interpretation.pdf` | Refresh only from verified patch-level image evidence; distinguish provisional labels from reviewed ones. |
| New prototype maps and masks | Show each prototype's city prevalence, 1-km distribution, and same-side/opposite-side examples with undetermined cases reported. |
