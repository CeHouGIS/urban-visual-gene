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

## Completed computational outputs

All seven queue entries, including extraction and downstream stages, have
passed their artifact gates. The usable sample contains 681,645 panoramas in
72,625 observed cells. The pair screen tested 76 candidates and retained seven
provisional family co-presence candidates. City rates cover 210 candidate/city
combinations. An 84-example four-view audit has frozen-SAE masks and independent
road probabilities; all final side labels remain undetermined. Chapter 5 figures
and captions have been refreshed and the complete thesis compiles successfully.
The public page is `https://cehougis.github.io/urban-visual-gene/city-patterns/`.

The procedures below document the completed computation and the remaining
semantic, calibration and roadside review. A completed compute queue does not
mark these scientific reviews complete or validate a landscape prototype.

## Composition revision: recurrent patterns rather than excess pairs

The current web atlas uses `code/composition_patterns.py`. The earlier seven
significant pairs answer a different question and remain in the thesis and
downloadable reference; they are not the entire landscape composition catalogue.

The revision merges 236 provisional Qwen fine labels into 103 raw canonical
response units, merging parts of the same object and retaining whole-object
subtypes. A declared landscape scope retains 78 units and excludes 25: people,
all vehicles and their parts, containers, construction machinery/scaffolds/cranes,
and listed movable or temporary objects. Fixed buildings, vegetation, road and
pedestrian space, barriers, water/terrain and fixed infrastructure remain.
Storage tanks remain as fixed industrial structures. This reversible analysis
filter does not delete SAE dimensions. Excluded dimensions cannot contribute
through a parent fallback; the all-4096 mass denominator is unchanged.
Each parent family contributes at most one subtype per panorama. Positive-share
60th-percentile thresholds are fitted on the 22 discovery cities, with cities,
observed cells and panoramas within cells balanced. Parent mass must be at least
0.5% of all-4096 activation mass; subtype mass must be at least 0.1%. Thresholds
are frozen for all cities. They measure relative response, not semantic presence.
An initial permissive 0.01% floor admitted widespread weak water/container
responses and was rejected; its diagnostic outputs remain server-side under
`composition_patterns_permissive_probe`, outside the published atlas.

Apriori mines cross-parent combinations of order 2–4 with at least 0.5% weighted
support, 30 panoramas, eight cells and three cities in both city partitions.
Near-redundant subsets are pruned if a discovered extension retains 90% support.
The final 185 combinations contain 131 pairs and 54 triples; no quadruple passes
the complete recurrence and redundancy rules.
All 30 cities and 72,625 observed cells are mapped, including sampled zero-match
cells. Counts retain the actual cell denominator (one to ten panoramas).

Raw panorama retrieval coverage is 78.5%; equal-city/equal-cell coverage is 81.3%.
These are not patch explanation rates or validated object recall. At quantiles
50/60/70, freezing the same 185 patterns yields weighted coverages 90.2/81.3/64.4%;
182/185/118 patterns repeat in the evaluation cities. This sensitivity is
published rather than choosing the threshold that maximizes category counts.
Evaluation cities have already been inspected, so this is an exploratory
revision. Descriptive spatial-block lift is not a fresh significance claim.

```bash
python -u -m experiments.vitl30_city_patterns.code.composition_patterns \
  --run-identity 6d6519a1cb0b78d2fbc384722030a3a3fb929764724630c016cf2ddb3c34e617
python -u -m experiments.vitl30_city_patterns.code.composition_evidence \
  --run-identity 6d6519a1cb0b78d2fbc384722030a3a3fb929764724630c016cf2ddb3c34e617 \
  --device cuda:0
```

The second command recomputes four-view patch attribution on GPU for 14
representative combinations, three distinct-city retrieval examples each. The
deterministic hash selection is not a prevalence audit. Masks are SAE attribution,
not object segmentations. No API calls, new SAE training or CPU inference are used.

The previous mobile-object draft is archived server-side under
`composition_patterns_with_mobile_objects_*`; its categories and images are
replaced on the public composition page. Raw score cache identity is independent
of the eligibility filter, but still checks the exact hierarchy, unit dimensions,
corpus and scoring version. The current mining protocol includes all exclusions.
Visual spot checks found some water responses on sky and bridge responses on
sign gantries. The site flags these examples and their compositions rather than
treating the response maps as actual object distributions. The small spot check
does not estimate accuracy or justify dropping entire SAE dimensions.

Remaining scientific tasks: review the labels and masks; calibrate response
thresholds against manually labeled positives and negatives; evaluate on new
spatial blocks or external data; then establish object counts, layout and road
sides before replacing confirmatory thesis claims. See the website progress panel.

## Roadside composition audit

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

## Automatic research task queue

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
- `chapter5_figures.py`: regenerate city composition, similarity, cell PPMI,
  spatial structure, a cell UMAP, candidate maps/intervals and four-view
  evidence. Retain original training and semantic provenance. Update captions
  and methods together, inventory inputs, and compile the complete thesis.

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

## Chapter 5 figure and writing audit

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
