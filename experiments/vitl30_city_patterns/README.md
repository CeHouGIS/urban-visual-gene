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
stages in dependency order using one noninteractive Codex run per stage. Each
stage writes its code, results, evidence and completion marker under
`followup/stages/<stage>/`. The runner checks the extraction identity and
required products before marking a stage complete and starting the next one.
An interrupted runner can be relaunched; verified completed stages are skipped.
Failures are retried at most three times and then left as `error`, with the
stage log retained for diagnosis. It does not skip a failed dependency.

Read `followup/task_queue.json` for task states and
`followup/auto_continue_status.json` for the active stage. Logs are in
`followup/auto_continue.log` and `followup/stages/<stage>/attempt_N.log`.

## Analysis to do after extraction

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

## Chapter 5 figure and writing checklist

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
  site. These are pending tasks, not outputs of the current extraction job.

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
