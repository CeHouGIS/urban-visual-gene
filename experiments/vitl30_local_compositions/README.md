# Panorama-level landscape relationship mining

This experiment searches for recurring **relationships among local visual
elements in a complete four-view panorama**. The unit of analysis is one
panorama with its 0°, 90°, 180°, and 270° views. A candidate can therefore span
two views and can contain repeated families, such as `building + tree + tree`.
The output is an exploratory candidate-motif catalogue; it is not an object
segmentation benchmark and it does not by itself establish a confirmed
landscape prototype.

## Data and fixed representation

The script reuses the frozen ViT-L/BatchTopK SAE sparse-activation cache from
the 30-city main experiment. It does not extract DINO features again and does
not retrain the SAE. Each perspective view is represented by a 28×28 patch
grid. The current run contains 1,856 complete panoramas from 29 cities:

- 1,392 training panoramas;
- 464 development panoramas;
- 7,424 directional images (four per panorama).

The sealed test partition is not read. The checkpoint, corpus, split, hierarchy,
and sample hashes are recorded in `outputs/panorama_motifs/results.json`.

## Panorama construction and relation key

SAE dimensions are first grouped into provisional visual families using the
existing post-hoc interpretation hierarchy. Family regions are obtained from
the sparse patch activations and are used as **feature masks**, not as object
instance masks.

For a region centroid at horizontal patch coordinate `x` in a view with heading
`h`, the experiment uses an explicit 90° field-of-view assumption:

\[
\theta = h + \left(\frac{x}{27}-\frac12\right)90^\circ \pmod{360^\circ}.
\]

This puts all four views on one circular horizontal coordinate, so the 270° and
0° views can be adjacent. A relation key contains the sorted family names and
the pairwise descriptors: circular angle bin (`near`, `adjacent`, `wide`,
`very_wide`, or `opposite`) plus relative vertical position (`up`, `level`, or
`down`). Family permutations are canonicalized, which makes the two trees in a
`building + tree + tree` candidate interchangeable. Repeated observations of
the same family at a view boundary are filtered when their masks or bearings
indicate a duplicate.

The four-view strip in the gallery is a readable display of four perspective
images. It is not an equirectangular reprojection or a claim that the source
images are geometrically rectified.

## Candidate search and statistical check

The search enumerates three-node combinations over all four directions, allowing
repeated families. A training candidate must occur in at least 30 panoramas and
8 cities. The frozen candidates are then evaluated on the development split.
Two exploratory nulls are generated 99 times while preserving city and heading
marginals:

1. shuffle family labels to test whether the family co-occurrence is above the
   marginal rate;
2. shuffle the within-panorama geometry to test whether the layout is above the
   co-occurrence rate.

P-values are corrected with BH-FDR over all frozen training candidates. With 99
   permutations, the smallest attainable empirical p-value is 0.01. These
   tests provide prioritization evidence only; they are not independent semantic
   validation.

## Current output

### Structured-label rerun

`structured_motifs.py` reuses the same 1,856-panorama train/dev sample and
four-view sparse activations. It assigns each candidate region the first
eligible Qwen fine label among its strongest contributing SAE dimensions.
Only labels explicitly linked in `docs/site/label_structure_rules.json` enter
this audit. An `is_a` subtype keeps its name; a `part_of` component is mapped
to the containing entity for the displayed fine composition. Regions from the
same building, vehicle, road, or other non-tree family cannot count as separate
objects. Two tree regions can contribute separately only if their bearings are
at least 30 degrees apart, and still require manual instance review.

The search retains the old rotation-invariant three-region layout key,
city-balanced sample, train-selected candidates, development-set co-occurrence
and layout permutations, and BH correction. Fine-label composition counts are
descriptive and are **not** individually tested by the broad-layout q-values.
No sealed test activations are read. The output is in `outputs/structured_motifs/`
and replaces the broad-family gallery on the current site when present.

```bash
PYTHONPATH=. python -u -m experiments.vitl30_local_compositions.structured_motifs \
  --permutations 99
```

The Qwen taxonomy is provisional. A component mapping is a textual guard
against double counting, not an instance segmentation result. A separate
building and its wall may also be incorrectly merged by this conservative
rule; manual review is needed before calling any layout a landscape prototype.

The reproducible run is stored under
`outputs/panorama_motifs/`:

- `results.json` contains the protocol, counts, candidate summaries, p/q-values,
  limitations, and selected gallery examples;
- `sample.csv` records the panorama split used by the run;
- `gallery/` contains four-view strips with provisional family masks;
- `QUALITATIVE_REVIEW.md` records the manual review of the principal relations.

The run found 58,595 observed relation keys and 1,583 training candidates. The
gallery contains 16 selected relation layouts. The five priority signatures
and their training/development support are:

| Relation signature | Train panoramas / cities | Development panoramas / cities |
| --- | ---: | ---: |
| `building + tree + tree` | 222 / 29 | 71 / 27 |
| `building + pedestrian + tree` | 42 / 18 | 19 / 14 |
| `barrier + road + tree` | 146 / 28 | 52 / 25 |
| `building + road + utility` | 163 / 26 | 61 / 23 |
| `low_vegetation + pedestrian + tree` | 42 / 19 | 18 / 13 |

The strongest qualitative example is `building + tree + tree`: the building can
be in one heading while the two tree regions occur in other headings of the
same panorama. The masks still need independent inspection because one physical
tree can be split into multiple feature regions. The other signatures are
useful search targets but include broad families such as road or pedestrian
space and should not yet be named as stable landscape prototypes.

## Reproduce

Run from the repository root after the frozen activation cache and its metadata
are available:

```bash
PYTHONPATH=. python -u -m experiments.vitl30_local_compositions.panorama_motifs \
  --out experiments/vitl30_local_compositions/outputs/panorama_motifs \
  --permutations 99 --max-gallery 16 --examples 6 --seed 20261017
```

The default sample is the committed experiment sample configured by the
shared setup code. Do not include the sealed test split. Any change to the
sample, family hierarchy, checkpoint, FOV, or permutation count must be written
to `results.json` and reviewed before comparing runs.

## Interpretation and next checks

The current output should be read as a set of candidate panoramic relations.
Before calling one a landscape prototype, the next checks are:

1. verify that every mask corresponds to a coherent visual element and that
   repeated family regions are independent instances;
2. rerun with additional seeds or a larger permutation budget;
3. hold out cities and spatial blocks for stability testing;
4. compare each three-way relation with its pairwise components to measure the
   extra information supplied by the combination;
5. use an independent VLM or human annotation pass to describe the masks after
   discovery, without feeding those descriptions back into the SAE search.

The older `complex_motifs.py` script is retained only as a historical,
single-view comparison. It is not part of the current main experiment and its
counts must not be merged with the panorama-level results.
