# Urban Visual Gene — Research Progress Presentation

## Slide 1 — Title and research question

**Title:** Urban Visual Gene: Discovering Interpretable Visual Primitives of Cities

**Core question:** Can cities be represented by a shared vocabulary of interpretable visual genes?

**Visual:** A montage of contrasting street views from Hong Kong, Singapore, Amsterdam, Cape Town, Paris, and Dhaka.

**Takeaway:** Urban identity may arise from different combinations of shared visual primitives.

## Slide 2 — Research motivation

- Conventional city classification produces a label but does not explain which visual elements define a city.
- We define a **visual gene** as a recurring local visual pattern learned across street-view images.
- A city's **visual genome** is the frequency, combination, and spatial distribution of these genes.

**Visual:** `street-view images → visual genes → city genome → cross-city comparison`.

## Slide 3 — End-to-end research pipeline

1. Define urban boundaries.
2. Sample the road network spatially.
3. Retrieve Google Street View panoramas and four directional views.
4. extract DINOv3 patch embeddings.
5. Train a BatchTopK sparse autoencoder.
6. identify stable visual genes.
7. Build a statistical taxonomy and city-level genome profiles.

**Status bar:** Data acquisition substantially complete; SAE training and stability analysis complete; external validation pending.

## Slide 4 — Global street-view dataset

- 110 cities across multiple world regions.
- Approximately 40.76 million downloaded panoramas.
- Approximately 163 million directional images.
- 6,876 committed and verified WebDataset shards.
- Urban boundaries, roads, sample points, and panorama metadata are complete for the monitored cities.

**Visual:** World map plus four large-number cards.

**Takeaway:** The project now has a scalable global data foundation rather than a small-city case study.

## Slide 5 — Sparse autoencoder design

**Pipeline:** `DINOv3 patch embedding → SAE encoder → 1,024 latent genes → Top-8 activation → reconstruction`.

**Working configuration:**

- Model: BatchTopK SAE
- Dictionary width: 1,024
- Active genes per patch: 8
- Training subset: 12 cities × 1,200 images
- Total: 14,400 images and approximately 11.29 million patches
- Training length: 60 epochs

**Takeaway:** Sparsity makes the dense vision representation decomposable into a small number of candidate visual genes per patch.

## Slide 6 — Model selection

| Configuration | Approx. reconstruction cosine | Main trade-off |
|---|---:|---|
| W1024/K4 | 0.809 | Very sparse, larger information loss |
| **W1024/K8** | **0.849** | **Preferred balance** |
| W1024/K16 | 0.881 | Better reconstruction, less sparse |
| W2048/K8 | 0.858 | Larger vocabulary, many unused genes |
| W2048/K16 | 0.891 | Best reconstruction, greater complexity |

**Takeaway:** W1024/K8 is the preferred working model, not an absolute optimum; it balances reconstruction, sparsity, utilization, and interpretation.

## Slide 7 — Reproducibility across random seeds

- Five independent runs: seeds 11, 23, 37, 53, and 71.
- All five completed 60 epochs.
- Final training loss is tightly concentrated between 0.154 and 0.155.
- 875 stable gene families were matched across runs.
- 738 families were recovered in all five seeds; another 137 appeared in four seeds.
- Almost every latent dimension was active in every run.

**Visual:** Five seed-specific dictionaries converging into stable gene families, plus exemplar images.

**Takeaway:** Much of the learned structure is reproducible and is not merely an initialization artifact.

## Slide 8 — From genes to an urban visual taxonomy

- 1,024 candidate genes organized into 32 coarse and 64 fine statistical clusters.
- The taxonomy uses decoder, encoder, position, city, and activation-context evidence.
- Current roles include stable structured, city-specific, rare stable, weakly structured, unstable, and nuisance components.
- The city genome combines gene prevalence, diversity, co-expression, and spatial organization.

**Visual:** A hierarchy or Sankey diagram with universal, regional, and city-specific exemplar genes.

**Takeaway:** Urban identity appears to combine universal components with city-specific gene mixtures.

## Slide 9 — Current limitations and next steps

**Current limitations**

- The main SAE experiment currently covers 12 cities, not the full 110-city corpus.
- Stability does not by itself prove semantic meaning.
- Some latent dimensions encode position, image quality, or acquisition artifacts.
- Final nuisance-control and external-validity analysis is incomplete because the latest analysis job encountered a read-only array assignment error.

**Next steps**

1. Fix and rerun external validation.
2. Separate robust semantic genes from nuisance and unstable components.
3. Expand encoding to the global city corpus.
4. Construct comparable city-genome profiles.
5. Test relationships with urban morphology, geography, and socioeconomic variables.

**Closing statement:** The project has progressed from data construction to reproducible gene discovery; scientific validation and global scaling are the immediate priorities.

## Optional appendix

- Savio job and pipeline status
- Full SAE width/Top-K ablation table
- Seed-matching thresholds
- Data quality and boundary definitions
- Additional visual-gene exemplars
- WebDataset verification protocol

