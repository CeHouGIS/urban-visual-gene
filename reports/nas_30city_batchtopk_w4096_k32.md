# BatchTopK SAE on a replacement 30-city street-view cohort

## Scope

The original Feature-MAE JPEG/TAR and token cohort is not mounted here. This
experiment uses a newly sampled NAS Google Street View cohort. Its figures are
SAE analogues of the archived Feature-MAE figures, **not paired comparisons**
on the same images. The exact sample and provenance are in
`outputs/experiments/nas_30city_batchtopk_w4096_k32/manifests/batch.json`.

## Data and model

| Item | Result |
| --- | ---: |
| Cities | 30 |
| Four-heading panoramas before quality screening | 92,692 |
| Images before quality screening | 370,768 |
| Images after black/tunnel screening | 363,700 |
| Images excluded by screening | 7,068 |
| DINOv3 patch representation | ViT-B/16, 14 x 14 x 768, L2-normalized |
| BatchTopK SAE | 768 input, 4,096 dictionary features, K=32 |
| Training | 10 epochs, 3,200 steps/epoch, four images per city/step |
| Best sampled validation squared L2 | 0.109847 |
| Best sampled validation cosine similarity | 0.943644 |

Each of the four directions is encoded separately by DINOv3. The SAE sees
individual patch vectors. At inference, an image's feature score is the mean
of its 20 strongest activations over 196 patches. The decoder vectors are
clustered by cosine distance into nested 32- and 64-cluster vocabularies.
Berlin (569 panoramas) and Chicago (2,523) did not reach the 3,200-panorama
target. Balanced training resamples shorter cities, so the number of training
draws should not be mistaken for the number of unique images.

The validation figure is calculated from sampled patches of panorama-disjoint
held-out data. It is not a full validation pass over every held-out patch.

## Figures and interpretation

The output directory contains 29 new PNG/PDF figure pairs. Of 37 archived
Feature-MAE figure names, 25 have a computed SAE analogue. The authoritative
mapping and method differences are in `figure_inventory.json`. In particular:

- The SAE decoder hierarchy is not the old E+D+P relation graph.
- The 64-cluster PPMI and city-composition views are not measurements on the
  old 512-dimensional Feature-MAE representation.
- The city-specific pair chart uses a city-versus-rest joint-rate ratio;
  same-image examples require at least four winning patches for each cluster.
  Neither figure reproduces the old E+D+P visual-distance weighting.
- The spatial comparison uses 1 km Web Mercator cells containing at least five
  panoramas and plots only cities with at least 20 adjacent valid grid pairs.
  The CSV retains values for all 30 cities, including sparse cases.
- The named clusters are numeric IDs. Their semantic meanings need validation
  against the original images before drawing substantive landscape conclusions.

Twelve archived figures remain uncomputed: one city archetype heatmap, three
compact graph figures, four other graph archetype figures, two image-layout
figures, and two Mask2Former segmentation figures. These need new graph/layout
models or a segmentation run on the replacement images. No archived MAE image
was copied into the SAE output.

## Files

- `outputs/experiments/nas_30city_batchtopk_w4096_k32/model_best.pt`: best SAE checkpoint.
- `outputs/experiments/nas_30city_batchtopk_w4096_k32/figures/Fig_BatchTopK_Training_Loss.png`: training/validation loss.
- `outputs/experiments/nas_30city_batchtopk_w4096_k32/figures/Fig_Image_Level_Atypical_Coactivation.png`: four-heading original image and two-cluster activation overlays.
- `outputs/experiments/nas_30city_batchtopk_w4096_k32/figures/Fig_Homogeneity_vs_Local_Continuity.png`: city spatial comparison.
- `outputs/experiments/nas_30city_batchtopk_w4096_k32/results/`: city composition, pair and spatial tables.
- `outputs/experiments/nas_30city_batchtopk_w4096_k32/figure_inventory.json`: every archived figure and its SAE counterpart or missing-analysis status.

To reproduce from the saved manifests and caches, run
`python -m sae_experiments.legacy_mae_30city.run_nas_experiment --train-device cpu`
from the repository root. The runner detects the completed checkpoint and
inference cache, then regenerates all figure stages.
