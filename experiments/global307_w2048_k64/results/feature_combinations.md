# SAE feature combinations: first held-out significance test

## Cohort and model

- Model: formal BatchTopK SAE, D=1024, dictionary W=2048, k=64.
- Held-out sample: 11,800 complete four-view panoramas, 40 in each of 295 cities. These panoramas were excluded from the 30M-patch SAE training cache.
- Every panorama is encoded from all 3,136 DINOv3 patches (28×28 per view), with a separate BatchTopK budget of 64×3,136. `city_counts/*.npy` stores the number of selected patches per feature and panorama.
- Cities are split by fixed seed into 207 discovery and 88 validation cities. Each feature's activation threshold is the 90th percentile of panorama patch count in discovery cities, with minimum 1.

## Pair test

Candidate pairs are selected **only** in discovery cities: at least 30 coactive panoramas, at least 15 above the simple marginal expectation, and SigLIP crop embedding cosine below 0.85. The 100 largest discovery excesses are carried unchanged to validation.

In validation cities, the null independently reassigns the second feature's panorama labels within each city and panorama richness quartile. This preserves the feature's frequency in each stratum and accounts approximately for scenes with more active SAE features. It does **not** preserve exact per-panorama degree. We use 9,999 permutations, one-sided excess-cooccurrence p-values, and Benjamini–Hochberg correction over the 100 fixed candidates.

All 100 candidates pass q<0.05; all achieved the minimum possible permutation p and q of 0.0001. Median validation lift is 2.131, and median city support is 62 of 88 validation cities. The complete effect sizes and supports are in `pair_results.csv`.

The semantic filter matters. An initial unfiltered ranking also produced 100 statistically significant pairs, but their SigLIP semantic cosine had median 0.969: they were dominated by near duplicates. The reported 100 candidates have semantic cosine below 0.85 (median 0.819). This cutoff is a heuristic screen, not proof of distinct human semantic categories.

## Spatial follow-up

For the first five reported pairs, we re-encoded up to 120 coactive validation panoramas per pair, with at most two panoramas per city. We measured the fraction of feature B's selected patches within two patch cells of feature A. The reference randomly circularly shifts B around the 28×112 panorama ring 199 times, retaining each map's shape and activation count. This analysis is exploratory; the five pairs were chosen using cooccurrence, and the spatial p-values are not confirmatory.

| Pair | Panoramas | Observed adjacency | Shifted adjacency | Spatial lift | Exploratory p |
|---|---:|---:|---:|---:|---:|
| 211 + 1546 | 120 | 0.0080 | 0.0144 | 0.557 | 1.000 |
| 1470 + 1875 | 120 | 0.0894 | 0.0915 | 0.977 | 0.850 |
| 13 + 1546 | 119 | 0.0034 | 0.0038 | 0.895 | 0.815 |
| 542 + 1012 | 87 | 0.0157 | 0.0172 | 0.911 | 0.825 |
| 823 + 2014 | 108 | 0.9449 | 0.4916 | 1.922 | 0.005 |

Thus cooccurrence is robust in this sample, but four of five example pairs do not show the tested local spatial arrangement. Pair 823+2014 is a candidate for further spatial validation. Its high adjacency may still reflect overlapping visual concepts; human review and an independent spatial test are needed before naming a street-scene template.

## Original-image evidence and limits

`validation_original_examples.jpg` shows four original headings for three validation panoramas from different cities for each of five pairs. These examples are selected for strong joint activation, so they illustrate combinations and do not estimate prevalence. They are not activation overlays; spatial attribution comes from the patch-level follow-up above.

The analysis concerns association among SAE features, not causal composition. Panoramas within a city can be spatially dependent. The city-restricted permutation addresses city-wide prevalence differences but the p-values should not be interpreted as independent street locations. SigLIP crop vectors are a proxy for semantics and the 0.85 cutoff has no universal meaning. Further template claims should use independent city samples and a prespecified spatial metric.

## Reproduce

```bash
python -m formal.analysis.test_feature_combinations --stage all --device cuda:0
python -m formal.analysis.plot_feature_combination_examples
python -m formal.analysis.test_feature_pair_spatial
```

The count cache makes `--stage test` sufficient for changing candidate or null settings. Checkpoint, anchor manifest, and semantic embedding hashes are recorded in `reports/data/feature_combinations_w2048_k64/`.
