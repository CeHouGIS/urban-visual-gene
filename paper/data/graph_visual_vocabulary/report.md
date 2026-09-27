# Multiview Graph Urban Visual Vocabulary

## Input correction

The analysis unit is a **complete four-direction panorama**, not an independent direction image. Four 640×640 views were jointly processed as 2560×640 → 896×224, producing a 14×56×512 latent tensor. The spatial view therefore uses all 784 circular panorama positions (P = 512×784). No DINOv3 or Feature-MAE retraining/re-inference was performed.

- Panoramas: 296,462
- Patches: 232,426,208
- Cities: 30

## A. Visual-relation graph

- Nodes: 512
- Edges: 24,673
- Density: 0.1886
- Connected components: 1
- Degree min/mean/max: 58 / 96.38 / 133

## B–C. Stable resolutions and selected scales

A single strong non-trivial plateau was detected at γ=2.65–2.75 using the transparently relaxed threshold 0.85. Mean seed NMI=0.856; mean adjacent-resolution NMI=0.886.

- Stable primary/coarse scale: γ=2.70, K=35
- Exploratory secondary/fine scale: γ=2.85, K=39
- Only one strongly stable plateau was identified; the fine scale is explicitly exploratory, not presented as a second stable plateau.
- Community numbers emerged from the resolution landscape and were not specified in advance.

## D. Independent visual coherence

- Within-community DINO similarity: 0.3530
- Between-community DINO similarity: 0.2989
- Difference: 0.0541
- Cohen's d: 0.414
- Size-preserving permutation p: 0.000999 (1000 permutations)

## E. Diagnostic comparison with Ward

- old_Ward_64_vs_new_fine: NMI=0.557, ARI=0.119 (64 vs 39 groups)
- old_Ward_32_vs_new_coarse: NMI=0.497, ARI=0.149 (32 vs 35 groups)

## F. Downstream conclusions

- 36/39 fine communities occur in every city; prevalence spans 0.000000–0.1897.
- Strongest city composition pair: Jakarta–Manila (cosine=0.997); weakest: Dhaka–Osaka (cosine=0.893).
- Global PPMI retains 104 positive supported recurring pairs; strongest: F009–F028 (1.50), F018–F028 (1.23), F013–F014 (0.70), F007–F028 (0.68), F008–F011 (0.57).
- Composition similarity and co-occurrence similarity remain distinct (city-pair Spearman ρ=0.457).
- Old/new spatial rank agreement: homogeneity ρ=0.653; continuity ρ=0.664.
- Highest new homogeneity: Singapore; highest local continuity: Bogota.

Semantic names were added only after all graph partitions were fixed and never entered graph construction or community detection.

## Outputs

- Data: `paper/data/graph_visual_vocabulary/`
- Figures: `paper/figures/graph_visual_vocabulary/`
- Heavy reusable arrays: `outputs/experiments/dinov3_multicity/graph_visual_vocabulary/`
