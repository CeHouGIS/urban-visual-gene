# Top-4 Feature-MAE Typed Semantic Hierarchy

## Scope

- Cities: `30`
- Complete four-direction panoramas: `296,462`
- Panorama patches: `232,426,208`
- Winner assignments: `929,704,832`
- Discovery input: per-patch Top-4 dimension identities only; activation magnitudes are discarded after ranking.
- Mapillary semantics validate Top-X and provide post-hoc names/audits; they do not enter relation scores or hierarchical linkage.

## Why Top-4

Top-4 is the smallest candidate reaching `95.9%` dimension coverage while preserving `39.9%` same-parent pair purity and `0.889` cross-city marginal cosine stability.

## Typed hierarchy

- Macro parents selected from the hierarchy: `48`
- High-confidence AND pairs: `487`
- High-confidence OR pairs: `12`
- Panorama presence threshold: `16` of 784 patches
- The reported macro cut is the coarsest evaluated cut whose largest parent contains no more than 25% of the 512 dimensions; the full dendrogram is retained.

AND means positive city-conditioned panorama co-occurrence. OR requires negative city-conditioned co-occurrence together with similar context-role and spatial-position signatures. Unresolved pairs are not forced into either class.

OR is a sparse lateral relation: an OR pair may connect dimensions assigned to different macro parents. It should not be read as requiring an orange dendrogram branch. Macro-parent labels are post-hoc semantic audits, not semantic supervision or ground-truth classes.

## Macro-cut audit

| requested_clusters | actual_clusters | silhouette_cosine | singleton_cluster_share | largest_cluster_size | largest_cluster_share | selection_score |
| --- | --- | --- | --- | --- | --- | --- |
| 8 | 8 | 0.152 | 0.5 | 442 | 0.8633 | 0.102 |
| 12 | 12 | 0.1233 | 0.4167 | 441 | 0.8613 | 0.08167 |
| 16 | 16 | 0.1118 | 0.375 | 221 | 0.4316 | 0.07428 |
| 24 | 24 | 0.1009 | 0.2917 | 203 | 0.3965 | 0.07168 |
| 32 | 32 | 0.09391 | 0.2188 | 174 | 0.3398 | 0.07204 |
| 48 | 48 | 0.09398 | 0.25 | 126 | 0.2461 | 0.06898 |
| 64 | 64 | 0.108 | 0.2969 | 92 | 0.1797 | 0.07836 |

## Largest macro parents

| macro_parent_id | posthoc_semantic_label | dimension_count | global_top4_patch_assignments | and_pairs | or_pairs | top_semantic_1 | top_semantic_1_share | top_semantic_2 | top_semantic_2_share | top_semantic_3 | top_semantic_3_share |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| P01 | Building / Vegetation | 126 | 325407709 | 147 | 2 | Building | 0.334 | Vegetation | 0.1565 | Road | 0.1127 |
| P02 | Vegetation / Road | 90 | 91026226 | 134 | 0 | Vegetation | 0.6371 | Road | 0.128 | Terrain | 0.1236 |
| P03 | Sky / Road | 86 | 421097244 | 17 | 1 | Sky | 0.4125 | Road | 0.2235 | Vegetation | 0.07789 |
| P04 | Vegetation / Bridge | 35 | 20531979 | 43 | 0 | Vegetation | 0.3068 | Bridge | 0.1728 | Truck | 0.08262 |
| P05 | Road / Building | 18 | 50116867 | 0 | 0 | Road | 0.3038 | Building | 0.2544 | Vegetation | 0.09641 |
| P06 | Road / Sky | 14 | 17862126 | 2 | 0 | Road | 0.7074 | Sky | 0.1218 | Building | 0.05422 |
| P07 | Building / Bus | 13 | 463328 | 0 | 0 | Building | 0.261 | Bus | 0.1319 | Truck | 0.1156 |
| P08 | Building / Vegetation | 12 | 1325219 | 0 | 0 | Building | 0.3334 | Vegetation | 0.3178 | Sky | 0.1187 |
| P09 | Vegetation / Road | 6 | 101796 | 0 | 0 | Vegetation | 0.3875 | Road | 0.1691 | Building | 0.07194 |
| P10 | Building / Sky | 6 | 43337 | 0 | 0 | Building | 0.4405 | Sky | 0.4171 | Vegetation | 0.0665 |
| P11 | Building / Road | 5 | 1290 | 0 | 0 | Building | 0.1912 | Road | 0.1541 | Wall | 0.1384 |
| P12 | Road / Building | 5 | 172113 | 0 | 0 | Road | 0.4648 | Building | 0.1088 | Sidewalk | 0.07942 |

## Strongest AND pairs

| dimension_i | dimension_j | and_score | city_conditioned_log2_lift | role_similarity |
| --- | --- | --- | --- | --- |
| D073 | D356 | 3.038 | 3.394 | 0.7854 |
| D235 | D344 | 2.61 | 2.699 | 0.794 |
| D073 | D182 | 2.466 | 2.598 | 0.8182 |
| D182 | D356 | 2.375 | 2.439 | 0.8077 |
| D182 | D215 | 2.283 | 2.289 | 0.8157 |
| D215 | D356 | 2.223 | 2.24 | 0.7399 |
| D073 | D215 | 2.189 | 2.192 | 0.8334 |
| D369 | D500 | 2.13 | 2.45 | 0.6697 |
| D179 | D235 | 2.122 | 2.122 | 0.7941 |
| D073 | D209 | 2.104 | 2.98 | 0.7784 |

## Strongest OR pairs

| dimension_i | dimension_j | or_score | city_conditioned_log2_lift | role_similarity | position_similarity |
| --- | --- | --- | --- | --- | --- |
| D050 | D175 | 0.4173 | -1.345 | 0.3273 | 0.901 |
| D250 | D294 | 0.3023 | -1.237 | 0.2675 | 0.8366 |
| D101 | D232 | 0.2862 | -0.9508 | 0.3476 | 0.7777 |
| D157 | D212 | 0.211 | -0.681 | 0.348 | 0.8479 |
| D113 | D369 | 0.2025 | -0.6253 | 0.367 | 0.7786 |
| D274 | D362 | 0.1953 | -0.8749 | 0.317 | 0.8644 |
| D362 | D426 | 0.1714 | -0.5744 | 0.3386 | 0.8349 |
| D069 | D130 | 0.146 | -1.346 | 0.2594 | 0.7909 |
| D020 | D289 | 0.137 | -0.475 | 0.3247 | 0.8448 |
| D313 | D430 | 0.1274 | -0.3142 | 0.4762 | 0.8321 |

## Figures

- `/workplace/urban_visual_gene/paper/figures/supplementary/feature_mae_top4_typed_hierarchy/Fig_TopX_Selection.png`
- `/workplace/urban_visual_gene/paper/figures/supplementary/feature_mae_top4_typed_hierarchy/Fig_AND_OR_Relation_Map.png`
- `/workplace/urban_visual_gene/paper/figures/supplementary/feature_mae_top4_typed_hierarchy/Fig_Typed_Semantic_Hierarchy.png`
- `/workplace/urban_visual_gene/paper/figures/supplementary/feature_mae_top4_typed_hierarchy/Fig_Macro_Parent_Semantic_Audit.png`
