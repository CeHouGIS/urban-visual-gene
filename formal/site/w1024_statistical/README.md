# Language-free W1024 statistical taxonomy

Generated with no human labels and no VLM descriptions:

```bash
cd /global/scratch/users/cehou/urban-visual-gene
/global/scratch/users/cehou/conda_envs/svi/bin/python \
  -m formal.interpretation.build_statistical_taxonomy
/global/scratch/users/cehou/conda_envs/svi/bin/python \
  -m formal.interpretation.compare_hierarchy_cuts
```

Open `formal/site/w1024_statistics.html` through an HTTP server rooted at
`formal/site/`.

The hierarchy uses five rank-fused views: decoder directions, encoder
directions, top-1 position distributions, city distributions, and PPMI
activation-context profiles. Natural-language category names are intentionally
absent. Cluster IDs (`Cxxx`, `Fxxx`) indicate empirical branches only.

Outputs:

- `gene_metrics.csv/json`: statistical role, granularity, stability, nuisance
  association, support, and branch assignment for all 1024 genes.
- `taxonomy.json`: 32 coarse and 64 fine nested statistical modules.
- `gene_relations.csv/json`: equivalence, specialization, sibling, and
  same-patch co-activation evidence.
- `statistical_arrays.npz`: fused similarity, spectral embedding, labels, and
  split-sample stability.
- `summary.json`: method and result summary.
- `hierarchy_comparison.csv/json`: K=8–256 separation, size balance,
  fragmentation, and independently rebuilt half-sample stability comparison.

Reruns reuse the expensive hierarchy/stability arrays. Pass
`--force-hierarchy` to recompute them.
