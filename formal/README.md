# Urban Visual Gene — pipeline (`formal/`)

DINOv3 patch features → Top-K Sparse Autoencoder → interpretable "visual genes"
of street scenes, across 12 cities. Live site: https://cehougis.github.io/urban-visual-gene

## Layout

```
formal/
├── gpu_run.py            # CORE: DINOv3 Extractor (+positional-subspace projection),
│                         #   Top-K SAE, de-biased road-weighted sampling (stratified_panos,
│                         #   auto-excludes QC-flagged panos via bad_panos), meta/points DBs.
├── dict/                 # dictionary training
│   ├── pos_subspace.py   #   estimate positional-artifact subspace (top-r PCs of per-position mean)
│   └── retrain_debias.py #   train the global K=512 SAE with that subspace projected out
├── genes/                # 512-gene dashboard + category taxonomy
│   ├── gene_encode.py    #   encode a cross-city pool -> sparse top-k acts + thumbs (GPU)
│   ├── gene_render.py    #   512 x top-20 jet activation overlays + manifest
│   ├── gene_orig.py      #   paired original crops (for the 原图⇄激活 toggle)
│   ├── gene_tree.py      #   balanced ward hierarchy (super/sub branches)
│   ├── gene_posdiag.py   #   positional-gene diagnostic (posR2)
│   ├── gene_coexpr.py    #   co-expression (lift) + Louvain modules
│   ├── gene_modules_viz.py, coexpr_web.py   # module exemplars + coexpr web data
│   ├── build_taxonomy.py, branch_cats.py    # 6-parent/16-child semantic taxonomy
│   └── cat_distinguish.py, explore_taxonomy.py   # category-redundancy diagnostics
├── interpretation/       # semantic audit + typed ontology
│   ├── build_w1024_audit.py  # all-gene diagnostics + stratified W1024 pilot
│   ├── build_statistical_taxonomy.py  # language-free multiview hierarchy + relations
│   └── ontology_v0_1.yaml    # semantic/status/relation vocabulary
├── web/                  # site data builders
│   ├── compose_cities.py / compose_analyze.py  # per-pano composition + scene types
│   ├── build_explorer.py # per-street category overlays (4 road cities)
│   ├── render_global.py / interp_montage.py / dendro512.py  # interpret-page assets
├── quality/              # image quality filter (drop black/blur/dark/tunnel)
│   ├── quality_filter.py # fast heuristics (brightness + tiled Laplacian)
│   ├── clip_tunnel.py    # CLIP zero-shot tunnel scoring
│   ├── qc_net.py         # small 4-class CNN {good,dark,blur,tunnel}, synthetic + labeled
│   ├── qc_net_cities.py  # full-sample CNN inference -> qc_full/<city>.parquet (shardable)
│   ├── qc_cities.py      # heuristic per-city blocklist
│   ├── qc_label_tool.py  # generate the online tunnel-labeling page
│   └── render_dropped.py # montage of dropped images
├── slurm/                # SLURM batch scripts (run via `-m formal.<pkg>.<mod>`)
├── site/                 # web pages (deployed to gh-pages)
├── figures/              # analysis figures (committed)
└── (gitignored data) formal_out_global2/ (de-biased K=512 dict + genes/web assets),
                     formal_out_expglobal2/, qc/ qc_full/ (QC blocklists), ondisk/
```

## Key artifacts
- `formal_out_global2/sae_448_k512.pt` — de-biased global K=512 dictionary (recon cos 0.892).
- `formal_out_global2/artifact_dirs_pos.npy` — 11-dim positional subspace projected out.
- `formal_out_global2/gene2cat.npy` + `taxonomy.json` — gene→parent (6-class) taxonomy.
- `qc_full/<city>.parquet` — full-sample QC blocklist (auto-used by `stratified_panos`).

## Run (examples; PYTHONPATH=repo root)
```bash
# dictionary (GPU): pos subspace then retrain
python -m formal.dict.pos_subspace ; python -m formal.dict.retrain_debias
# gene dashboard (GPU encode + CPU render): sbatch slurm/gene_panel2.sbatch
# quality filter full-sample (CPU array): sbatch slurm/qc_full_array.sbatch  (then qc_full_combine)
# BatchTopK W1024/K8 audit (CPU; then open site/w1024_audit.html via HTTP)
python -m formal.interpretation.build_w1024_audit
# Fully statistical W1024 taxonomy (no annotations or language model)
python -m formal.interpretation.build_statistical_taxonomy
# Compare hierarchy cuts with separation, balance, and half-sample stability
python -m formal.interpretation.compare_hierarchy_cuts
# Add exemplars for statistically active genes omitted by the old city threshold
python -m formal.web.build_ablation_genes batchtopk_w1024_k8 --backfill-missing
```
Env: `/global/scratch/users/cehou/conda_envs/svi/bin/python`. Deploy: gh-pages worktree + SSH push.
