# `formal/`: BatchTopK urban visual genes

This directory contains the paper-facing analysis built on frozen DINOv3 patch
features and the final **BatchTopK sparse autoencoder**. The formal checkpoint
is written to `formal/formal_out_panorama_context/sae_448_k1024.pt`.

## Formal sampling design

The formal study uses 37 representative cities spanning the major world
regions. `configs/formal_city_manifest.json` assigns 12,800 street-view
panoramas to every city. Each selected panorama must have four directional
images (0°, 90°, 180°, and 270°), for 51,200 directional images per city
before quality control. The manifest was checked against the NAS catalogue;
the final paper should still report the retained count after quality control.

An older 12-city, 14,400-directional-image pilot remains in
`batchtopk_w1024_k8/summary.json` for audit history only. Its metrics are not
formal results and must not be carried into the 37-city analysis.

## Main model

- Backbone: `facebook/dinov3-vitl16-pretrain-lvd1689m`, frozen.
- Input: 448 x 448 pixels, yielding a 28 x 28 spatial patch grid.
- Patch feature width: 1024, after removing prefix/register tokens and L2
  normalizing each patch.
- SAE: BatchTopK, dictionary width W=1024, average batch sparsity K=8,
  unit-normalized decoder columns.
- Training: cosine reconstruction loss, Adam, learning rate `1e-3`, no weight
  decay, batch size `16384`, 60 epochs.

## Important files

| Path | Purpose |
| --- | --- |
| `gpu_run.py` | Resumable DINOv3 extraction, BatchTopK training/inference, and image metadata handling. |
| `interpretation/` | W1024 audits, hierarchy comparisons, and statistical taxonomy. |
| `web/` | City composition, gene sharing, and site-data builders. |
| `quality/` | Image quality and tunnel/blur screening utilities. |
| `slurm/batchtopk_w1024_k8.sbatch` | Cluster training job. |
| `slurm/encode_batchtopk_w1024_k8.sbatch` | Cluster encoding and prevalence summary job. |
| `batchtopk_w1024_k8/summary.json` | Pilot-run metrics, not formal 30--40-city results. |

The older `genes/` and `dict/` scripts and their W512 outputs are retained as
historical analyses. They are not the final paper method.

## Pre-DINO quality gate

`gpu_run.py` applies `formal/quality/tiny_qc.py` before DINOv3. It reads each
complete four-view panorama, computes a 96-pixel descriptor, and rejects a
panorama if any view is near-black or at least two views are tunnel-like. A
small `HistGradientBoostingClassifier` checkpoint can be trained from weak
labels; without a checkpoint the same conservative black/tunnel rules run as a
deterministic fallback. The gate never changes DINOv3 or SAE features.

Train a local checkpoint on a representative image sample:

```bash
python -m formal.quality.tiny_qc \
  --glob '/host/root/mnt/nas/huangyj/GoogleSV/images/**/*_0.jpg' \
  --limit 50000 --output formal/quality/qc_model.joblib
```

Pass a different checkpoint with `--qc-model`; use `--no-qc` only for an
explicit ablation. S0 logs `candidates`, `black`, `tunnel`, and `kept` counts
for every city so the retained sample is auditable.

## Data and GPU2

Street-view images and metadata are on NAS:

```text
/host/root/mnt/nas/huangyj/GoogleSV/images
/host/root/mnt/nas/huangyj/GoogleSV/metadata
```

`gpu_run.py` uses this root by default and accepts `GOOGLE_SV_ROOT` to override
it. The formal city aliases and catalogue paths are defined in
`formal/gpu_run.py` and are kept in sync with the manifest.

For direct execution on the current server, use both available GPUs for the
DINOv3 extraction stage:

```bash
export CUDA_VISIBLE_DEVICES=1,2
export PYTHONPATH="$PWD"
```

For a pilot or a manifest-driven shard, run:

```bash
python -m formal.gpu_run --dict-panos 12800 --K-list 1024 --topk 8 \
  --epochs 60 --batch 32 --sample-json configs/formal_city_manifest.json
```

The `HEADINGS` list in `gpu_run.py` is fixed to all four directions. The Slurm
scripts leave GPU visibility to the scheduler; set `CUDA_VISIBLE_DEVICES=1,2`
only for a direct, non-Slurm launch on this server.

## Relationship to archived code

`archive/road_mrlu/` contains the removed road MRLU pipeline. It is not needed
for BatchTopK training, encoding, gene analysis, or the chapter's reported
results.
