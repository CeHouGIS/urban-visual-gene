#!/usr/bin/env bash
# Generate Top100 semantic evidence for the matched W2048/K64 SAE.
set -euo pipefail

repo=$(git -C "$(dirname "${BASH_SOURCE[0]}")" rev-parse --show-toplevel)
cd "$repo"
export OMP_NUM_THREADS=4
export OPENBLAS_NUM_THREADS=4

model_root=/host/root/mnt/nas/huangyj/GoogleSV/urban_visual_gene
checkpoint="$model_root/formal_batchtopk_307city_w2048_k64/sae_448_k2048.pt"
output="$repo/reports/data/semantic_umap_w2048_k64_top100"
manifest="$repo/reports/data/semantic_umap_w4096_k64_top100/images_source.jsonl"

if [[ ! -f "$checkpoint" ]]; then
  echo "Missing trained checkpoint: $checkpoint" >&2
  exit 1
fi

python -m formal.analysis.semantic_feature_umap \
  --manifest "$manifest" \
  --checkpoint "$checkpoint" \
  --dataset 'BatchTopK SAE W2048 k64; trained on 304 cities and 30M patches; evidence: same 295 cities / 1180 panos / 4720 views as W4096' \
  --outdir "$output" \
  --expected-cities 295 --expected-width 2048 --expected-k 64 \
  --top 100 --min-examples 20 --device cuda:1 --pano-batch 4 \
  --crop-workers 8 --embedding-workers 4 --embedding-batch 128 \
  --embedding-precision bf16 \
  --vlm-model /root/.cache/urban_scene_models/siglip-base-patch16-224 \
  --stage "${1:-all}"
