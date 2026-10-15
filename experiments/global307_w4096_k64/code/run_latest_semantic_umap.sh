#!/usr/bin/env bash
# Bounded Top100 semantic evidence for the latest completed W4096/K64 SAE.
set -euo pipefail
repo=$(git -C "$(dirname "${BASH_SOURCE[0]}")" rev-parse --show-toplevel)
cd "$repo"
export OMP_NUM_THREADS=4
export OPENBLAS_NUM_THREADS=4
model_root=/host/root/mnt/nas/huangyj/GoogleSV/urban_visual_gene
output="$repo/reports/data/semantic_umap_w4096_k64_top100"
stage=${1:-all}
echo "Starting stage $stage at $(date -u)"
python -m formal.analysis.semantic_feature_umap \
  --manifest "$output/images_source.jsonl" \
  --checkpoint "$model_root/formal_batchtopk_307city_w4096_k64/sae_448_k4096.pt" \
  --dataset 'Latest SAE W4096 k64; trained on 304 cities; evidence: 295 cities / 1180 panos / 4720 views' \
  --outdir "$output" \
  --expected-cities 295 --expected-width 4096 --expected-k 64 \
  --top 100 --min-examples 20 --device cuda:1 --pano-batch 4 \
  --crop-workers 8 --embedding-workers 4 --embedding-batch 128 \
  --embedding-precision bf16 \
  --vlm-model /root/.cache/urban_scene_models/siglip-base-patch16-224 \
  --stage "$stage"
echo "Completed stage $stage at $(date -u)"
