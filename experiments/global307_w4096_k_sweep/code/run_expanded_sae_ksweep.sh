#!/usr/bin/env bash
# Train the W4096 sparsity sweep and build four-view activation galleries.
set -euo pipefail

repo=$(git -C "$(dirname "${BASH_SOURCE[0]}")" rev-parse --show-toplevel)
model_root=/host/root/mnt/nas/huangyj/GoogleSV/urban_visual_gene
gallery_root="$repo/reports/sae_4096"
log_root="$repo/formal/logs"

retry() {
  local attempt=1
  while ! "$@"; do
    if (( attempt >= 5 )); then
      echo "Failed after ${attempt} attempts: $*" >&2
      return 1
    fi
    echo "Attempt ${attempt} failed at $(date -u); resuming in 60 seconds" >&2
    sleep 60
    ((attempt+=1))
  done
}

if [[ ${1:-} == --worker ]]; then
  k=$2
  gpu=$3
  cd "$repo"
  export CUDA_VISIBLE_DEVICES="$gpu"
  export OMP_NUM_THREADS=8
  export OPENBLAS_NUM_THREADS=8
  echo "Starting W4096/K${k} on physical GPU ${gpu} at $(date -u)"
  retry python -m formal.analysis.train_expanded_sae \
    --width 4096 --k "$k" --outdir "$model_root/formal_batchtopk_307city_w4096_k${k}"
  retry python -m formal.analysis.build_expanded_sae_gallery \
    --checkpoint "$model_root/formal_batchtopk_307city_w4096_k${k}/sae_448_k4096.pt" \
    --outdir "$gallery_root/k${k}" --device cuda:0
  echo "Completed W4096/K${k} at $(date -u)"
  exit 0
fi

mkdir -p "$log_root"
for spec in 16:0 32:2 64:3; do
  k=${spec%%:*}
  gpu=${spec##*:}
  session="uvg_w4096_k${k}"
  if [[ -f "$gallery_root/k${k}/gallery.json" ]]; then
    echo "K${k}: gallery already exists"
    continue
  fi
  if tmux has-session -t "$session" 2>/dev/null; then
    echo "K${k}: already running (tmux ${session})"
    continue
  fi
  tmux new-session -d -s "$session" -c "$repo" \
    "bash formal/analysis/run_expanded_sae_ksweep.sh --worker $k $gpu > formal/logs/expanded_sae_w4096_k${k}.log 2>&1"
  echo "K${k}: started on GPU ${gpu} (tmux ${session})"
done
