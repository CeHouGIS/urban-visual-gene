#!/bin/bash
#SBATCH --job-name=uvg-btk-w1024-k8
#SBATCH --account=fc_globalsvf
#SBATCH --partition=savio3_gpu
#SBATCH --qos=a40_gpu3_normal
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:A40:1
#SBATCH --time=12:00:00
#SBATCH --requeue
#SBATCH --output=/global/scratch/users/cehou/urban-visual-gene/formal/slurm-batchtopk-w1024-k8-%j.out
set -eu

REPO=/global/scratch/users/cehou/urban-visual-gene
ENV=/global/scratch/users/cehou/conda_envs/svi
PY=$ENV/bin/python

export LD_LIBRARY_PATH=$ENV/lib:${LD_LIBRARY_PATH:-}
export PYTHONPATH=$REPO OMP_NUM_THREADS=8 KMP_DUPLICATE_LIB_OK=TRUE

cd "$REPO"
# On a directly allocated server, use GPU2 with CUDA_VISIBLE_DEVICES=2.
# Slurm allocations set CUDA_VISIBLE_DEVICES themselves; do not override it here.
nvidia-smi -L
FORMAL_OUT="$REPO/formal/formal_out_panorama_context" \
  "$PY" -u -m formal.gpu_run \
  --cities $($PY - <<'PY'
import json
print(" ".join(json.load(open("configs/formal_city_manifest.json"))))
PY
  ) \
  --dict-panos 12800 \
  --K-list 1024 \
  --topk 8 \
  --epochs 60 \
  --context-weight 0.25 \
  --sample-json "$REPO/configs/formal_city_manifest.json" \
  --skip-infer
echo "=== BATCHTOPK W1024 K8 DONE $(date '+%F %T') ==="
