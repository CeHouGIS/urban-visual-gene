#!/bin/bash
#SBATCH --job-name=uvg-enc-btk1024
#SBATCH --account=fc_globalsvf
#SBATCH --partition=savio3_gpu
#SBATCH --qos=a40_gpu3_normal
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:A40:1
#SBATCH --time=06:00:00
#SBATCH --requeue
#SBATCH --output=/global/scratch/users/cehou/urban-visual-gene/formal/slurm-encode-batchtopk-w1024-k8-%j.out
set -eu

REPO=/global/scratch/users/cehou/urban-visual-gene
ENV=/global/scratch/users/cehou/conda_envs/svi
PY=$ENV/bin/python
FORMAL_OUT=${FORMAL_OUT:-/host/root/mnt/nas/huangyj/GoogleSV/urban_visual_gene/formal_batchtopk_37city_fullpatch}

export LD_LIBRARY_PATH=$ENV/lib:${LD_LIBRARY_PATH:-}
export HF_TOKEN="$(cat ~/.cache/huggingface/token)" HF_HUB_DISABLE_XET=1
export PYTHONPATH=$REPO OMP_NUM_THREADS=8 KMP_DUPLICATE_LIB_OK=TRUE

cd "$REPO"
# On a directly allocated server, use GPU2 with CUDA_VISIBLE_DEVICES=2.
# Slurm allocations set CUDA_VISIBLE_DEVICES themselves; do not override it here.
nvidia-smi -L
"$PY" -u -m sae_experiments.evaluation.encode_batchtopk_w1024_k8 \
  --checkpoint "$FORMAL_OUT/sae_448_k1024.pt" \
  --outdir "$FORMAL_OUT/encoded" \
  --ref-sparse "$FORMAL_OUT/genes/sparse_acts.npz" \
  --res 448 \
  --batch 32 \
  --threshold 0.0005 \
  --context-weight 0.25 \
  --force
echo "=== ENCODE BATCHTOPK W1024 K8 DONE $(date '+%F %T') ==="
