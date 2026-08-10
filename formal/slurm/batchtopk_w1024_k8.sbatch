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
nvidia-smi -L
"$PY" -u -m sae_experiments.training.train_sae_from_features \
  --sample-path "$REPO/formal/formal_out_global3/dict_sample.f16.npy" \
  --outdir "$REPO/formal/batchtopk_w1024_k8" \
  --model-type batch_topk \
  --width 1024 \
  --k 8 \
  --epochs 60 \
  --batch-size 16384 \
  --lr 0.0003 \
  --weight-decay 0.00001 \
  --grad-clip 1.0 \
  --load-ram \
  --log-every 100
echo "=== BATCHTOPK W1024 K8 DONE $(date '+%F %T') ==="
