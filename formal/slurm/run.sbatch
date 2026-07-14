#!/bin/bash
#SBATCH --job-name=uvg-formal
#SBATCH --account=fc_globalsvf
#SBATCH --partition=savio3_gpu
#SBATCH --qos=a40_gpu3_normal
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:A40:1
#SBATCH --time=08:00:00
#SBATCH --requeue
#SBATCH --open-mode=append
#SBATCH --output=/global/scratch/users/cehou/urban-visual-gene/formal/slurm-formal-%j.out
set -u
REPO=/global/scratch/users/cehou/urban-visual-gene
ENV=/global/scratch/users/cehou/conda_envs/svi
PY=$ENV/bin/python3
export LD_LIBRARY_PATH=$ENV/lib:${LD_LIBRARY_PATH:-}
export HF_TOKEN="$(cat ~/.cache/huggingface/token 2>/dev/null)"
export HF_HUB_DISABLE_XET=1
export PYTHONPATH=$REPO
export OMP_NUM_THREADS=8 OPENBLAS_NUM_THREADS=8 MKL_NUM_THREADS=8 KMP_DUPLICATE_LIB_OK=TRUE
cd "$REPO"
echo "=== UVG FORMAL job=${SLURM_JOB_ID:-NA} host=$(hostname) $(date '+%F %T') ==="
nvidia-smi -L || echo "WARN: no GPU"
GPU_ARGS="${GPU_ARGS:-}"
echo ">>> gpu_run args: $GPU_ARGS"
"$PY" -u -m formal.gpu_run $GPU_ARGS
echo "=== DONE $(date '+%F %T') ==="
