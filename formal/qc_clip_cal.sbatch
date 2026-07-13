#!/bin/bash
#SBATCH --job-name=uvg-clipcal
#SBATCH --account=fc_globalsvf
#SBATCH --partition=savio4_gpu
#SBATCH --qos=savio_lowprio
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:L40:1
#SBATCH --time=00:25:00
#SBATCH --requeue
#SBATCH --output=/global/scratch/users/cehou/urban-visual-gene/formal/slurm-clipcal-%j.out
set -eu
REPO=/global/scratch/users/cehou/urban-visual-gene; ENV=/global/scratch/users/cehou/conda_envs/svi; PY=$ENV/bin/python
export LD_LIBRARY_PATH=$ENV/lib:${LD_LIBRARY_PATH:-}
export HF_TOKEN="$(cat ~/.cache/huggingface/token)" HF_HUB_DISABLE_XET=1 TRANSFORMERS_OFFLINE=1
export PYTHONPATH=$REPO OMP_NUM_THREADS=8 KMP_DUPLICATE_LIB_OK=TRUE
cd "$REPO"; nvidia-smi -L
"$PY" -u -m formal.clip_tunnel --glob 'formal/formal_out_global2/genes/thumbs/*.jpg' --calibrate
echo "=== CLIPCAL DONE $(date '+%F %T') ==="
