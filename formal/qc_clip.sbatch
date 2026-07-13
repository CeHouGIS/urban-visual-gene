#!/bin/bash
#SBATCH --job-name=uvg-qcclip
#SBATCH --account=fc_globalsvf
#SBATCH --partition=savio4_gpu
#SBATCH --qos=savio_lowprio
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:L40:1
#SBATCH --time=00:40:00
#SBATCH --requeue
#SBATCH --output=/global/scratch/users/cehou/urban-visual-gene/formal/slurm-qcclip-%j.out
set -eu
REPO=/global/scratch/users/cehou/urban-visual-gene; ENV=/global/scratch/users/cehou/conda_envs/svi; PY=$ENV/bin/python
export LD_LIBRARY_PATH=$ENV/lib:${LD_LIBRARY_PATH:-}
export HF_TOKEN="$(cat ~/.cache/huggingface/token)" HF_HUB_DISABLE_XET=1
export PYTHONPATH=$REPO OMP_NUM_THREADS=8 KMP_DUPLICATE_LIB_OK=TRUE
cd "$REPO"; nvidia-smi -L
echo "=== calibrate on 6000 thumbs ==="
"$PY" -u -m formal.clip_tunnel --glob 'formal/formal_out_global2/genes/thumbs/*.jpg' --calibrate
echo "=== update per-city QC parquets (CLIP tunnel, thresh 0.55) ==="
"$PY" -u -m formal.clip_tunnel --update-qc --thresh 0.55
echo "=== QCCLIP DONE $(date '+%F %T') ==="
