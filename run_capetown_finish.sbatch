#!/bin/bash
#SBATCH --job-name=uvg-ct
#SBATCH --account=fc_globalsvf
#SBATCH --partition=savio3_gpu
#SBATCH --qos=a40_gpu3_normal
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:A40:1
#SBATCH --time=3:00:00
#SBATCH --requeue
#SBATCH --open-mode=append
#SBATCH --output=/global/scratch/users/cehou/urban-visual-gene/outputs/slurm-uvg-%x-%j.out
# Finish CapeTown: only Stage 4-5 (torch) + Stage 6 (geo), reusing the existing
# (row-group-fixed) road_context_features.parquet. stage1-3 already succeeded.
set -u
REPO=/global/scratch/users/cehou/urban-visual-gene
PY=/global/scratch/users/cehou/conda_envs/svi/bin/python3
OUT=outputs/SouthAfrica/CapeTown
export PROJ_DATA=/global/scratch/users/cehou/conda_envs/svi/share/proj
export PYTHONPATH=$REPO
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1 KMP_DUPLICATE_LIB_OK=TRUE
cd "$REPO"

echo "=== CapeTown finish job=${SLURM_JOB_ID:-NA} host=$(hostname) $(date '+%F %T') ==="
nvidia-smi -L || echo "WARN: no GPU"

echo ">>> Stage 4-5 (torch) $(date '+%T')"
"$PY" -u -m scripts.pipeline.run_stage45 --out "$OUT" --K 32 --epochs 50 || { echo "stage45 FAILED"; exit 1; }
echo ">>> Stage 6 (geo) $(date '+%T')"
"$PY" -u -m scripts.pipeline.run_stage6 --out "$OUT" --min-confidence 0.5 || { echo "stage6 FAILED"; exit 1; }
echo "=== CapeTown DONE $(date '+%F %T') ==="
