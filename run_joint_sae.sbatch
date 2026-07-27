#!/bin/bash
#SBATCH --job-name=uvg-jsae
#SBATCH --account=fc_globalsvf
#SBATCH --partition=savio3_gpu
#SBATCH --qos=a40_gpu3_normal
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:A40:1
#SBATCH --time=6:00:00
#SBATCH --requeue
#SBATCH --open-mode=append
#SBATCH --output=/global/scratch/users/cehou/urban-visual-gene/outputs/slurm-uvg-%x-%j.out
# ============================================================================
# JOINT SAE (Phase B, GPU): train ONE basis on pooled covered nodes from all 4
# cities (DINOv3 4096-d Z_road), then infer per-city activations with it.
# Needs all 4 cities' road_context_features.parquet ready (HK + Phase-A geo).
#   sbatch run_joint_sae.sbatch
# ============================================================================
set -u
REPO=/global/scratch/users/cehou/urban-visual-gene
ENV=/global/scratch/users/cehou/conda_envs/svi
PY=$ENV/bin/python3
export LD_LIBRARY_PATH=$ENV/lib:${LD_LIBRARY_PATH:-}
export PYTHONPATH=$REPO
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1 KMP_DUPLICATE_LIB_OK=TRUE
cd "$REPO"

echo "=== UVG JOINT SAE job=${SLURM_JOB_ID:-NA} host=$(hostname) $(date '+%F %T') ==="
nvidia-smi -L || echo "WARN: no GPU"
"$PY" -u -m scripts.pipeline.run_stage45_joint --K 32 --epochs 50 --per-city-cap 120000
echo "=== JOINT SAE DONE rc=$? $(date '+%F %T') ==="
