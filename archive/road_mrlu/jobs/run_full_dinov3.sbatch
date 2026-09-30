#!/bin/bash
#SBATCH --job-name=uvg-full
#SBATCH --account=fc_globalsvf
#SBATCH --partition=savio3_gpu
#SBATCH --qos=a40_gpu3_normal
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:A40:1
#SBATCH --time=24:00:00
#SBATCH --requeue
#SBATCH --open-mode=append
#SBATCH --output=/global/scratch/users/cehou/urban-visual-gene/outputs/slurm-uvg-%x-%j.out
# ============================================================================
# FULL-library DINOv3 MRLU pipeline for one city. No sampling: every download=1
# pano gets features; stages 2-6 run on the full set.
#   CITY=HongKong sbatch run_full_dinov3.sbatch
# Resumable: extract_full checkpoints (resume on requeue); each city skipped if
# its final unit geojson already exists.
# ============================================================================
set -u
CITY="${CITY:?set CITY, e.g. CITY=HongKong sbatch run_full_dinov3.sbatch}"
REPO=/global/scratch/users/cehou/urban-visual-gene
ENV=/global/scratch/users/cehou/conda_envs/svi
PY=$ENV/bin/python3
declare -A OUT=( [HongKong]=outputs/China/HongKong [Singapore]=outputs/Singapore/Singapore \
                 [Amsterdam]=outputs/Netherlands/Amsterdam [CapeTown]=outputs/SouthAfrica/CapeTown )
O="${OUT[$CITY]}"

# env: conda libstdc++ (transformers->scipy), gated HF token, proj, threads
export LD_LIBRARY_PATH=$ENV/lib:${LD_LIBRARY_PATH:-}
export HF_TOKEN="$(cat ~/.cache/huggingface/token 2>/dev/null)"
export HF_HUB_DISABLE_XET=1
export PROJ_DATA=$ENV/share/proj
export PYTHONPATH=$REPO
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1 KMP_DUPLICATE_LIB_OK=TRUE
cd "$REPO"

echo "=== UVG FULL DINOv3 city=$CITY job=${SLURM_JOB_ID:-NA} host=$(hostname) $(date '+%F %T') ==="
nvidia-smi -L || echo "WARN: no GPU"

echo ">>> [1] full DINOv3 feature extraction (resumable) $(date '+%T')"
"$PY" -u -m scripts.dedup.extract_full_pano_features "$CITY" --model dinov3_vitl16 --batch-panos 96 --workers 8 || { echo "extract FAILED"; exit 1; }

echo ">>> [2] bridge full_feats -> pano_features.parquet $(date '+%T')"
"$PY" -u -m scripts.dedup.build_full_pano_features "$CITY" --model dinov3_vitl16 || { echo "bridge FAILED"; exit 1; }

echo ">>> [3] Stage 2 map-match (geo) $(date '+%T')"
"$PY" -u -m scripts.pipeline.run_stage2 --city "$CITY" || { echo "stage2 FAILED"; exit 1; }
echo ">>> [4] Stage 3 Z_road (geo) $(date '+%T')"
"$PY" -u -m scripts.pipeline.run_stage3 --out "$O" || { echo "stage3 FAILED"; exit 1; }
echo ">>> [5] Stage 4-5 basis+activation (torch) $(date '+%T')"
"$PY" -u -m scripts.pipeline.run_stage45 --out "$O" --K 32 --epochs 50 || { echo "stage45 FAILED"; exit 1; }
echo ">>> [6] Stage 6 units (geo) $(date '+%T')"
"$PY" -u -m scripts.pipeline.run_stage6 --out "$O" --min-confidence 0.5 || { echo "stage6 FAILED"; exit 1; }
echo "=== UVG FULL DINOv3 DONE city=$CITY $(date '+%F %T') ==="
