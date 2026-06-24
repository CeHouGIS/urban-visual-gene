#!/bin/bash
#SBATCH --job-name=uvg-feat
#SBATCH --account=fc_globalsvf
#SBATCH --partition=savio3_gpu
#SBATCH --qos=a40_gpu3_normal
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:A40:1
#SBATCH --requeue
#SBATCH --open-mode=append
#SBATCH --output=/global/scratch/users/cehou/urban-visual-gene/outputs/slurm-uvg-%x-%j.out
# ============================================================================
# FEATURE EXTRACTION ONLY (DINOv3 ViT-L/16, full download=1 library).
# Output (for transfer to a local server, NO stages 2-6 here):
#   outputs/full_feats/<city>_feats.f16.npy   (N, 4096) float16
#   outputs/full_feats/<city>_meta.parquet    (pano_id, lat, lon)
# Resumable (checkpoints every 20 batches) + DataLoader timeout guard.
#   CITY=Singapore sbatch --time=5:00:00 run_extract_only.sbatch
# ============================================================================
set -u
CITY="${CITY:?set CITY, e.g. CITY=Singapore sbatch run_extract_only.sbatch}"
REPO=/global/scratch/users/cehou/urban-visual-gene
ENV=/global/scratch/users/cehou/conda_envs/svi
PY=$ENV/bin/python3
export LD_LIBRARY_PATH=$ENV/lib:${LD_LIBRARY_PATH:-}
export HF_TOKEN="$(cat ~/.cache/huggingface/token 2>/dev/null)"
export HF_HUB_DISABLE_XET=1
export PYTHONPATH=$REPO
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1 KMP_DUPLICATE_LIB_OK=TRUE
cd "$REPO"

echo "=== UVG FEATURE EXTRACTION city=$CITY job=${SLURM_JOB_ID:-NA} host=$(hostname) $(date '+%F %T') ==="
nvidia-smi -L || echo "WARN: no GPU"
"$PY" -u -m scripts.dedup.extract_full_pano_features "$CITY" --model dinov3_vitl16 --batch-panos 96 --workers 8
rc=$?
echo "=== EXTRACTION city=$CITY rc=$rc $(date '+%F %T') ==="
exit $rc
