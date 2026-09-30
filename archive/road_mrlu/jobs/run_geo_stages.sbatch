#!/bin/bash
#SBATCH --job-name=uvg-geo
#SBATCH --account=fc_globalsvf
#SBATCH --partition=savio3_bigmem
#SBATCH --qos=savio_normal
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --time=10:00:00
#SBATCH --requeue
#SBATCH --open-mode=append
#SBATCH --output=/global/scratch/users/cehou/urban-visual-gene/outputs/slurm-uvg-%x-%j.out
# ============================================================================
# GEO STAGES (CPU, no GPU) on DINOv3 FULL features for one city:
#   bridge full_feats -> pano_features.parquet, Stage 2 (map-match + graph),
#   Stage 3 (Z_road context, 4096-d). bigmem: CapeTown matches the full pano
#   library (~628k x 4096) + builds ~774k-node context -> ~40-50GB peak.
#   CITY=Singapore sbatch run_geo_stages.sbatch
# ============================================================================
set -u
CITY="${CITY:?set CITY, e.g. CITY=Singapore sbatch run_geo_stages.sbatch}"
REPO=/global/scratch/users/cehou/urban-visual-gene
ENV=/global/scratch/users/cehou/conda_envs/svi
PY=$ENV/bin/python3
declare -A OUT=( [HongKong]=outputs/China/HongKong [Singapore]=outputs/Singapore/Singapore \
                 [Amsterdam]=outputs/Netherlands/Amsterdam [CapeTown]=outputs/SouthAfrica/CapeTown )
O="${OUT[$CITY]}"
export LD_LIBRARY_PATH=$ENV/lib:${LD_LIBRARY_PATH:-}
export PROJ_DATA=$ENV/share/proj
export PYTHONPATH=$REPO
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1 KMP_DUPLICATE_LIB_OK=TRUE
cd "$REPO"

echo "=== UVG GEO city=$CITY job=${SLURM_JOB_ID:-NA} host=$(hostname) $(date '+%F %T') ==="
echo ">>> bridge full_feats -> pano_features.parquet $(date '+%T')"
"$PY" -u -m scripts.dedup.build_full_pano_features "$CITY" --model dinov3_vitl16 || { echo "bridge FAILED"; exit 1; }
echo ">>> Stage 2 map-match $(date '+%T')"
"$PY" -u -m scripts.pipeline.run_stage2 --city "$CITY" || { echo "stage2 FAILED"; exit 1; }
echo ">>> Stage 3 Z_road $(date '+%T')"
"$PY" -u -m scripts.pipeline.run_stage3 --out "$O" || { echo "stage3 FAILED"; exit 1; }
echo "=== UVG GEO DONE city=$CITY $(date '+%F %T') ==="
