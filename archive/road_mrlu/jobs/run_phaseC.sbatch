#!/bin/bash
#SBATCH --job-name=uvg-phaseC
#SBATCH --account=fc_globalsvf
#SBATCH --partition=savio3_bigmem
#SBATCH --qos=savio_normal
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --time=8:00:00
#SBATCH --requeue
#SBATCH --open-mode=append
#SBATCH --output=/global/scratch/users/cehou/urban-visual-gene/outputs/slurm-uvg-%x-%j.out
# ============================================================================
# PHASE C (CPU): Stage 6 units (4 cities, joint activations) + rebuild dashboard
# (map + joint UMAP feature space) + analysis. NO publish here (done by hand).
#   sbatch run_phaseC.sbatch
# ============================================================================
set -u
REPO=/global/scratch/users/cehou/urban-visual-gene
ENV=/global/scratch/users/cehou/conda_envs/svi
PY=$ENV/bin/python3
declare -A OUT=( [HongKong]=outputs/China/HongKong [Singapore]=outputs/Singapore/Singapore \
                 [Amsterdam]=outputs/Netherlands/Amsterdam [CapeTown]=outputs/SouthAfrica/CapeTown )
export LD_LIBRARY_PATH=$ENV/lib:${LD_LIBRARY_PATH:-}
export PROJ_DATA=$ENV/share/proj
export PYTHONPATH=$REPO
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1 KMP_DUPLICATE_LIB_OK=TRUE NUMBA_NUM_THREADS=8
cd "$REPO"

echo "=== UVG PHASE C job=${SLURM_JOB_ID:-NA} host=$(hostname) $(date '+%F %T') ==="
for CITY in HongKong Singapore Amsterdam CapeTown; do
  echo ">>> Stage 6 $CITY $(date '+%T')"
  "$PY" -u -m scripts.pipeline.run_stage6 --out "${OUT[$CITY]}" --min-confidence 0.5 || { echo "stage6 $CITY FAILED"; exit 1; }
done
echo ">>> build_core_dashboard (map + joint UMAP) $(date '+%T')"
"$PY" -u -m scripts.dash.build_core_dashboard || { echo "dashboard FAILED"; exit 1; }
echo ">>> build_core_analysis $(date '+%T')"
"$PY" -u -m scripts.dash.build_core_analysis || { echo "analysis FAILED"; exit 1; }
echo "=== PHASE C DONE $(date '+%F %T') ==="
