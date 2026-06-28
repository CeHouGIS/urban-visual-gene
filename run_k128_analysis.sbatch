#!/bin/bash
#SBATCH --job-name=uvg-k128an
#SBATCH --account=fc_globalsvf
#SBATCH --partition=savio3_bigmem
#SBATCH --qos=savio_normal
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --time=4:00:00
#SBATCH --requeue
#SBATCH --open-mode=append
#SBATCH --output=/global/scratch/users/cehou/urban-visual-gene/outputs/slurm-uvg-%x-%j.out
set -u
REPO=/global/scratch/users/cehou/urban-visual-gene
ENV=/global/scratch/users/cehou/conda_envs/svi
PY=$ENV/bin/python3
export LD_LIBRARY_PATH=$ENV/lib:${LD_LIBRARY_PATH:-}
export PYTHONPATH=$REPO
export OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 MKL_NUM_THREADS=4 KMP_DUPLICATE_LIB_OK=TRUE
cd "$REPO"
echo "=== K=128 analysis job=${SLURM_JOB_ID:-NA} host=$(hostname) $(date '+%F %T') ==="
for CITY in HongKong Singapore Amsterdam CapeTown; do
  echo ">>> infer K=128 $CITY $(date '+%T')"
  "$PY" -u -m scripts.pipeline.infer_joint_city --city "$CITY" \
      --joint-dir outputs/transfer/joint_k128 \
      --act-out road_basis_activation_k128.parquet \
      --basis-out road_landscape_basis_k128.npy || { echo "infer $CITY FAILED"; exit 1; }
done
echo ">>> build_core_analysis (K=128) $(date '+%T')"
"$PY" -u -m scripts.dash.build_core_analysis --K 128 \
    --act-file road_basis_activation_k128.parquet \
    --basis-file road_landscape_basis_k128.npy \
    --out analysis_k128.json || { echo "analysis FAILED"; exit 1; }
echo "=== K=128 analysis DONE $(date '+%F %T') ==="
