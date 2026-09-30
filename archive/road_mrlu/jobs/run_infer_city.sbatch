#!/bin/bash
#SBATCH --job-name=uvg-infer
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
# Re-infer one city with the trained joint SAE basis, on a high-RAM node
# (the GPU node OOM'd on CapeTown's 774k x 4096 inference).
#   CITY=CapeTown sbatch run_infer_city.sbatch
set -u
CITY="${CITY:?set CITY}"
REPO=/global/scratch/users/cehou/urban-visual-gene
ENV=/global/scratch/users/cehou/conda_envs/svi
export LD_LIBRARY_PATH=$ENV/lib:${LD_LIBRARY_PATH:-}
export PYTHONPATH=$REPO
export OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 MKL_NUM_THREADS=4 KMP_DUPLICATE_LIB_OK=TRUE
cd "$REPO"
echo "=== UVG INFER city=$CITY job=${SLURM_JOB_ID:-NA} host=$(hostname) $(date '+%F %T') ==="
"$ENV/bin/python3" -u -m scripts.pipeline.infer_joint_city --city "$CITY"
echo "=== INFER DONE city=$CITY rc=$? $(date '+%F %T') ==="
