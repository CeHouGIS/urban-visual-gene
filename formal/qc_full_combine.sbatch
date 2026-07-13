#!/bin/bash
#SBATCH --job-name=uvg-qccomb
#SBATCH --account=fc_globalsvf
#SBATCH --partition=savio4_htc
#SBATCH --qos=savio_normal
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --time=00:30:00
#SBATCH --output=/global/scratch/users/cehou/urban-visual-gene/formal/slurm-qccomb-%j.out
set -u
REPO=/global/scratch/users/cehou/urban-visual-gene; ENV=/global/scratch/users/cehou/conda_envs/svi
export LD_LIBRARY_PATH=$ENV/lib:${LD_LIBRARY_PATH:-} PYTHONPATH=$REPO OMP_NUM_THREADS=4
cd "$REPO"
"$ENV/bin/python" -u -m formal.qc_net_cities --combine --thr 0.6
echo "=== QCCOMB DONE $(date '+%F %T') ==="
