#!/bin/bash
#SBATCH --job-name=uvg-qcarr
#SBATCH --account=fc_globalsvf
#SBATCH --partition=savio4_htc
#SBATCH --qos=savio_normal
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --array=0-39%20
#SBATCH --time=05:00:00
#SBATCH --requeue
#SBATCH --output=/global/scratch/users/cehou/urban-visual-gene/formal/slurm-qcarr-%A_%a.out
set -u
REPO=/global/scratch/users/cehou/urban-visual-gene; ENV=/global/scratch/users/cehou/conda_envs/svi
export LD_LIBRARY_PATH=$ENV/lib:${LD_LIBRARY_PATH:-} PYTHONPATH=$REPO OMP_NUM_THREADS=8 KMP_DUPLICATE_LIB_OK=TRUE
cd "$REPO"
"$ENV/bin/python" -u -m formal.qc_net_cities --shard $SLURM_ARRAY_TASK_ID --nshards 40 --chunk 100000 --workers 4
