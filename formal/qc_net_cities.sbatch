#!/bin/bash
#SBATCH --job-name=uvg-qcfull
#SBATCH --account=fc_globalsvf
#SBATCH --partition=savio4_gpu
#SBATCH --qos=savio_lowprio
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:L40:1
#SBATCH --time=06:00:00
#SBATCH --requeue
#SBATCH --output=/global/scratch/users/cehou/urban-visual-gene/formal/slurm-qcfull-%j.out
set -u
REPO=/global/scratch/users/cehou/urban-visual-gene; ENV=/global/scratch/users/cehou/conda_envs/svi
export LD_LIBRARY_PATH=$ENV/lib:${LD_LIBRARY_PATH:-} PYTHONPATH=$REPO OMP_NUM_THREADS=8 KMP_DUPLICATE_LIB_OK=TRUE
cd "$REPO"; nvidia-smi -L
"$ENV/bin/python" -u -m formal.qc_net_cities --thr 0.6 --batch 512 --workers 8
echo "=== QCFULL DONE $(date '+%F %T') ==="
