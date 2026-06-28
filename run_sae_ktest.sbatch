#!/bin/bash
#SBATCH --job-name=uvg-ktest
#SBATCH --account=fc_globalsvf
#SBATCH --partition=savio3_gpu
#SBATCH --qos=a40_gpu3_normal
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:A40:1
#SBATCH --time=4:00:00
#SBATCH --requeue
#SBATCH --open-mode=append
#SBATCH --output=/global/scratch/users/cehou/urban-visual-gene/outputs/slurm-uvg-%x-%j.out
set -u
REPO=/global/scratch/users/cehou/urban-visual-gene
ENV=/global/scratch/users/cehou/conda_envs/svi
export LD_LIBRARY_PATH=$ENV/lib:${LD_LIBRARY_PATH:-}
export PYTHONPATH=$REPO
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1 KMP_DUPLICATE_LIB_OK=TRUE
cd "$REPO"
echo "=== SAE K-test K=${K:-128} job=${SLURM_JOB_ID:-NA} host=$(hostname) $(date '+%F %T') ==="
nvidia-smi -L || echo "no gpu"
"$ENV/bin/python3" -u -m scripts.pipeline.run_joint_sae_test --K ${K:-128} --epochs 50
echo "=== K-test DONE rc=$? $(date '+%F %T') ==="
