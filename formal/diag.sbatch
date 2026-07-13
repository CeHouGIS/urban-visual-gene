#!/bin/bash
#SBATCH --job-name=uvg-diag
#SBATCH --account=fc_globalsvf
#SBATCH --partition=savio3_gpu
#SBATCH --qos=a40_gpu3_normal
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:A40:1
#SBATCH --time=00:20:00
#SBATCH --output=/global/scratch/users/cehou/urban-visual-gene/formal/slurm-diag-%j.out
export LD_LIBRARY_PATH=/global/scratch/users/cehou/conda_envs/svi/lib:${LD_LIBRARY_PATH:-}
export HF_TOKEN="$(cat ~/.cache/huggingface/token)"
export HF_HUB_DISABLE_XET=1 KMP_DUPLICATE_LIB_OK=TRUE
/global/scratch/users/cehou/conda_envs/svi/bin/python -u /global/scratch/users/cehou/urban-visual-gene/formal/diag_norm.py
