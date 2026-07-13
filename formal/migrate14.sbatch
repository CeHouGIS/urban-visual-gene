#!/bin/bash
#SBATCH --job-name=uvg-mig14
#SBATCH --account=fc_globalsvf
#SBATCH --partition=savio4_gpu
#SBATCH --qos=savio_lowprio
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:L40:1
#SBATCH --time=01:00:00
#SBATCH --requeue
#SBATCH --output=/global/scratch/users/cehou/urban-visual-gene/formal/slurm-mig14-%j.out
set -eu
REPO=/global/scratch/users/cehou/urban-visual-gene; ENV=/global/scratch/users/cehou/conda_envs/svi; PY=$ENV/bin/python
export LD_LIBRARY_PATH=$ENV/lib:${LD_LIBRARY_PATH:-}
export HF_TOKEN="$(cat ~/.cache/huggingface/token)" HF_HUB_DISABLE_XET=1
export PYTHONPATH=$REPO OMP_NUM_THREADS=8 KMP_DUPLICATE_LIB_OK=TRUE
export DICTDIR=$REPO/formal/formal_out_global2
export GENE_PROJ=$REPO/formal/formal_out_global/artifact_dirs_pos.npy
export CATJSON=$REPO/formal/categories_new.json
cd "$REPO"; nvidia-smi -L
rm -f $DICTDIR/web2/panos.parquet   # force recompute composition with 14-branch g2c
echo "=== 1/4 compose_cities 14-cat (GPU) ==="; "$PY" -u -m formal.compose_cities
echo "=== 2/4 compose_analyze (CPU) ==="; "$PY" -u -m formal.compose_analyze
echo "=== 3/4 build_explorer (CPU) ==="; EXPDIR=formal_out_expglobal2 DICTDIR_NAME=formal_out_global2 "$PY" -u formal/build_explorer.py
echo "=== 4/4 interpret montage (CPU) ==="; "$PY" -u formal/interp_montage.py
echo "=== MIG14 DONE $(date '+%F %T') ==="
