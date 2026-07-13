#!/bin/bash
#SBATCH --job-name=uvg-mig6
#SBATCH --account=fc_globalsvf
#SBATCH --partition=savio4_gpu
#SBATCH --qos=savio_lowprio
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:L40:1
#SBATCH --time=01:10:00
#SBATCH --requeue
#SBATCH --output=/global/scratch/users/cehou/urban-visual-gene/formal/slurm-mig6-%j.out
set -eu
REPO=/global/scratch/users/cehou/urban-visual-gene; ENV=/global/scratch/users/cehou/conda_envs/svi; PY=$ENV/bin/python
export LD_LIBRARY_PATH=$ENV/lib:${LD_LIBRARY_PATH:-}
export HF_TOKEN="$(cat ~/.cache/huggingface/token)" HF_HUB_DISABLE_XET=1
export PYTHONPATH=$REPO OMP_NUM_THREADS=8 KMP_DUPLICATE_LIB_OK=TRUE
export DICTDIR=$REPO/formal/formal_out_global2
export GENE_PROJ=$REPO/formal/formal_out_global/artifact_dirs_pos.npy
export CATJSON=$REPO/formal/categories_new.json
cd "$REPO"; nvidia-smi -L
rm -f $DICTDIR/web2/panos.parquet
echo "=== 1/5 compose_cities 6-parent (GPU) ==="; "$PY" -u -m formal.compose_cities
echo "=== 2/5 compose_analyze (CPU) ==="; "$PY" -u -m formal.compose_analyze
echo "=== 3/5 build_explorer (CPU) ==="; EXPDIR=formal_out_expglobal2 DICTDIR_NAME=formal_out_global2 "$PY" -u formal/build_explorer.py
echo "=== 4/5 interp_montage (CPU) ==="; "$PY" -u formal/interp_montage.py
echo "=== 5/5 coexpr_web (CPU) ==="; "$PY" -u formal/coexpr_web.py
echo "=== MIG6 DONE $(date '+%F %T') ==="
