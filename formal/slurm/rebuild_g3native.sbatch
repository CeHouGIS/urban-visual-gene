#!/bin/bash
#SBATCH --job-name=uvg-g3nat
#SBATCH --account=fc_globalsvf
#SBATCH --partition=savio4_gpu
#SBATCH --qos=savio_lowprio
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:L40:1
#SBATCH --time=01:30:00
#SBATCH --requeue
#SBATCH --output=/global/scratch/users/cehou/urban-visual-gene/formal/slurm-g3nat-%j.out
set -eu
REPO=/global/scratch/users/cehou/urban-visual-gene; ENV=/global/scratch/users/cehou/conda_envs/svi; PY=$ENV/bin/python
export LD_LIBRARY_PATH=$ENV/lib:${LD_LIBRARY_PATH:-}
export HF_TOKEN="$(cat ~/.cache/huggingface/token)" HF_HUB_DISABLE_XET=1
export PYTHONPATH=$REPO OMP_NUM_THREADS=8 KMP_DUPLICATE_LIB_OK=TRUE
export DICTDIR=$REPO/formal/formal_out_global3
export GENE_PROJ=$REPO/formal/formal_out_global/artifact_dirs_pos.npy
export CATJSON=$REPO/formal/categories_g3.json
export GENESDIR=$DICTDIR/genes
cd "$REPO"; nvidia-smi -L
rm -f $DICTDIR/web2/panos.parquet
echo "=== GPU compose_cities (5-parent composition) ==="; "$PY" -u -m formal.web.compose_cities
echo "=== CPU patch_manifest (global3-native taxonomy) ==="; "$PY" -u -m formal.genes.patch_manifest
echo "=== CPU gene_tree ==="; "$PY" -u -m formal.genes.gene_tree --super 80 --sub 28
echo "=== CPU interp_montage ==="; "$PY" -u -m formal.web.interp_montage
echo "=== CPU coexpr_web ==="; "$PY" -u -m formal.genes.coexpr_web
echo "=== CPU build_explorer ==="; EXPDIR=formal_out_expglobal3 DICTDIR_NAME=formal_out_global3 "$PY" -u -m formal.web.build_explorer
echo "=== CPU compose_analyze ==="; "$PY" -u -m formal.web.compose_analyze
echo "=== CPU dendro512 ==="; DICTDIR_NAME=formal_out_global3 "$PY" -u -m formal.web.dendro512
echo "=== G3NAT DONE $(date '+%F %T') ==="
