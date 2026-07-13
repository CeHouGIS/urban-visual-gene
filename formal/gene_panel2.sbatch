#!/bin/bash
#SBATCH --job-name=uvg-panel2
#SBATCH --account=fc_globalsvf
#SBATCH --partition=savio4_gpu
#SBATCH --qos=savio_lowprio
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:L40:1
#SBATCH --time=01:30:00
#SBATCH --requeue
#SBATCH --output=/global/scratch/users/cehou/urban-visual-gene/formal/slurm-panel2-%j.out
set -eu
REPO=/global/scratch/users/cehou/urban-visual-gene
ENV=/global/scratch/users/cehou/conda_envs/svi
PY=$ENV/bin/python
export LD_LIBRARY_PATH=$ENV/lib:${LD_LIBRARY_PATH:-}
export HF_TOKEN="$(cat ~/.cache/huggingface/token)" HF_HUB_DISABLE_XET=1
export PYTHONPATH=$REPO OMP_NUM_THREADS=8 KMP_DUPLICATE_LIB_OK=TRUE
# new dict + its matching 11-dim positional projection
export DICTDIR=$REPO/formal/formal_out_global2
export GENE_PROJ=$REPO/formal/formal_out_global/artifact_dirs_pos.npy
export GENESDIR=$DICTDIR/genes
cd "$REPO"
nvidia-smi -L
echo "=== 1/5 encode (GPU) ==="; "$PY" -u -m formal.gene_encode
echo "=== 2/5 render (CPU) ==="; "$PY" -u formal/gene_render.py
echo "=== 3/5 originals (CPU) ==="; "$PY" -u formal/gene_orig.py
echo "=== 4/5 balanced tree (CPU) ==="; "$PY" -u formal/gene_tree.py --super 110 --sub 28
echo "=== 5/5 positional diag (CPU) ==="; "$PY" -u formal/gene_posdiag.py
echo "=== PANEL2 DONE $(date '+%F %T') ==="
