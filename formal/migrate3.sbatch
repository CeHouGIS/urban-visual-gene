#!/bin/bash
#SBATCH --job-name=uvg-migrate3
#SBATCH --account=fc_globalsvf
#SBATCH --partition=savio4_gpu
#SBATCH --qos=savio_lowprio
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:L40:1
#SBATCH --time=02:30:00
#SBATCH --requeue
#SBATCH --output=/global/scratch/users/cehou/urban-visual-gene/formal/slurm-migrate3-%j.out
set -eu
REPO=/global/scratch/users/cehou/urban-visual-gene
ENV=/global/scratch/users/cehou/conda_envs/svi
PY=$ENV/bin/python
export LD_LIBRARY_PATH=$ENV/lib:${LD_LIBRARY_PATH:-}
export HF_TOKEN="$(cat ~/.cache/huggingface/token)" HF_HUB_DISABLE_XET=1
export PYTHONPATH=$REPO OMP_NUM_THREADS=8 KMP_DUPLICATE_LIB_OK=TRUE
# new de-biased dict + its 11-dim positional projection + new category names
export DICTDIR=$REPO/formal/formal_out_global2
export GENE_PROJ=$REPO/formal/formal_out_global/artifact_dirs_pos.npy
export CATJSON=$REPO/formal/categories_new.json
cd "$REPO"
nvidia-smi -L
echo "=== 1/6 compose_cities (GPU, 3000 panos) ==="; "$PY" -u -m formal.compose_cities
echo "=== 2/6 render_global gallery/montage (GPU) ==="; "$PY" -u -m formal.render_global
echo "=== 3/6 explorer street inference on new dict (GPU) ==="
FORMAL_OUT=$REPO/formal/formal_out_expglobal2 "$PY" -u -m formal.gpu_run --infer-only \
  --sae-path $DICTDIR/sae_448_k512.pt --project-dirs $GENE_PROJ \
  --cities HongKong Singapore Amsterdam CapeTown --K-list 512
echo "=== 4/6 compose_analyze (CPU) ==="; "$PY" -u -m formal.compose_analyze
echo "=== 5/6 build_explorer (CPU) ==="
EXPDIR=formal_out_expglobal2 DICTDIR_NAME=formal_out_global2 "$PY" -u formal/build_explorer.py
echo "=== 6/6 dendrogram_512 (CPU) ==="; DICTDIR_NAME=formal_out_global2 "$PY" -u formal/dendro512.py
echo "=== MIGRATE3 DONE $(date '+%F %T') ==="
