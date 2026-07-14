#!/bin/bash
#SBATCH --job-name=uvg-mig-g3
#SBATCH --account=fc_globalsvf
#SBATCH --partition=savio4_gpu
#SBATCH --qos=savio_lowprio
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:L40:1
#SBATCH --time=02:30:00
#SBATCH --requeue
#SBATCH --output=/global/scratch/users/cehou/urban-visual-gene/formal/slurm-migg3-%j.out
set -eu
REPO=/global/scratch/users/cehou/urban-visual-gene; ENV=/global/scratch/users/cehou/conda_envs/svi; PY=$ENV/bin/python
export LD_LIBRARY_PATH=$ENV/lib:${LD_LIBRARY_PATH:-}
export HF_TOKEN="$(cat ~/.cache/huggingface/token)" HF_HUB_DISABLE_XET=1
export PYTHONPATH=$REPO OMP_NUM_THREADS=8 KMP_DUPLICATE_LIB_OK=TRUE
export DICTDIR=$REPO/formal/formal_out_global3
export GENE_PROJ=$REPO/formal/formal_out_global/artifact_dirs_pos.npy
export CATJSON=$REPO/formal/categories_new.json
export GENESDIR=$DICTDIR/genes
cd "$REPO"; nvidia-smi -L
echo "=== GPU 1/4 gene_encode ===";   "$PY" -u -m formal.genes.gene_encode
echo "=== GPU 2/4 compose_cities ==="; "$PY" -u -m formal.web.compose_cities
echo "=== GPU 3/4 render_global ==="; "$PY" -u -m formal.web.render_global
echo "=== GPU 4/4 explorer street inference ==="
FORMAL_OUT=$REPO/formal/formal_out_expglobal3 "$PY" -u -m formal.gpu_run --infer-only \
  --sae-path $DICTDIR/sae_448_k512.pt --project-dirs $GENE_PROJ \
  --cities HongKong Singapore Amsterdam CapeTown --K-list 512
echo "=== CPU 1/9 gene_render ===";  "$PY" -u -m formal.genes.gene_render
echo "=== CPU 2/9 gene_orig ===";    "$PY" -u -m formal.genes.gene_orig
echo "=== CPU 3/9 gene_posdiag ==="; "$PY" -u -m formal.genes.gene_posdiag
echo "=== CPU 4/9 patch_manifest (transferred taxonomy) ==="; "$PY" -u -m formal.genes.patch_manifest
echo "=== CPU 5/9 gene_tree ===";    "$PY" -u -m formal.genes.gene_tree --super 80 --sub 28
echo "=== CPU 6/9 coexpr_web ===";   "$PY" -u -m formal.genes.coexpr_web
echo "=== CPU 7/9 compose_analyze ==="; "$PY" -u -m formal.web.compose_analyze
echo "=== CPU 8/9 build_explorer ==="
EXPDIR=formal_out_expglobal3 DICTDIR_NAME=formal_out_global3 "$PY" -u -m formal.web.build_explorer
echo "=== CPU 9/9 interp_montage + dendro512 ==="
"$PY" -u -m formal.web.interp_montage
DICTDIR_NAME=formal_out_global3 "$PY" -u -m formal.web.dendro512
echo "=== MIGG3 DONE $(date '+%F %T') ==="
