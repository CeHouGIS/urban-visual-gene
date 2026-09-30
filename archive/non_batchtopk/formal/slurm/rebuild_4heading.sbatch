#!/bin/bash
#SBATCH --job-name=uvg-4head
#SBATCH --account=fc_globalsvf
#SBATCH --partition=savio4_gpu
#SBATCH --qos=a5k_gpu4_normal
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:A5000:1
#SBATCH --time=03:00:00
#SBATCH --requeue
#SBATCH --output=/global/scratch/users/cehou/urban-visual-gene/formal/slurm-4head-%j.out
set -eu
# Rebuild the 512-gene dashboard AND the compose page from 4-heading sampling:
#   gene_encode  -> each pano contributes all 4 headings as separate exemplar images
#   compose_cities -> per-point composition pools patches over all 4 headings (360°)
# Dictionary + native 5-parent taxonomy (gene2cat.npy) are UNCHANGED, so we do NOT
# re-run build_taxonomy_g3; the explorer is already 4-heading, so it is skipped too.
REPO=/global/scratch/users/cehou/urban-visual-gene; ENV=/global/scratch/users/cehou/conda_envs/svi; PY=$ENV/bin/python
export LD_LIBRARY_PATH=$ENV/lib:${LD_LIBRARY_PATH:-}
export HF_TOKEN="$(cat ~/.cache/huggingface/token)" HF_HUB_DISABLE_XET=1
export PYTHONPATH=$REPO OMP_NUM_THREADS=8 KMP_DUPLICATE_LIB_OK=TRUE
export DICTDIR=$REPO/formal/formal_out_global3
export GENE_PROJ=$REPO/formal/formal_out_global/artifact_dirs_pos.npy
export CATJSON=$REPO/formal/categories_g3.json
export GENESDIR=$DICTDIR/genes
cd "$REPO"; nvidia-smi -L

# clean stale outputs so nothing carries over from the heading-0 build
rm -f  $DICTDIR/web2/panos.parquet $GENESDIR/sparse_acts.npz $GENESDIR/atoms.npy
rm -rf $GENESDIR/thumbs $GENESDIR/web/exem $DICTDIR/web2/thumbs

echo "=== GPU 1/2 gene_encode (4 headings/pano) ==="; "$PY" -u -m formal.genes.gene_encode
echo "=== GPU 2/2 compose_cities (4-heading pooled) ==="; "$PY" -u -m formal.web.compose_cities

echo "=== CPU 1/8 gene_render ===";  "$PY" -u -m formal.genes.gene_render
echo "=== CPU 2/8 gene_orig ===";    "$PY" -u -m formal.genes.gene_orig
echo "=== CPU 3/8 gene_posdiag ==="; "$PY" -u -m formal.genes.gene_posdiag
echo "=== CPU 4/8 patch_manifest (native 5-parent taxonomy) ==="; "$PY" -u -m formal.genes.patch_manifest
echo "=== CPU 5/8 gene_tree ===";    "$PY" -u -m formal.genes.gene_tree --super 80 --sub 28
echo "=== CPU 6/8 coexpr_web ===";   "$PY" -u -m formal.genes.coexpr_web
echo "=== CPU 7/8 compose_analyze ==="; "$PY" -u -m formal.web.compose_analyze
echo "=== CPU 8/8 interp_montage + dendro512 ==="
"$PY" -u -m formal.web.interp_montage
DICTDIR_NAME=formal_out_global3 "$PY" -u -m formal.web.dendro512
echo "=== 4HEAD DONE $(date '+%F %T') ==="
