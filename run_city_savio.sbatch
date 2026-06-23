#!/bin/bash
#SBATCH --job-name=uvg
#SBATCH --account=fc_globalsvf
#SBATCH --partition=savio3_gpu
#SBATCH --qos=a40_gpu3_normal
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:A40:1
#SBATCH --time=24:00:00
#SBATCH --requeue
#SBATCH --open-mode=append
#SBATCH --output=/global/scratch/users/cehou/urban-visual-gene/outputs/slurm-uvg-%x-%j.out
# ============================================================================
# Urban Visual Gene — MRLU pipeline (Stage 1-6) over several cities, in ONE
# GPU allocation, run sequentially. Resumable: a city whose final artefact
# already exists is skipped, so a requeue (e.g. lowprio preemption) continues
# where it left off.
#
#   CITIES="HongKong Singapore Amsterdam CapeTown" sbatch run_city_savio.sbatch
#
# torch (cu13.0) lives in ~/.local -> do NOT set PYTHONNOUSERSITE here.
# Process isolation (run_experiment subprocesses) avoids the torch-vs-geo
# OpenMP clash; _env.py + run_experiment THREAD_ENV lock thread pools to 1.
# ============================================================================
set -u
CITIES="${CITIES:-HongKong Singapore Amsterdam CapeTown}"
REPO=/global/scratch/users/cehou/urban-visual-gene
PY=/global/scratch/users/cehou/conda_envs/svi/bin/python3
export PROJ_DATA=/global/scratch/users/cehou/conda_envs/svi/share/proj
export PYTHONPATH=$REPO
cd "$REPO"

declare -A OUT=( [HongKong]=outputs/China/HongKong [Singapore]=outputs/Singapore/Singapore \
                 [Amsterdam]=outputs/Netherlands/Amsterdam [CapeTown]=outputs/SouthAfrica/CapeTown )

echo "=== UVG batch cities='$CITIES' job=${SLURM_JOB_ID:-NA} host=$(hostname) $(date '+%F %T') ==="
nvidia-smi -L || echo "WARN: no GPU visible"

for CITY in $CITIES; do
  done_marker="${OUT[$CITY]}/minimum_road_landscape_units.geojson"
  if [ -f "$done_marker" ]; then
    echo ">>> [$CITY] already complete ($done_marker) — skipping  $(date '+%F %T')"
    continue
  fi
  echo ">>> [$CITY] START $(date '+%F %T')"
  "$PY" -u run_experiment.py --city "$CITY" --max-panos 5000 --K 32 --epochs 50
  rc=$?
  echo ">>> [$CITY] END rc=$rc $(date '+%F %T')"
  if [ $rc -ne 0 ]; then echo "!!! [$CITY] FAILED rc=$rc"; fi
done
echo "=== UVG batch DONE $(date '+%F %T') ==="
