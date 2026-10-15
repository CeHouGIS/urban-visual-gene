#!/usr/bin/env bash
# Start SAE only after both disjoint city-cache workers finish successfully.
set -u

ROOT=/root/code/urban-visual-gene
OUT=${FORMAL_OUT:-/host/root/mnt/nas/huangyj/GoogleSV/urban_visual_gene/formal_batchtopk_307city_balanced_3000}
LOG=${FORMAL_TRAIN_LOG:-$ROOT/formal/logs/formal_batchtopk_307city_train.log}
WATCHDOG_LOG=${FORMAL_TRAIN_WATCHDOG_LOG:-$ROOT/formal/logs/formal_batchtopk_307city_train_watchdog.log}
CITY_MAP=$ROOT/configs/final_global_300_google_city_map.json
BAIDU_MAP=$ROOT/configs/final_global_7_baidu_city_map.json
SAMPLE_JSON=$ROOT/configs/final_global_307_sample_3000.json
WORKER_0_PID=${WORKER_0_PID:-783654}
WORKER_1_PID=${WORKER_1_PID:-783655}

mkdir -p "$ROOT/formal/logs"
cd "$ROOT"
note() { printf '[%s] %s\n' "$(date -u '+%F %T UTC')" "$*" >> "$WATCHDOG_LOG"; }

while :; do
  ready=1
  for index in 0 1; do
    worker_log="$ROOT/formal/logs/formal_batchtopk_cache_worker_${index}_watchdog.log"
    if ! grep -q 'city-cache worker complete' "$worker_log"; then
      ready=0
      worker_pid_var="WORKER_${index}_PID"
      worker_pid=${!worker_pid_var}
      if ! kill -0 "$worker_pid" 2>/dev/null; then
        note "cache worker $index stopped before completion; training not started"
        exit 1
      fi
    fi
  done
  [[ "$ready" == 1 ]] && break
  sleep 120
done

if ! python - "$OUT" "$CITY_MAP" "$BAIDU_MAP" <<'PY' >> "$WATCHDOG_LOG" 2>&1
import json
import sys
from pathlib import Path

out = Path(sys.argv[1])
google = json.loads(Path(sys.argv[2]).read_text())
baidu = json.loads(Path(sys.argv[3]).read_text())
cities = list(google) + [city for city in baidu if city not in google]
cached = {p.name[5:-len('_p784.f16.npy')] for p in out.glob('dict_*_p784.f16.npy')}
missing = [city for city in cities if city not in cached]
print(f'city cache complete: {len(cached & set(cities))}/{len(cities)}; missing={missing}', flush=True)
if set(missing) != {'Houston', 'Dallas', 'Indianapolis'}:
    raise SystemExit('unexpected missing city caches; training not started')
PY
then
  note "city-cache validation failed; training not started"
  exit 1
fi

while [[ ! -f "$OUT/sae_448_k1024.pt" ]]; do
  note "starting/resuming BatchTopK SAE from city feature caches"
  env CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-1} FORMAL_OUT="$OUT" PYTHONPATH="$ROOT" \
    python -u -m formal.gpu_run \
      --city-map "$CITY_MAP" --baidu-map "$BAIDU_MAP" \
      --sample-json "$SAMPLE_JSON" --train-only --skip-infer \
      --keep-patches 784 --K-list 1024 --topk 8 --epochs 60 \
      --sae-batch-size 8192 --max-train-patches 30000000 --train-block-rows 262144 --mix-stripe-rows 1024 \
      --context-weight 0.25 >> "$LOG" 2>&1 &
  child=$!
  wait "$child"
  rc=$?
  if [[ -f "$OUT/sae_448_k1024.pt" ]]; then
    note "BatchTopK SAE checkpoint complete (child rc=$rc)"
    exit 0
  fi
  note "training child exited rc=$rc without final checkpoint; retrying in 30 seconds"
  sleep 30
done
note "BatchTopK SAE checkpoint already exists"
