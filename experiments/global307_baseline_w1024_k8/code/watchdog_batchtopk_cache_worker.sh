#!/usr/bin/env bash
# Resumable city-cache worker. Static modulo assignment keeps workers from
# touching the same city across restarts.
set -u

ROOT=/root/code/urban-visual-gene
OUT=${FORMAL_OUT:-/host/root/mnt/nas/huangyj/GoogleSV/urban_visual_gene/formal_batchtopk_307city_balanced_3000}
CITY_MAP=${CITY_MAP:-$ROOT/configs/final_global_300_google_city_map.json}
BAIDU_MAP=${BAIDU_MAP:-$ROOT/configs/final_global_7_baidu_city_map.json}
SAMPLE_JSON=${SAMPLE_JSON:-$ROOT/configs/final_global_307_sample_3000.json}
MODEL=${DINO_MODEL_PATH:-/root/.cache/modelscope/models/facebook--dinov3-vitl16-pretrain-lvd1689m/snapshots/master}
WORKER_INDEX=${WORKER_INDEX:?set WORKER_INDEX to 0 or 1}
WORKER_COUNT=${WORKER_COUNT:-2}
BATCH=${BATCH:-512}
MICROBATCH=${MICROBATCH:-128}
IO_WORKERS=${IO_WORKERS:-96}
LOG=${FORMAL_LOG:-$ROOT/formal/logs/formal_batchtopk_cache_worker_${WORKER_INDEX}.log}
WATCHDOG_LOG=${FORMAL_WATCHDOG_LOG:-$ROOT/formal/logs/formal_batchtopk_cache_worker_${WORKER_INDEX}_watchdog.log}

mkdir -p "$ROOT/formal/logs" "$OUT"
cd "$ROOT"
stamp() { date -u '+%F %T UTC'; }
note() { printf '[%s] %s\n' "$(stamp)" "$*" >> "$WATCHDOG_LOG"; }
child=""
cleanup() {
  if [[ -n "$child" ]] && kill -0 "$child" 2>/dev/null; then
    kill -TERM "$child" 2>/dev/null || true
    wait "$child" 2>/dev/null || true
  fi
  exit 143
}
trap cleanup TERM INT

mapfile -t CITIES < <(python - "$CITY_MAP" "$BAIDU_MAP" "$WORKER_INDEX" "$WORKER_COUNT" <<'PY'
import json, sys
google = list(json.load(open(sys.argv[1])).keys())
baidu = list(json.load(open(sys.argv[2])).keys())
cities = google + [city for city in baidu if city not in google]
index, count = int(sys.argv[3]), int(sys.argv[4])
for city in cities[index::count]:
    print(city)
PY
)
note "worker $WORKER_INDEX/$WORKER_COUNT assigned ${#CITIES[@]} cities on CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-unset}"

while :; do
  note "starting/resuming city-cache worker"
  env DINO_MODEL_PATH="$MODEL" FORMAL_OUT="$OUT" \
    GOOGLE_SV_ROOT=/host/root/mnt/nas/huangyj/GoogleSV PYTHONPATH="$ROOT" \
    python -u -m formal.gpu_run \
      --city-map "$CITY_MAP" --baidu-map "$BAIDU_MAP" --cities "${CITIES[@]}" \
      --dict-panos 3000 --keep-patches 784 --K-list 1024 --topk 8 --epochs 60 \
      --sae-batch-size 16384 --train-block-rows 262144 --mix-stripe-rows 1024 \
      --context-weight 0.25 --sample-json "$SAMPLE_JSON" --skip-infer \
      --cache-only --batch "$BATCH" --microbatch "$MICROBATCH" --io-workers "$IO_WORKERS" \
      >> "$LOG" 2>&1 &
  child=$!
  wait "$child"
  rc=$?
  child=""
  if [[ "$rc" == 0 ]]; then
    note "city-cache worker complete"
    exit 0
  fi
  note "worker exited rc=$rc; retrying in 30 seconds"
  sleep 30
done
