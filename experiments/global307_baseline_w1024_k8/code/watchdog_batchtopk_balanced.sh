#!/usr/bin/env bash
# Resumable BatchTopK run for the balanced GoogleSV + BaiduSV city sample.
set -u

ROOT=/root/code/urban-visual-gene
OUT=${FORMAL_OUT:-/host/root/mnt/nas/huangyj/GoogleSV/urban_visual_gene/formal_batchtopk_307city_balanced_3000}
CITY_MAP=${CITY_MAP:-$ROOT/configs/final_global_300_google_city_map.json}
BAIDU_MAP=${BAIDU_MAP:-$ROOT/configs/final_global_7_baidu_city_map.json}
SAMPLE_JSON=${SAMPLE_JSON:-$ROOT/configs/final_global_307_sample_3000.json}
LOG=${FORMAL_LOG:-$ROOT/formal/logs/formal_batchtopk_307city_balanced_3000.log}
WATCHDOG_LOG=${FORMAL_WATCHDOG_LOG:-$ROOT/formal/logs/formal_batchtopk_307city_balanced_3000_watchdog.log}
PID_FILE=${FORMAL_PID_FILE:-$ROOT/formal/logs/formal_batchtopk_307city_balanced_3000.pid}
MODEL=${DINO_MODEL_PATH:-/root/.cache/modelscope/models/facebook--dinov3-vitl16-pretrain-lvd1689m/snapshots/master}
BATCH=${BATCH:-512}
MICROBATCH=${MICROBATCH:-128}
IO_WORKERS=${IO_WORKERS:-96}

mkdir -p "$ROOT/formal/logs" "$OUT"
cd "$ROOT"
stamp() { date -u '+%F %T UTC'; }
note() { printf '[%s] %s\n' "$(stamp)" "$*" >> "$WATCHDOG_LOG"; }

while :; do
  if [[ -f "$OUT/sae_448_k1024.pt" ]]; then
    note "balanced 300-city SAE checkpoint exists; watchdog complete"
    rm -f "$PID_FILE"
    exit 0
  fi

  note "starting/resuming balanced 307-city BatchTopK"
  extra_args=()
  if [[ -f "$OUT/dict_sample_allpatches.f16.npy" || -f "$OUT/sample_manifest.json" ]]; then
    extra_args+=(--train-only)
  fi
  env CUDA_VISIBLE_DEVICES=1,2 \
    DINO_MODEL_PATH="$MODEL" \
    FORMAL_OUT="$OUT" \
    GOOGLE_SV_ROOT=/host/root/mnt/nas/huangyj/GoogleSV \
    PYTHONPATH="$ROOT" \
    python -u -m formal.gpu_run \
      --city-map "$CITY_MAP" \
      --baidu-map "$BAIDU_MAP" \
      --dict-panos 3000 --keep-patches 784 --K-list 1024 --topk 8 --epochs 60 \
      --sae-batch-size 16384 --train-block-rows 262144 --mix-stripe-rows 1024 \
      --context-weight 0.25 --sample-json "$SAMPLE_JSON" \
      --skip-infer --batch "$BATCH" --microbatch "$MICROBATCH" --io-workers "$IO_WORKERS" \
      "${extra_args[@]}" >> "$LOG" 2>&1 &
  child=$!
  printf '%s\n' "$child" > "$PID_FILE"
  note "child PID $child started"
  wait "$child"
  rc=$?
  rm -f "$PID_FILE"

  if [[ -f "$OUT/sae_448_k1024.pt" ]]; then
    note "balanced 307-city BatchTopK finished successfully (child rc=$rc)"
    exit 0
  fi
  note "child PID $child exited rc=$rc without checkpoint; retrying in 30 seconds"
  sleep 30
done
