#!/usr/bin/env bash
# Keep the resumable formal BatchTopK job alive across terminal/session loss.
# The first PID may be an already-running job; later jobs are child processes
# of this watchdog and are waited on directly.
set -u

ROOT=/root/code/urban-visual-gene
OUT="$ROOT/formal/formal_out_panorama_context"
LOG="$ROOT/formal/logs/formal_batchtopk_gpu12_resume.log"
WATCHDOG_LOG="$ROOT/formal/logs/formal_batchtopk_watchdog.log"
PID_FILE="$ROOT/formal/logs/formal_batchtopk_gpu12.pid"
MODEL=/root/.cache/modelscope/models/facebook--dinov3-vitl16-pretrain-lvd1689m/snapshots/master
ADOPT_PID="${1:-}"

mkdir -p "$ROOT/formal/logs"
cd "$ROOT"

stamp() { date -u '+%F %T UTC'; }
note() { printf '[%s] %s\n' "$(stamp)" "$*" >> "$WATCHDOG_LOG"; }

if [[ -n "$ADOPT_PID" ]]; then
  note "adopting existing BatchTopK PID $ADOPT_PID"
  while kill -0 "$ADOPT_PID" 2>/dev/null; do
    sleep 30
  done
  note "adopted PID $ADOPT_PID exited; checking checkpoint"
  ADOPT_PID=""
fi

while :; do
  if [[ -f "$OUT/sae_448_k1024.pt" ]]; then
    note "formal SAE checkpoint exists; watchdog complete"
    rm -f "$PID_FILE"
    exit 0
  fi

  note "starting/resuming formal BatchTopK"
  env CUDA_VISIBLE_DEVICES=1,2 \
    DINO_MODEL_PATH="$MODEL" \
    FORMAL_OUT="$OUT" \
    GOOGLE_SV_ROOT=/host/root/mnt/nas/huangyj/GoogleSV \
    PYTHONPATH="$ROOT" \
    python -u -m formal.gpu_run \
      --cities $(python -c 'import json; print(" ".join(json.load(open("configs/formal_city_manifest.json"))))') \
      --dict-panos 12800 --K-list 1024 --topk 8 --epochs 60 \
      --context-weight 0.25 --sample-json configs/formal_city_manifest.json \
      --skip-infer --batch 256 --microbatch 64 --io-workers 32 \
      >> "$LOG" 2>&1 &
  child=$!
  printf '%s\n' "$child" > "$PID_FILE"
  note "child PID $child started"
  wait "$child"
  rc=$?
  rm -f "$PID_FILE"

  if [[ -f "$OUT/sae_448_k1024.pt" ]]; then
    note "formal BatchTopK finished successfully (child rc=$rc)"
    exit 0
  fi
  note "child PID $child exited rc=$rc without checkpoint; retrying in 30 seconds"
  sleep 30
done
