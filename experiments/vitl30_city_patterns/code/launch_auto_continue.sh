#!/usr/bin/env bash
set -euo pipefail

repo=$(cd "$(dirname "$0")/../../.." && pwd)
root=/host/root/mnt/nas/huangyj/GoogleSV/urban_visual_gene/vitl30_city_patterns_1km10
output="$root/followup"
pid_file="$output/auto_continue.pid"
mkdir -p "$output"

if [[ -f "$pid_file" ]]; then
  read -r old_pid < "$pid_file"
  old_args=$(ps -p "$old_pid" -o args= 2>/dev/null || true)
  if [[ "$old_args" == *vitl30_city_patterns.code.auto_continue* ]]; then
    printf 'Sequential task runner already running: PID %s\n' "$old_pid"
    exit 0
  fi
fi

cd "$repo"
setsid env PYTHONPATH=. python -u -m experiments.vitl30_city_patterns.code.auto_continue \
  --extraction "$root/extraction" --followup "$output" \
  > "$output/auto_continue.log" 2>&1 < /dev/null &
printf '%s\n' "$!" > "$pid_file"
printf 'Sequential task runner started: PID %s\n' "$!"
