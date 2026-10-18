#!/usr/bin/env bash
set -euo pipefail

repo=$(cd "$(dirname "$0")/../../.." && pwd)
output=/host/root/mnt/nas/huangyj/GoogleSV/urban_visual_gene/vitl30_city_patterns_1km10/extraction
sample=/host/root/mnt/nas/huangyj/GoogleSV/urban_visual_gene/vitl30_city_patterns_1km10/selection
if [[ ! -f "$sample/manifest.json" ]]; then
  printf 'Selection manifest missing: %s\n' "$sample/manifest.json" >&2
  exit 1
fi
mkdir -p "$output/logs"

for gpu in 0 1 2 3; do
  pid_file="$output/logs/shard${gpu}.pid"
  if [[ -f "$pid_file" ]]; then
    read -r old_pid < "$pid_file"
    if kill -0 "$old_pid" 2>/dev/null; then
      printf 'shard %s already running: PID %s\n' "$gpu" "$old_pid"
      continue
    fi
  fi
  setsid bash "$repo/experiments/vitl30_city_patterns/code/worker.sh" \
    "$repo" "$output" "$sample" "$gpu" \
    > "$output/logs/shard${gpu}.log" 2>&1 < /dev/null &
  printf '%s\n' "$!" > "$pid_file"
  printf 'shard %s started: PID %s\n' "$gpu" "$!"
done
