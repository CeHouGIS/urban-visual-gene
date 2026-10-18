#!/usr/bin/env bash
set -u

repo=$1
output=$2
sample=$3
gpu=$4
cd "$repo" || exit 1

for attempt in $(seq 1 10); do
  printf 'shard %s attempt %s started %s\n' "$gpu" "$attempt" "$(date -u +%FT%TZ)"
  if PYTHONPATH=. python -u -m experiments.vitl30_city_patterns.code.extract \
      --output "$output" --sample "$sample" --shards 4 --shard "$gpu" --device "cuda:$gpu" \
      --rows-per-chunk 1024 --batch-panos 2 --io-workers 4; then
    printf 'shard %s complete %s\n' "$gpu" "$(date -u +%FT%TZ)"
    exit 0
  fi
  printf 'shard %s failed on attempt %s at %s\n' "$gpu" "$attempt" "$(date -u +%FT%TZ)" >&2
  sleep 60
done

printf 'shard %s exhausted retries at %s\n' "$gpu" "$(date -u +%FT%TZ)" >&2
exit 1
