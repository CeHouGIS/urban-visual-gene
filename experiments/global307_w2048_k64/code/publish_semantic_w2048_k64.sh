#!/usr/bin/env bash
set -euo pipefail

repo=$(git -C "$(dirname "${BASH_SOURCE[0]}")" rev-parse --show-toplevel)
checkpoint=/host/root/mnt/nas/huangyj/GoogleSV/urban_visual_gene/formal_batchtopk_307city_w2048_k64/sae_448_k2048.pt
result="$repo/reports/data/semantic_umap_w2048_k64_top100"
pages=/tmp/uvg-gh-pages-scaling-20261010
log="$repo/formal/logs/semantic_w2048_k64_publish.log"

mkdir -p "$(dirname "$log")"
exec > >(tee -a "$log") 2>&1
cd "$repo"
echo "Waiting for W2048/K64 training checkpoint at $(date -u)"
while [[ ! -s "$checkpoint" ]]; do sleep 30; done

if [[ -n $(git -C "$pages" status --porcelain) ]]; then
  echo "Pages worktree has uncommitted changes; publication requires review" >&2
  exit 1
fi

echo "Generating W2048/K64 semantic examples at $(date -u)"
bash formal/analysis/run_semantic_w2048_k64.sh all

echo "Exporting interactive site at $(date -u)"
python -m formal.analysis.build_semantic_site \
  --result "$result" \
  --out "$pages/semantic-2048" \
  --shared-assets "$pages/semantic" \
  --image-base ../semantic/ \
  --workers 8

echo "Preparing GitHub Pages update at $(date -u)"
git -C "$pages" fetch origin gh-pages
git -C "$pages" merge --ff-only origin/gh-pages
install -m 644 "$repo/formal/analysis/semantic_site/index.html" "$pages/semantic/index.html"
install -m 644 "$repo/formal/analysis/semantic_site/app.js" "$pages/semantic/app.js"
git -C "$pages" add --sparse semantic-2048 semantic/index.html semantic/app.js
if git -C "$pages" diff --cached --quiet; then
  echo "No Pages changes to publish"
  exit 0
fi
git -C "$pages" commit -m 'Publish W2048 BatchTopK semantic atlas'
git -C "$pages" push origin HEAD:gh-pages
echo "Published W2048/K64 semantic atlas at $(date -u)"
