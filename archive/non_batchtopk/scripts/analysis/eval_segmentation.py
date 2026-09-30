"""Evaluate any saved segmentation (node→unit labels) in the common DINO space.

Standalone scorer reused by stage7 and usable on external/ablation labels.

  python -m scripts.analysis.eval_segmentation --out outputs/sweep/Vienna_N2000 \
         --labels outputs/sweep/Vienna_N2000/baselines/labels_sae.parquet
"""
from __future__ import annotations

import scripts._env  # noqa: F401

import argparse
import json
from pathlib import Path

import pandas as pd

from scripts.analysis.baseline_common import evaluate, load_city


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True, help="city dir (for Z_road + graph)")
    ap.add_argument("--labels", required=True, help="parquet with road_node_id, unit")
    ap.add_argument("--full-graph", action="store_true")
    args = ap.parse_args()

    node_ids, Z, A, lat, lon, edges, n_panos = load_city(
        args.out, covered_only=not args.full_graph)
    lab_df = pd.read_parquet(args.labels)
    labels = dict(zip(lab_df["road_node_id"], lab_df["unit"]))

    metrics = evaluate(labels, node_ids, Z, edges)
    print(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    main()
