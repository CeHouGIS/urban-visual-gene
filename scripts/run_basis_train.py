"""Train one basis matrix X from one or more cities' covered nodes (S3).

Used to learn Vienna-only, HongKong-only and joint bases for cross-city
alignment. Saves the K×D basis to <out>/road_landscape_basis.npy.

  python -m scripts.run_basis_train --dirs outputs/sweep/Vienna_N2000 \
      --out outputs/transfer/vienna --K 32 --epochs 50
"""
from __future__ import annotations

import scripts._env  # noqa: F401

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from scripts.stage4_train_road_basis_model import train_basis_model


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dirs", nargs="+", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--K", type=int, default=32)
    ap.add_argument("--epochs", type=int, default=50)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()
    out = Path(args.out); out.mkdir(parents=True, exist_ok=True)

    ctxs, edges_list = [], []
    for d in args.dirs:
        d = Path(d)
        c = pd.read_parquet(d / "road_context_features.parquet")
        c = c[c["n_panos"] > 0]
        ctxs.append(c)
        edges_list.append(pd.read_parquet(d / "road_graph_edges.parquet"))
    ctx = pd.concat(ctxs).reset_index(drop=True)
    edges = pd.concat(edges_list).reset_index(drop=True)
    print(f"training on {len(ctx)} covered nodes from {len(args.dirs)} city/cities",
          flush=True)

    _, r = train_basis_model(ctx, edges, K=args.K, hidden=512, epochs=args.epochs,
                             lambda_sparse=5e-3, seed=args.seed, output_dir=out)
    X = np.load(out / "road_landscape_basis.npy")
    print(f"saved basis {X.shape} -> {out/'road_landscape_basis.npy'} "
          f"(loss {r['initial_train_loss']:.3f}->{r['final_train_loss']:.3f})")


if __name__ == "__main__":
    main()
