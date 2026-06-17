"""Train + evaluate a basis model at one K (for the K-sweep / H1 elbow).

Trains on the covered nodes of the given dir(s), infers activations, and records
reconstruction error, sparsity, and effective vocabulary (dominant/dead bases).

  python -m scripts.sampling.run_k_eval --dirs outputs/sweep/Vienna_N2000 --K 16 \
      --out outputs/ksweep/Vienna_K16
"""
from __future__ import annotations

import scripts._env  # noqa: F401

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from scripts.pipeline.stage4_train_road_basis_model import train_basis_model
from scripts.pipeline.stage5_infer_road_basis_activation import infer_activation

TAU = 0.01
MIN_DOM = 0.002


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dirs", nargs="+", required=True)
    ap.add_argument("--K", type=int, required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--epochs", type=int, default=50)
    args = ap.parse_args()
    out = Path(args.out); out.mkdir(parents=True, exist_ok=True)

    ctxs, edges = [], []
    for d in args.dirs:
        d = Path(d)
        c = pd.read_parquet(d / "road_context_features.parquet")
        ctxs.append(c[c["n_panos"] > 0])
        edges.append(pd.read_parquet(d / "road_graph_edges.parquet"))
    ctx = pd.concat(ctxs).reset_index(drop=True)
    edges = pd.concat(edges).reset_index(drop=True)

    model, r4 = train_basis_model(ctx, edges, K=args.K, hidden=512,
                                  epochs=args.epochs, lambda_sparse=5e-3,
                                  output_dir=out)
    act_df, r5 = infer_activation(ctx, model)
    a_cols = sorted([c for c in act_df.columns if c.startswith("a_")])
    A = act_df[a_cols].values.astype(np.float64)

    dom_frac = np.bincount(A.argmax(1), minlength=args.K) / len(A)
    act_frac = (A > TAU).mean(0)
    dominant = int((dom_frac >= MIN_DOM).sum())
    dead = int(((dom_frac < MIN_DOM) & (act_frac < 0.05)).sum())
    s = np.sort(dom_frac)[::-1]
    eff90 = int(np.searchsorted(np.cumsum(s), 0.90) + 1)

    rec = {"K": args.K, "recon_error": round(r5["mean_recon_error"], 4),
           "median_active": float(r5["median_active_basis"]),
           "dominant": dominant, "dead": dead, "eff_vocab_90": eff90,
           "n_covered": len(A)}
    (out / "k_eval.json").write_text(json.dumps(rec, indent=2))
    print(rec)


if __name__ == "__main__":
    main()
