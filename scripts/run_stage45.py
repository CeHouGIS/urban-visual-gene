"""Isolated Stage 4+5 runner (uses torch) — train basis + infer activation.

Reads Z_road, trains the sparse autoencoder on covered nodes, infers the
activation A for all nodes, and writes the model + activation. Kept separate
from Stage 3/6 because torch must not be imported in those (it corrupts
numpy/scipy fancy-indexing in this environment).

  python -m scripts.run_stage45 --out outputs/China/HongKong --K 32 --epochs 50
"""
from __future__ import annotations

import scripts._env  # noqa: F401  (sets thread limits before numpy/torch)

import argparse
from pathlib import Path

import pandas as pd

from scripts.io_utils import save_report
from scripts.stage4_train_road_basis_model import train_basis_model
from scripts.stage5_infer_road_basis_activation import infer_activation


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--K", type=int, default=32)
    ap.add_argument("--hidden", type=int, default=512)
    ap.add_argument("--epochs", type=int, default=50)
    ap.add_argument("--lambda-sparse", type=float, default=5e-3)
    args = ap.parse_args()
    out = Path(args.out)

    ctx_df = pd.read_parquet(out / "road_context_features.parquet")
    edges = pd.read_parquet(out / "road_graph_edges.parquet")

    ctx_covered = ctx_df[ctx_df["n_panos"] > 0].copy().reset_index(drop=True)
    print(f"[Stage 4] training on {len(ctx_covered)}/{len(ctx_df)} covered nodes")
    model, r4 = train_basis_model(
        ctx_covered, edges,
        K=args.K, hidden=args.hidden, epochs=args.epochs,
        lambda_sparse=args.lambda_sparse,
        lambda_spatial=1e-3, lambda_div=1e-3,
        output_dir=Path("models"),
    )
    save_report(out / "stage_reports/stage4_report.json", r4)
    print(f"  ✓ loss {r4['initial_train_loss']:.4f} → {r4['final_train_loss']:.4f}")

    act_df, r5 = infer_activation(ctx_df, model)
    act_df.to_parquet(out / "road_basis_activation.parquet", index=False)
    save_report(out / "stage_reports/stage5_report.json", r5)
    print(f"[Stage 5] ✓ median_active={r5['median_active_basis']:.1f}/{args.K}, "
          f"mean_recon={r5['mean_recon_error']:.4f}")


if __name__ == "__main__":
    main()
