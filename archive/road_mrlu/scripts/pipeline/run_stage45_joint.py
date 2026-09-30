"""Joint Stage 4-5 (torch) — ONE SAE basis shared across all cities.

Pools covered road nodes (Z_road) from every city, balanced-subsamples them,
trains a SINGLE sparse autoencoder, then infers per-city activations with that
shared basis. This makes basis #k mean the same thing in every city, so the
map colouring, feature space and transition matrices are genuinely cross-city
comparable (unlike per-city bases).

Cross-city pooling has no shared road graph, so the spatial-smoothness term is
disabled here (lambda_spatial=0); the basis is learned from reconstruction +
sparsity + diversity. Per-city inference is a plain encoder pass.

  python -m scripts.pipeline.run_stage45_joint --K 32 --epochs 50
"""
import scripts._env  # noqa: F401
import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from scripts.core.io_utils import save_report
from scripts.pipeline.stage4_train_road_basis_model import train_basis_model
from scripts.pipeline.stage5_infer_road_basis_activation import infer_activation

CITY_OUT = {
    "HongKong":  "outputs/China/HongKong",
    "Singapore": "outputs/Singapore/Singapore",
    "Amsterdam": "outputs/Netherlands/Amsterdam",
    "CapeTown":  "outputs/SouthAfrica/CapeTown",
}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cities", nargs="+", default=list(CITY_OUT))
    ap.add_argument("--K", type=int, default=32)
    ap.add_argument("--hidden", type=int, default=512)
    ap.add_argument("--epochs", type=int, default=50)
    ap.add_argument("--per-city-cap", type=int, default=120000,
                    help="max covered nodes pooled per city (balanced training set)")
    ap.add_argument("--joint-dir", default="outputs/transfer/joint")
    args = ap.parse_args()

    # ---- 1. pool covered nodes, balanced per city ----
    parts = []
    for c in args.cities:
        ctx = pd.read_parquet(Path(CITY_OUT[c]) / "road_context_features.parquet")
        cov = ctx[ctx["n_panos"] > 0].copy()
        if len(cov) > args.per_city_cap:
            cov = cov.sample(args.per_city_cap, random_state=0)
        cov["road_node_id"] = c + "_" + cov["road_node_id"].astype(str)   # globally unique
        parts.append(cov)
        print(f"[joint] {c}: pooled {len(cov):,} covered nodes", flush=True)
    pooled = pd.concat(parts, ignore_index=True)
    D = len(pooled["road_context_embedding"].iloc[0])
    print(f"[joint] total pooled = {len(pooled):,} nodes, D={D}", flush=True)

    # ---- 2. train ONE basis (no cross-city spatial term) ----
    empty_edges = pd.DataFrame({"src_node_id": [], "dst_node_id": []})
    jd = Path(args.joint_dir); jd.mkdir(parents=True, exist_ok=True)
    model, r4 = train_basis_model(
        pooled, empty_edges, K=args.K, hidden=args.hidden, epochs=args.epochs,
        lambda_sparse=5e-3, lambda_spatial=0.0, lambda_div=1e-3, output_dir=jd)
    basis = model.basis.detach().cpu().numpy()
    np.save(jd / "road_landscape_basis.npy", basis)
    torch.save(model.state_dict(), jd / "road_basis_model.pt")
    save_report(jd / "stage4_joint_report.json", r4)
    print(f"[joint] basis trained: loss {r4['initial_train_loss']:.4f} -> "
          f"{r4['final_train_loss']:.4f}", flush=True)

    # ---- 3. infer per-city activations with the shared basis ----
    for c in args.cities:
        out = Path(CITY_OUT[c])
        (out / "stage_reports").mkdir(parents=True, exist_ok=True)
        ctx = pd.read_parquet(out / "road_context_features.parquet")
        act, r5 = infer_activation(ctx, model)
        act.to_parquet(out / "road_basis_activation.parquet", index=False)
        np.save(out / "road_landscape_basis.npy", basis)   # shared joint basis
        save_report(out / "stage_reports/stage5_report.json", r5)
        print(f"[joint] {c}: inferred {len(act):,} nodes, "
              f"median_active={r5['median_active_basis']:.1f}, "
              f"recon={r5['mean_recon_error']:.4f}", flush=True)
    print(f"[joint] DONE — shared basis -> {jd}/road_landscape_basis.npy", flush=True)


if __name__ == "__main__":
    main()
