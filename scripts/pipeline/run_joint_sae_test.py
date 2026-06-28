"""TEST a different joint-SAE K without touching production outputs.

Trains a joint basis at --K on the pooled covered nodes (same pooling as
run_stage45_joint), evaluates reconstruction error + sparsity on a held-out
sample, and compares against the existing production K=32 basis on the SAME
sample. Writes the test basis to outputs/transfer/joint_k<K>/ only — does NOT
overwrite any city's road_basis_activation.parquet.

  python -m scripts.pipeline.run_joint_sae_test --K 128 --epochs 50
"""
import scripts._env  # noqa: F401
import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from scripts.core.io_utils import save_report
from scripts.core.road_basis_model import RoadBasisAutoEncoder
from scripts.pipeline.stage4_train_road_basis_model import train_basis_model
from scripts.pipeline.stage5_infer_road_basis_activation import infer_activation

CITY_OUT = {
    "HongKong":  "outputs/China/HongKong",
    "Singapore": "outputs/Singapore/Singapore",
    "Amsterdam": "outputs/Netherlands/Amsterdam",
    "CapeTown":  "outputs/SouthAfrica/CapeTown",
}


def eval_metrics(model, df_eval):
    """recon error + sparsity of a model on an eval DataFrame."""
    act, r5 = infer_activation(df_eval, model)
    return {"mean_recon_error": float(r5["mean_recon_error"]),
            "median_active": float(r5["median_active_basis"])}


def effective_bases(basis):
    Xn = basis / (np.linalg.norm(basis, axis=1, keepdims=True) + 1e-9)
    w = np.linalg.eigvalsh(Xn @ Xn.T); w = np.clip(w, 0, None)
    return float((w.sum() ** 2) / (np.sum(w ** 2) + 1e-12))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--K", type=int, default=128)
    ap.add_argument("--hidden", type=int, default=512)
    ap.add_argument("--epochs", type=int, default=50)
    ap.add_argument("--per-city-cap", type=int, default=120000)
    ap.add_argument("--eval-n", type=int, default=100000)
    args = ap.parse_args()

    # ---- pool covered nodes (same as run_stage45_joint) ----
    parts = []
    for c in CITY_OUT:
        ctx = pd.read_parquet(Path(CITY_OUT[c]) / "road_context_features.parquet")
        cov = ctx[ctx["n_panos"] > 0].copy()
        if len(cov) > args.per_city_cap:
            cov = cov.sample(args.per_city_cap, random_state=0)
        cov["road_node_id"] = c + "_" + cov["road_node_id"].astype(str)
        parts.append(cov)
    pooled = pd.concat(parts, ignore_index=True)
    D = len(pooled["road_context_embedding"].iloc[0])
    eval_df = pooled.sample(min(args.eval_n, len(pooled)), random_state=7).reset_index(drop=True)
    print(f"[test] pooled={len(pooled):,} D={D}  eval={len(eval_df):,}", flush=True)

    # ---- baseline: production K=32 basis on the eval sample ----
    base = {}
    prod = Path("outputs/transfer/joint")
    try:
        b32 = np.load(prod / "road_landscape_basis.npy")
        m32 = RoadBasisAutoEncoder(D=b32.shape[1], K=b32.shape[0], hidden=args.hidden)
        m32.load_state_dict(torch.load(prod / "road_basis_model.pt", map_location="cpu"))
        m32.eval()
        base = eval_metrics(m32, eval_df)
        base["effective_bases"] = effective_bases(b32); base["K"] = b32.shape[0]
        print(f"[test] K={b32.shape[0]} (production): recon={base['mean_recon_error']:.4f} "
              f"median_active={base['median_active']:.1f} eff_bases={base['effective_bases']:.1f}", flush=True)
    except Exception as e:
        print(f"[test] could not eval production K=32: {e}", flush=True)

    # ---- train + eval the new K ----
    empty_edges = pd.DataFrame({"src_node_id": [], "dst_node_id": []})
    jd = Path(f"outputs/transfer/joint_k{args.K}"); jd.mkdir(parents=True, exist_ok=True)
    model, r4 = train_basis_model(pooled, empty_edges, K=args.K, hidden=args.hidden,
                                  epochs=args.epochs, lambda_sparse=5e-3, lambda_spatial=0.0,
                                  lambda_div=1e-3, output_dir=jd)
    basis = model.basis.detach().cpu().numpy()
    np.save(jd / "road_landscape_basis.npy", basis)
    torch.save(model.state_dict(), jd / "road_basis_model.pt")
    save_report(jd / "stage4_report.json", r4)
    new = eval_metrics(model, eval_df)
    new["effective_bases"] = effective_bases(basis); new["K"] = args.K
    print(f"[test] K={args.K}: recon={new['mean_recon_error']:.4f} "
          f"median_active={new['median_active']:.1f} eff_bases={new['effective_bases']:.1f} "
          f"train_loss {r4['initial_train_loss']:.4f}->{r4['final_train_loss']:.4f}", flush=True)

    # ---- comparison ----
    print("\n==== SAE K comparison (same eval sample) ====", flush=True)
    print(f"{'K':>5} {'recon_err':>10} {'median_active':>14} {'eff_bases':>10}", flush=True)
    for r in ([base] if base else []) + [new]:
        print(f"{r['K']:>5} {r['mean_recon_error']:>10.4f} {r['median_active']:>14.1f} {r['effective_bases']:>10.1f}", flush=True)
    save_report(jd / "k_comparison.json", {"baseline": base, "test": new})


if __name__ == "__main__":
    main()
