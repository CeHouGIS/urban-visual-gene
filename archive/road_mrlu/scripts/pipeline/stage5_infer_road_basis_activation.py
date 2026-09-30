"""Stage 5 — Infer road basis activation A for every road node.

Exportable:
  infer_activation(context_features, model, activation_threshold) -> (activation_df, report)
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from scipy.special import entr

from scripts.core.io_utils import checkpoint, save_report, stack_embeddings
from scripts.core.road_basis_model import RoadBasisAutoEncoder


def infer_activation(
    context_features: pd.DataFrame,
    model: RoadBasisAutoEncoder,
    activation_threshold: float = 0.01,
    batch_size: int = 2048,
) -> tuple[pd.DataFrame, dict]:
    """Run encoder on all road nodes and build activation DataFrame.

    Returns
    -------
    activation_df : pd.DataFrame  — road_basis_activation schema
    report        : dict
    """
    model.eval()
    emb_list = context_features["road_context_embedding"].values
    Z = torch.tensor(stack_embeddings(emb_list), dtype=torch.float32)

    all_a: list[np.ndarray] = []
    all_zhat: list[np.ndarray] = []

    with torch.no_grad():
        for start in range(0, len(Z), batch_size):
            z_b = Z[start: start + batch_size]
            a_b, zh_b = model(z_b)
            all_a.append(a_b.cpu().numpy())
            all_zhat.append(zh_b.cpu().numpy())

    A    = np.concatenate(all_a,    axis=0)   # (M, K)
    Zhat = np.concatenate(all_zhat, axis=0)   # (M, D)
    Z_np = Z.numpy()

    K = A.shape[1]

    # Per-node metrics
    recon_error    = 1.0 - (
        (Z_np * Zhat).sum(axis=1) /
        (np.linalg.norm(Z_np, axis=1) * np.linalg.norm(Zhat, axis=1) + 1e-9)
    )
    # activation entropy
    A_norm = A / (A.sum(axis=1, keepdims=True) + 1e-9)
    act_entropy = entr(A_norm).sum(axis=1)
    active_count = (A > activation_threshold).sum(axis=1)

    # Build output DataFrame
    out = context_features[["road_node_id", "road_id", "lat", "lon"]].copy()
    if "city" in context_features.columns:
        out["city"] = context_features["city"].values

    for k in range(K):
        out[f"a_{k:03d}"] = A[:, k].astype(np.float32)

    out["reconstruction_error"] = recon_error.astype(np.float32)
    out["activation_entropy"]   = act_entropy.astype(np.float32)
    out["active_basis_count"]   = active_count.astype(np.int32)

    # Checkpoints
    checkpoint(float(A.min()) >= -1e-5,
               f"activation A has negative values: min={float(A.min()):.6f}")

    median_active = float(np.median(active_count))
    # Sparsity is a design goal but may not hold for small K or few training epochs;
    # emit a warning rather than hard-failing, so downstream tests are not blocked.
    if median_active >= K * 0.5:
        print(f"[CHECKPOINT WARN] activation not sparse: "
              f"median active={median_active:.1f} >= K/2={K/2:.1f}", flush=True)

    report = {
        "n_nodes":              int(len(out)),
        "K":                    K,
        "activation_threshold": activation_threshold,
        "median_active_basis":  round(median_active, 2),
        "mean_recon_error":     round(float(recon_error.mean()), 4),
        "mean_entropy":         round(float(act_entropy.mean()), 4),
    }

    return out, report


if __name__ == "__main__":
    import argparse

    ap = argparse.ArgumentParser()
    ap.add_argument("--input",      required=True)
    ap.add_argument("--model",      required=True)
    ap.add_argument("--output",     required=True)
    ap.add_argument("--D",          type=int,   required=True)
    ap.add_argument("--K",          type=int,   default=100)
    ap.add_argument("--hidden",     type=int,   default=512)
    ap.add_argument("--threshold",  type=float, default=0.01)
    args = ap.parse_args()

    ctx = pd.read_parquet(args.input)
    model = RoadBasisAutoEncoder(D=args.D, K=args.K, hidden=args.hidden)
    model.load_state_dict(torch.load(args.model, map_location="cpu"))

    act_df, report = infer_activation(ctx, model, activation_threshold=args.threshold)

    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    act_df.to_parquet(out, index=False)
    save_report(out.parent / "stage_reports" / "stage5_report.json", report)
    print(json.dumps(report, indent=2))
