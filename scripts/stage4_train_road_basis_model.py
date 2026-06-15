"""Stage 4 — Train sparse autoencoder: Z_road ≈ A X.

Exportable:
  train_basis_model(context_features, road_edges, K, hidden, ...) -> (model, report)
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

from scripts.io_utils import checkpoint, save_report, stack_embeddings
from scripts.road_basis_model import (
    RoadBasisAutoEncoder,
    cosine_recon_loss,
    diversity_loss,
    spatial_smoothness_loss,
)


def train_basis_model(
    context_features: pd.DataFrame,
    road_edges: pd.DataFrame,
    K: int = 100,
    hidden: int = 512,
    lambda_sparse: float = 1e-4,
    lambda_spatial: float = 1e-3,
    lambda_div: float = 1e-3,
    lr: float = 1e-3,
    batch_size: int = 1024,
    epochs: int = 100,
    val_frac: float = 0.1,
    seed: int = 42,
    output_dir: Path | str | None = None,
) -> tuple[RoadBasisAutoEncoder, dict[str, Any]]:
    torch.manual_seed(seed)
    np.random.seed(seed)

    # ── Data ──────────────────────────────────────────────────────────────────
    emb_list = context_features["road_context_embedding"].values
    Z = torch.tensor(stack_embeddings(emb_list), dtype=torch.float32)   # (M, D)
    D = Z.shape[1]
    M = Z.shape[0]

    # Node index map for spatial loss
    node_ids = context_features["road_node_id"].tolist()
    nid2idx  = {n: i for i, n in enumerate(node_ids)}

    # Build edge index tensors (only edges where both endpoints are in Z)
    src_list, dst_list = [], []
    for _, row in road_edges.iterrows():
        si = nid2idx.get(row["src_node_id"])
        di = nid2idx.get(row["dst_node_id"])
        if si is not None and di is not None:
            src_list.append(si)
            dst_list.append(di)
    edge_src = torch.tensor(src_list, dtype=torch.long)
    edge_dst = torch.tensor(dst_list, dtype=torch.long)

    # Train / val split
    perm   = torch.randperm(M)
    n_val  = max(1, int(M * val_frac))
    val_idx, train_idx = perm[:n_val], perm[n_val:]

    train_ds = TensorDataset(Z[train_idx], train_idx)
    val_ds   = TensorDataset(Z[val_idx],   val_idx)
    train_dl = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    val_dl   = DataLoader(val_ds,   batch_size=batch_size)

    # ── Model ─────────────────────────────────────────────────────────────────
    model = RoadBasisAutoEncoder(D=D, K=K, hidden=hidden)
    opt   = torch.optim.Adam(model.parameters(), lr=lr)

    history: list[dict] = []
    initial_loss: float | None = None

    for epoch in range(epochs):
        model.train()
        train_loss = 0.0
        ep = {"recon": 0.0, "sparse": 0.0, "spatial": 0.0, "div": 0.0}
        for z_batch, idx_batch in train_dl:
            opt.zero_grad()
            a, z_hat = model(z_batch)

            l_recon  = cosine_recon_loss(z_batch, z_hat)
            l_sparse = lambda_sparse * a.abs().mean()
            l_div    = lambda_div    * diversity_loss(model)

            # spatial loss: only use edges within this batch.
            # Work on the plain Python int lists (src_list/dst_list) — never
            # iterate over torch tensors element-wise (slow + triggers torch
            # deploy interpreter errors). g2l.get() folds the batch-membership
            # test and the global→batch-local remap into one pass.
            g2l = {g: l for l, g in enumerate(idx_batch.tolist())}
            bs_list, bd_list = [], []
            for s, d in zip(src_list, dst_list):
                ls = g2l.get(s)
                ld = g2l.get(d)
                if ls is not None and ld is not None:
                    bs_list.append(ls)
                    bd_list.append(ld)
            if bs_list:
                bs = torch.tensor(bs_list, dtype=torch.long)
                bd = torch.tensor(bd_list, dtype=torch.long)
                l_spatial = lambda_spatial * spatial_smoothness_loss(a, bs, bd)
            else:
                l_spatial = torch.tensor(0.0)

            loss = l_recon + l_sparse + l_div + l_spatial
            loss.backward()
            opt.step()
            # Re-normalise basis after each step
            model._normalise_basis()
            nb = len(z_batch)
            train_loss += loss.item() * nb
            ep["recon"] += l_recon.item() * nb
            ep["sparse"] += float(l_sparse) * nb
            ep["spatial"] += float(l_spatial) * nb
            ep["div"] += float(l_div) * nb

        train_loss /= len(train_idx)
        for kk in ep:
            ep[kk] /= len(train_idx)
        if initial_loss is None:
            initial_loss = train_loss

        # Val
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for z_batch, _ in val_dl:
                a, z_hat = model(z_batch)
                val_loss += cosine_recon_loss(z_batch, z_hat).item() * len(z_batch)
        val_loss /= max(len(val_idx), 1)
        history.append({"epoch": epoch, "train_loss": train_loss,
                        "val_loss": val_loss, **ep})

    final_loss = history[-1]["train_loss"]

    # ── Checkpoints ───────────────────────────────────────────────────────────
    # A nonneg (Softplus guarantees this, but verify on a batch)
    model.eval()
    with torch.no_grad():
        a_sample, _ = model(Z[:min(256, M)])
    checkpoint(float(a_sample.min()) >= -1e-6,
               f"A has negative values: min={float(a_sample.min()):.4f}")

    # X normalised
    x_norms = model.basis.norm(dim=1)
    checkpoint(torch.allclose(x_norms, torch.ones_like(x_norms), atol=1e-3),
               f"X rows not unit-normed: max dev={float((x_norms-1).abs().max()):.4f}")

    # Loss decreased
    checkpoint(initial_loss is None or final_loss < initial_loss * 0.99,
               f"loss did not decrease: {initial_loss:.4f} → {final_loss:.4f}")

    report = {
        "D": D, "K": K, "hidden": hidden,
        "epochs": epochs, "n_train": int(len(train_idx)), "n_val": int(len(val_idx)),
        "initial_train_loss": round(initial_loss or 0, 6),
        "final_train_loss":   round(final_loss, 6),
        "final_val_loss":     round(history[-1]["val_loss"], 6),
        "lambda_sparse": lambda_sparse,
        "lambda_spatial": lambda_spatial,
        "lambda_div": lambda_div,
        "history": [{k: round(v, 6) for k, v in h.items()} for h in history],
    }

    if output_dir is not None:
        out = Path(output_dir)
        out.mkdir(parents=True, exist_ok=True)
        torch.save(model.state_dict(), out / "road_basis_model.pt")
        np.save(out / "road_landscape_basis.npy",
                model.basis.detach().cpu().numpy())
        save_report(out / "training_report.json", report)

    return model, report


if __name__ == "__main__":
    import argparse

    ap = argparse.ArgumentParser()
    ap.add_argument("--input",         required=True)
    ap.add_argument("--road-edges",    required=True)
    ap.add_argument("--output-model",  default="models/road_basis_model.pt")
    ap.add_argument("--output-basis",  default="models/road_landscape_basis.npy")
    ap.add_argument("--K",             type=int,   default=100)
    ap.add_argument("--hidden",        type=int,   default=512)
    ap.add_argument("--epochs",        type=int,   default=100)
    ap.add_argument("--lambda-sparse", type=float, default=1e-4)
    ap.add_argument("--lambda-spatial",type=float, default=1e-3)
    ap.add_argument("--lambda-div",    type=float, default=1e-3)
    args = ap.parse_args()

    ctx   = pd.read_parquet(args.input)
    edges = pd.read_parquet(args.road_edges)
    out_dir = Path(args.output_model).parent

    model, report = train_basis_model(
        ctx, edges,
        K=args.K, hidden=args.hidden, epochs=args.epochs,
        lambda_sparse=args.lambda_sparse,
        lambda_spatial=args.lambda_spatial,
        lambda_div=args.lambda_div,
        output_dir=out_dir,
    )
    print(json.dumps(report, indent=2))
