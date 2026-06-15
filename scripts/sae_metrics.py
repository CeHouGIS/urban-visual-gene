"""SAE training/quality dashboard (deep-learning metrics).

Trains the sparse autoencoder capturing the full loss history, infers
activations, and plots: learning curves, loss decomposition, reconstruction
error distribution, sparsity (active bases), activation entropy, and the
reconstruction-vs-K curve.

  python -m scripts.sae_metrics --out outputs/sweep/Vienna_N2000_clean --K 32 --epochs 50

Output: outputs/figures/sae_metrics.png + outputs/sweep/sae_metrics.csv
"""
from __future__ import annotations

import scripts._env  # noqa: F401

import argparse
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from scripts.stage4_train_road_basis_model import train_basis_model
from scripts.stage5_infer_road_basis_activation import infer_activation

FIG = Path("outputs/figures"); FIG.mkdir(parents=True, exist_ok=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--K", type=int, default=32)
    ap.add_argument("--epochs", type=int, default=50)
    args = ap.parse_args()
    out = Path(args.out)

    ctx = pd.read_parquet(out / "road_context_features.parquet")
    ctx = ctx[ctx["n_panos"] > 0].reset_index(drop=True)
    edges = pd.read_parquet(out / "road_graph_edges.parquet")

    model, r = train_basis_model(ctx, edges, K=args.K, hidden=512,
                                 epochs=args.epochs, lambda_sparse=5e-3,
                                 lambda_spatial=1e-3, lambda_div=1e-3)
    hist = pd.DataFrame(r["history"])
    act, r5 = infer_activation(ctx, model)
    rec = act["reconstruction_error"].values
    act_cnt = act["active_basis_count"].values
    ent = act["activation_entropy"].values

    fig, ax = plt.subplots(2, 3, figsize=(16, 9))

    # (a) learning curves
    ax[0, 0].plot(hist.epoch, hist.train_loss, label="train total", color="#1f77b4")
    ax[0, 0].plot(hist.epoch, hist.val_loss, label="val (recon)", color="#ff7f0e")
    ax[0, 0].plot(hist.epoch, hist.recon, "--", label="train recon", color="#2ca02c")
    ax[0, 0].set_title("(a) learning curves"); ax[0, 0].set_xlabel("epoch")
    ax[0, 0].set_ylabel("loss"); ax[0, 0].legend(fontsize=8); ax[0, 0].grid(alpha=.3)

    # (b) loss decomposition (log)
    for c, col in [("recon", "#2ca02c"), ("sparse", "#d62728"),
                   ("spatial", "#9467bd"), ("div", "#8c564b")]:
        ax[0, 1].plot(hist.epoch, hist[c].clip(lower=1e-6), label=c, color=col)
    ax[0, 1].set_yscale("log"); ax[0, 1].set_title("(b) loss terms (log)")
    ax[0, 1].set_xlabel("epoch"); ax[0, 1].legend(fontsize=8); ax[0, 1].grid(alpha=.3)

    # (c) reconstruction error distribution
    ax[0, 2].hist(rec, bins=50, color="#2ca02c", alpha=.8)
    ax[0, 2].axvline(rec.mean(), color="k", ls="--",
                     label=f"mean={rec.mean():.3f}")
    ax[0, 2].set_title("(c) per-node reconstruction error (1-cos)")
    ax[0, 2].set_xlabel("recon error"); ax[0, 2].set_ylabel("# nodes")
    ax[0, 2].legend(fontsize=8)

    # (d) sparsity: active bases per node
    ax[1, 0].hist(act_cnt, bins=range(0, args.K + 2), color="#1f77b4", alpha=.8)
    ax[1, 0].axvline(np.median(act_cnt), color="k", ls="--",
                     label=f"median={int(np.median(act_cnt))}/{args.K}")
    ax[1, 0].set_title("(d) sparsity: active bases / node (L0)")
    ax[1, 0].set_xlabel("# active bases"); ax[1, 0].set_ylabel("# nodes")
    ax[1, 0].legend(fontsize=8)

    # (e) activation entropy
    ax[1, 1].hist(ent, bins=50, color="#9467bd", alpha=.8)
    ax[1, 1].set_title("(e) activation entropy / node")
    ax[1, 1].set_xlabel("entropy"); ax[1, 1].set_ylabel("# nodes")

    # (f) reconstruction vs K (elbow), if available
    ks = Path("outputs/sweep/k_sweep.csv")
    if ks.exists():
        kdf = pd.read_csv(ks).sort_values("K")
        ax[1, 2].plot(kdf.K, kdf.recon_error, "o-", color="#d62728")
        ax[1, 2].set_xscale("log", base=2); ax[1, 2].set_xticks(kdf.K)
        ax[1, 2].get_xaxis().set_major_formatter(matplotlib.ticker.ScalarFormatter())
        ax[1, 2].set_title("(f) reconstruction vs K")
        ax[1, 2].set_xlabel("K"); ax[1, 2].set_ylabel("recon error"); ax[1, 2].grid(alpha=.3)
    else:
        ax[1, 2].axis("off")

    tag = out.name.split("_")[0]          # e.g. Vienna / HongKong
    fig.suptitle(f"SAE metrics — {out.name} (K={args.K}, recon={rec.mean():.3f}, "
                 f"median active={int(np.median(act_cnt))}/{args.K})", fontsize=14)
    fig.tight_layout(rect=[0, 0, 1, 0.96])
    p = FIG / f"sae_metrics_{tag}.png"
    fig.savefig(p, dpi=130, bbox_inches="tight"); plt.close(fig)

    # NB: use bracket indexing — `hist.sparse` collides with pandas' .sparse accessor
    row = pd.DataFrame([{
        "city": tag,
        "final_train_loss": r["final_train_loss"],
        "final_val_loss": r["final_val_loss"],
        "mean_recon_error": round(float(rec.mean()), 4),
        "median_active": int(np.median(act_cnt)),
        "mean_entropy": round(float(ent.mean()), 4),
        "final_recon": hist["recon"].iloc[-1], "final_sparse": hist["sparse"].iloc[-1],
        "final_spatial": hist["spatial"].iloc[-1], "final_div": hist["div"].iloc[-1],
    }])
    csv = Path("outputs/sweep/sae_metrics.csv")
    if csv.exists():
        prev = pd.read_csv(csv)
        prev = prev[prev.get("city", "") != tag] if "city" in prev else prev
        row = pd.concat([prev, row], ignore_index=True)
    row.to_csv(csv, index=False)
    print("saved", p)
    print(f"recon={rec.mean():.3f} median_active={int(np.median(act_cnt))}/{args.K} "
          f"final loss {r['final_train_loss']:.3f}")


if __name__ == "__main__":
    main()
