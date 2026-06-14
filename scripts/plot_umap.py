"""UMAP of the road visual-feature space (Z_road) for Vienna + Hong Kong.

Embeds the 3072-D context features of pano-covered road nodes into 2-D with a
single JOINT UMAP (both cities together, so the layout is comparable), then
renders two panels: coloured by city, and coloured by dominant visual basis.

torch-free (umap / sklearn / numpy / matplotlib) and isolated from the geo/torch
stages — run in its own process.

  python -m scripts.plot_umap [--max-per-city 6000]

Output: outputs/figures/umap_feature_space.png
"""
from __future__ import annotations

import scripts._env  # noqa: F401  (sets thread limits before numpy/numba)

import argparse
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

CITIES = [
    ("Vienna",   "outputs/Austria/Vienna",   "#1f77b4"),
    ("HongKong", "outputs/China/HongKong",   "#d62728"),
]
FIG = Path("outputs/figures")
FIG.mkdir(parents=True, exist_ok=True)
SEED = 42


def load_city(out: str, max_per_city: int):
    """Return (embeddings NxD, dominant_basis N) for pano-covered nodes."""
    out = Path(out)
    ctx = pd.read_parquet(out / "road_context_features.parquet")
    act = pd.read_parquet(out / "road_basis_activation.parquet")
    ctx = ctx[ctx["n_panos"] > 0][["road_node_id", "road_context_embedding"]]

    a_cols = sorted([c for c in act.columns if c.startswith("a_")])
    act = act[["road_node_id", *a_cols]]
    dom = dict(zip(act["road_node_id"], np.asarray(act[a_cols].values).argmax(1)))

    df = ctx[ctx["road_node_id"].isin(dom)]
    if max_per_city and len(df) > max_per_city:
        df = df.sample(max_per_city, random_state=SEED)

    emb = np.stack([np.asarray(e, dtype=np.float32)
                    for e in df["road_context_embedding"].values])
    basis = np.array([dom[n] for n in df["road_node_id"]])
    return emb, basis


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--max-per-city", type=int, default=6000,
                    help="subsample covered nodes per city for UMAP speed")
    ap.add_argument("--n-neighbors", type=int, default=30)
    ap.add_argument("--min-dist", type=float, default=0.1)
    args = ap.parse_args()

    from umap import UMAP

    embs, bases, city_id, names = [], [], [], []
    for i, (city, out, _) in enumerate(CITIES):
        e, b = load_city(out, args.max_per_city)
        embs.append(e); bases.append(b)
        city_id.append(np.full(len(e), i))
        names.append(city)
        print(f"{city}: {len(e)} covered nodes, D={e.shape[1]}", flush=True)

    X = np.vstack(embs)
    basis = np.concatenate(bases)
    cid = np.concatenate(city_id)

    print(f"running joint UMAP on {X.shape[0]} points × {X.shape[1]} dims ...", flush=True)
    Z = UMAP(n_components=2, n_neighbors=args.n_neighbors,
             min_dist=args.min_dist, metric="cosine",
             random_state=SEED).fit_transform(X)

    fig, axes = plt.subplots(1, 2, figsize=(16, 7.5))

    # (a) coloured by city
    ax = axes[0]
    for i, (city, _, color) in enumerate(CITIES):
        m = cid == i
        ax.scatter(Z[m, 0], Z[m, 1], s=4, c=color, alpha=0.45, label=city, linewidths=0)
    ax.set_title("(a) Z_road feature space by city"); ax.legend(markerscale=3)
    ax.set_xlabel("UMAP-1"); ax.set_ylabel("UMAP-2")

    # (b) coloured by dominant basis
    ax = axes[1]
    cmap = plt.get_cmap("tab20")
    sc = ax.scatter(Z[:, 0], Z[:, 1], s=4, c=[cmap(int(b) % 20) for b in basis],
                    alpha=0.5, linewidths=0)
    ax.set_title("(b) coloured by dominant visual basis (K=32)")
    ax.set_xlabel("UMAP-1"); ax.set_ylabel("UMAP-2")

    fig.suptitle(
        f"UMAP of road visual-feature space (Z_road, D={X.shape[1]}; "
        f"joint over both cities, n={X.shape[0]})", fontsize=14)
    fig.tight_layout(rect=[0, 0, 1, 0.96])
    p = FIG / "umap_feature_space.png"
    fig.savefig(p, dpi=130, bbox_inches="tight")
    plt.close(fig)
    print("saved", p, flush=True)


if __name__ == "__main__":
    main()
