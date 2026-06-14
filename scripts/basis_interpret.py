"""Basis interpretability (RQ4): top-activated street-view examples per basis.

For each learned visual basis k, find the road nodes with the highest activation
a_k, map each to its nearest pano, and tile the panos' front images into a grid
(one row per basis). Lets a reader read off what each basis "means".

torch-free; resolves images locally with NAS fallback.

  python -m scripts.basis_interpret --city Vienna --out outputs/sweep/Vienna_N2000 --top 6

Output: outputs/figures/basis_interpret_<city>.png
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
from scipy.spatial import cKDTree

from scripts.cities import img_root, img_root_fallback, path_style
from scripts.stage1_extract_pano_features import _img_path_hk, _img_path_vienna

FIG = Path("outputs/figures"); FIG.mkdir(parents=True, exist_ok=True)


def _resolve(city, pid, heading=0):
    fn = _img_path_hk if path_style(city) == "hongkong" else _img_path_vienna
    p = fn(img_root(city), pid, heading)
    if not p.exists():
        fb = img_root_fallback(city)
        if fb is not None:
            p = fn(fb, pid, heading)
    return p


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--city", required=True, choices=["Vienna", "HongKong"])
    ap.add_argument("--out", required=True)
    ap.add_argument("--top", type=int, default=6, help="examples per basis")
    ap.add_argument("--bases", type=int, default=None, help="limit #bases shown")
    args = ap.parse_args()
    out = Path(args.out)

    from PIL import Image

    act = pd.read_parquet(out / "road_basis_activation.parquet")
    panos = pd.read_parquet(out / "road_matched_panos.parquet")
    a_cols = sorted([c for c in act.columns if c.startswith("a_")])
    K = len(a_cols)
    bases = range(K if args.bases is None else min(args.bases, K))

    # nearest pano for each node (by coordinates)
    ptree = cKDTree(panos[["lat", "lon"]].values)
    node_ll = act[["lat", "lon"]].values
    _, nn = ptree.query(node_ll, k=1)
    node_pano = panos["pano_id"].values[nn]

    A = act[a_cols].values
    fig, axes = plt.subplots(len(list(bases)), args.top,
                             figsize=(args.top * 1.8, len(list(bases)) * 1.8))
    axes = np.atleast_2d(axes)
    for r, k in enumerate(bases):
        order = np.argsort(-A[:, k])
        shown, ci = 0, 0
        seen = set()
        while shown < args.top and ci < len(order):
            pid = node_pano[order[ci]]; ci += 1
            if pid in seen:
                continue
            seen.add(pid)
            ax = axes[r, shown]; ax.axis("off")
            try:
                img = Image.open(_resolve(args.city, pid, 0)).convert("RGB")
                ax.imshow(img.resize((160, 160)))
            except Exception:
                ax.text(0.5, 0.5, "n/a", ha="center", va="center")
            if shown == 0:
                ax.set_ylabel(f"b{k}", rotation=0, labelpad=18,
                              fontsize=9, va="center")
            shown += 1
        for j in range(shown, args.top):
            axes[r, j].axis("off")

    fig.suptitle(f"{args.city} — top-activated street views per visual basis "
                 f"(K={K})", fontsize=13)
    fig.tight_layout(rect=[0, 0, 1, 0.99])
    p = FIG / f"basis_interpret_{args.city}.png"
    fig.savefig(p, dpi=120, bbox_inches="tight")
    plt.close(fig)
    print("saved", p)


if __name__ == "__main__":
    main()
