#!/usr/bin/env python3
"""Show what different pano-pair cosine similarities actually look like: for each
similarity band, pick a 15m-neighbour pair and lay out both panos' 4 views.
Saves outputs/figures/similarity_examples_<City>.png
"""
import os, sys
from pathlib import Path
import numpy as np
from scipy.spatial import cKDTree
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from PIL import Image

ROOT = Path(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
DATA = ROOT / "dashboard" / "data"
FEAT = ROOT / "outputs" / "full_feats"
CC = {"Vienna": "Austria/Vienna", "HongKong": "China/HongKong"}
BANDS = [(0.90, 1.01), (0.85, 0.90), (0.80, 0.85), (0.75, 0.80), (0.70, 0.75), (0.60, 0.70)]


def url(city, pid, h):
    base = ROOT / "data/SVIs/GSV/images" / CC[city]
    return (base / pid[0] / pid[1] / pid / f"{pid}_{h}.jpg" if city == "Vienna"
            else base / pid[0].lower() / pid[1].lower() / pid[2].lower() / f"{pid}_{h}.jpg")


def main():
    city = sys.argv[1] if len(sys.argv) > 1 else "Vienna"
    ids = (DATA / (f"panos_{city}_ids.full.txt" if (DATA / f"panos_{city}_ids.full.txt").exists()
                   else f"panos_{city}_ids.txt")).read_text().split("\n")
    xy = np.fromfile(DATA / (f"panos_{city}_xy.full.bin" if (DATA / f"panos_{city}_xy.full.bin").exists()
                             else f"panos_{city}_xy.bin"), np.float32).reshape(-1, 2).astype(np.float64)
    F = np.load(FEAT / f"{city}_feats.f16.npy").astype(np.float32)
    nrm = np.linalg.norm(F, axis=1); v = nrm > 1e-6; F[v] /= nrm[v, None]

    lat0 = xy[:, 1].mean(); kx = 111320 * np.cos(np.radians(lat0))
    tree = cKDTree(np.column_stack([xy[:, 0] * kx, xy[:, 1] * 111320]))
    pairs = tree.query_pairs(15, output_type="ndarray")
    cos = np.einsum("ij,ij->i", F[pairs[:, 0]], F[pairs[:, 1]])
    dm = np.hypot((xy[pairs[:, 0], 0] - xy[pairs[:, 1], 0]) * kx,
                  (xy[pairs[:, 0], 1] - xy[pairs[:, 1], 1]) * 111320)
    rng = np.random.default_rng(0)

    fig, axes = plt.subplots(len(BANDS), 8, figsize=(15, 2.1 * len(BANDS)))
    fig.patch.set_facecolor("#0b1322")
    for r, (lo, hi) in enumerate(BANDS):
        m = (cos >= lo) & (cos < hi) & (dm > 1)
        for c in range(8):
            axes[r, c].axis("off"); axes[r, c].set_facecolor("#0b1322")
        if not m.any():
            axes[r, 0].text(0, .5, f"[{lo:.2f},{hi:.2f}) 无样本", color="#bbb"); continue
        # pick the pair nearest the band centre
        cand = np.where(m)[0]; pick = cand[np.argmin(np.abs(cos[cand] - (lo + hi) / 2))]
        a, b = pairs[pick]
        for k, (pid, who) in enumerate([(ids[a], "A"), (ids[b], "B")]):
            for j, h in enumerate([0, 90, 180, 270]):
                ax = axes[r, k * 4 + j]
                try:
                    ax.imshow(Image.open(url(city, pid, h)).convert("RGB").resize((160, 160)))
                except Exception:
                    ax.set_facecolor("#222")
                if r == 0: ax.set_title(f"{who} {h}°", color="#9fc6ff", fontsize=9)
        axes[r, 0].axis("on"); axes[r, 0].set_xticks([]); axes[r, 0].set_yticks([])
        axes[r, 0].set_ylabel(f"cos={cos[pick]:.3f}\n{dm[pick]:.0f}m", color="#39d6ff", fontsize=11, rotation=0,
                              ha="right", va="center", labelpad=34)
    fig.suptitle(f"{city} — 15m 邻居 pano 对在不同相似度下的样子 (左 A 4向 | 右 B 4向)",
                 color="#eaf2ff", fontsize=13)
    fig.tight_layout(rect=[0.04, 0, 1, 0.97])
    out = ROOT / "outputs" / "figures" / f"similarity_examples_{city}.png"
    fig.savefig(out, dpi=110, facecolor=fig.get_facecolor()); print("saved", out)


if __name__ == "__main__":
    main()
