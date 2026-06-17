#!/usr/bin/env python3
"""Hierarchical clustering of the 32 joint visual bases, two views:
 (A) basis-VECTOR cosine distance — shows they are near-orthogonal (merge near 1.0);
 (B) activation CO-OCCURRENCE correlation — which independent bases fire together.
Saves outputs/figures/basis_hierarchy.png
"""
import os
for _v in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
    os.environ.setdefault(_v, "1")
from pathlib import Path
import numpy as np
import pandas as pd
from scipy.cluster.hierarchy import linkage, dendrogram
from scipy.spatial.distance import squareform
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
ACT = [f"a_{i:03d}" for i in range(32)]
FG = "#dbe4f0"; BG = "#0b1322"


def main():
    X = np.load(ROOT / "outputs/transfer/joint/road_landscape_basis.npy").astype(np.float64)
    X /= (np.linalg.norm(X, axis=1, keepdims=True) + 1e-9)
    S = np.abs(X @ X.T)                          # |cos| between basis vectors
    Dvec = 1.0 - S; np.fill_diagonal(Dvec, 0.0); Dvec = (Dvec + Dvec.T) / 2
    Lvec = linkage(squareform(Dvec, checks=False), method="average")

    # activation co-occurrence (pool both cities' nodes)
    A = np.vstack([pd.read_parquet(ROOT / f"outputs/dedup_bld/{c}/road_basis_activation_joint.parquet",
                                   columns=ACT).to_numpy(np.float32)
                   for c in ("Vienna", "HongKong")])
    var = A.var(0)
    dead = np.where(var < 1e-8)[0]               # bases that almost never activate
    C = np.corrcoef(A.T)                         # 32x32 activation correlation
    C = np.nan_to_num(C)                         # dead bases -> 0 corr (shown grey)
    Dco = 1.0 - C; np.fill_diagonal(Dco, 0.0); Dco = np.clip((Dco + Dco.T) / 2, 0, None)
    Lco = linkage(squareform(Dco, checks=False), method="average")

    fig = plt.figure(figsize=(15, 7), facecolor=BG)
    # Panel A: basis-vector dendrogram
    axA = fig.add_subplot(1, 2, 1); axA.set_facecolor(BG)
    dn = dendrogram(Lvec, labels=[f"#{i}" for i in range(32)], ax=axA,
                    color_threshold=0, above_threshold_color="#6f8bbd")
    axA.set_title("(A) Basis-vector clustering (cosine dist 1-|cos|)\nnear-orthogonal: merge only near ~1.0",
                  color=FG, fontsize=12)
    axA.set_ylabel("distance 1-|cos|", color=FG)
    for s in axA.spines.values(): s.set_color("#33425e")
    axA.tick_params(colors=FG); [t.set_color(FG) for t in axA.get_xticklabels()]

    # Panel B: co-occurrence clustered heatmap + dendrogram
    order = dendrogram(Lco, no_plot=True)["leaves"]
    axB = fig.add_subplot(1, 2, 2); axB.set_facecolor(BG)
    Cr = C[np.ix_(order, order)]
    im = axB.imshow(Cr, cmap="RdBu_r", vmin=-0.5, vmax=0.5)
    axB.set_xticks(range(32)); axB.set_yticks(range(32))
    axB.set_xticklabels([f"{i}" for i in order], fontsize=6, color=FG, rotation=90)
    axB.set_yticklabels([f"{i}" for i in order], fontsize=6, color=FG)
    axB.set_title(f"(B) Activation co-occurrence corr(A), clustered\northogonal in direction, yet co-activating "
                  f"({len(dead)} dead bases -> grey)", color=FG, fontsize=12)
    cb = fig.colorbar(im, ax=axB, fraction=0.046, pad=0.04); cb.ax.tick_params(colors=FG)
    cb.set_label("activation correlation", color=FG)

    fig.suptitle("Relationships among the 32 joint visual bases: near-orthogonal vectors (A) + co-activation grammar (B)",
                 color="#eaf2ff", fontsize=13)
    fig.tight_layout(rect=[0, 0, 1, 0.95])
    out = ROOT / "outputs/figures/basis_hierarchy.png"
    fig.savefig(out, dpi=120, facecolor=BG)
    print("saved", out)
    # report tightest co-occurrence pairs
    np.fill_diagonal(C, 0)
    fl = [(i, j, C[i, j]) for i in range(32) for j in range(i + 1, 32)]
    fl.sort(key=lambda x: -x[2])
    print("共现最强的基对:", [(i, j, round(c, 2)) for i, j, c in fl[:6]])
    print("基向量非对角 |cos| 最大:", round(float(S[~np.eye(32, dtype=bool)].max()), 3))


if __name__ == "__main__":
    main()
