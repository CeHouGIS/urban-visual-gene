"""Are the learned visual bases similar to each other? (quantitative, image-free)

Measures the structure of the basis set X (K×D) directly:
  (a) pairwise cosine X X^T  -> are bases near-orthogonal (distinct) or redundant?
  (b) hierarchical clustering on 1-cos -> which bases group together
  (c) activation co-occurrence corr(A) -> which bases fire on the same nodes
Reports mean/max off-diagonal cosine and the effective number of bases
(participation ratio of X's singular values).

  python -m scripts.analysis.basis_similarity --basis outputs/transfer/joint \
      --act outputs/sweep/Vienna_N2000

Output: outputs/figures/basis_similarity.png + outputs/sweep/basis_similarity.csv
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
from scipy.cluster.hierarchy import dendrogram, linkage
from scipy.spatial.distance import squareform

FIG = Path("outputs/figures"); FIG.mkdir(parents=True, exist_ok=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--basis", required=True, help="dir with road_landscape_basis.npy")
    ap.add_argument("--act", default=None, help="dir with road_basis_activation.parquet")
    args = ap.parse_args()

    X = np.load(Path(args.basis) / "road_landscape_basis.npy").astype(np.float64)
    if X.shape[0] > X.shape[1]:
        X = X.T                                   # (K, D)
    K = X.shape[0]
    Xn = X / (np.linalg.norm(X, axis=1, keepdims=True) + 1e-12)
    G = Xn @ Xn.T                                 # (K, K) cosine

    off = G[~np.eye(K, dtype=bool)]
    # effective number of bases: participation ratio of singular values
    sv = np.linalg.svd(X, compute_uv=False)
    pr = float((sv.sum() ** 2) / (np.sum(sv ** 2) + 1e-12))   # 1..K

    stats = dict(K=K, mean_abs_offdiag=round(float(np.abs(off).mean()), 4),
                 max_abs_offdiag=round(float(np.abs(off).max()), 4),
                 frac_offdiag_gt_0p3=round(float((np.abs(off) > 0.3).mean()), 4),
                 effective_n_bases=round(pr, 2))
    print(stats)

    # activation co-occurrence correlation (optional)
    R = None
    if args.act:
        a = pd.read_parquet(Path(args.act) / "road_basis_activation.parquet")
        acols = sorted([c for c in a.columns if c.startswith("a_")])
        A = a[acols].values.astype(np.float64)
        R = np.corrcoef(A.T)
        R = np.nan_to_num(R)

    pd.DataFrame([stats]).to_csv("outputs/sweep/basis_similarity.csv", index=False)

    n_panel = 3 if R is not None else 2
    fig, ax = plt.subplots(1, n_panel, figsize=(5.2 * n_panel, 4.6))

    im = ax[0].imshow(G, cmap="RdBu_r", vmin=-1, vmax=1)
    ax[0].set_title(f"(a) basis cosine $XX^T$\nmean|off-diag|={stats['mean_abs_offdiag']:.2f}, "
                    f"eff. #bases={pr:.1f}/{K}")
    ax[0].set_xlabel("basis"); ax[0].set_ylabel("basis")
    fig.colorbar(im, ax=ax[0], fraction=0.046, label="cosine")

    D = np.clip(1 - G, 0, 2); np.fill_diagonal(D, 0)
    Z = linkage(squareform(D, checks=False), method="average")
    dendrogram(Z, ax=ax[1], color_threshold=0.7, no_labels=False, leaf_font_size=7)
    ax[1].set_title("(b) basis clustering (1-cos)")
    ax[1].set_xlabel("basis"); ax[1].set_ylabel("distance")

    if R is not None:
        im2 = ax[2].imshow(R, cmap="RdBu_r", vmin=-1, vmax=1)
        ax[2].set_title("(c) activation co-occurrence corr(A)")
        ax[2].set_xlabel("basis"); ax[2].set_ylabel("basis")
        fig.colorbar(im2, ax=ax[2], fraction=0.046, label="corr")

    fig.suptitle("Are the learned visual bases similar? (distinct ≈ near-orthogonal)",
                 fontsize=13)
    fig.tight_layout(rect=[0, 0, 1, 0.94])
    p = FIG / "basis_similarity.png"
    fig.savefig(p, dpi=130, bbox_inches="tight"); plt.close(fig)
    print("saved", p)
    print("\nReading: near-zero off-diagonal cosine and effective #bases close to K "
          "= bases are distinct (not redundant). Clusters / high corr blocks = "
          "groups of similar bases.")


if __name__ == "__main__":
    main()
