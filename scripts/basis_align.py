"""Cross-city basis alignment (S3 / H1+H4).

If a basis learned on Vienna alone matches one learned on Hong Kong alone, the
visual bases are not coincidental — they are a shared, transferable vocabulary.
Aligns basis matrices by Hungarian matching on cosine similarity and compares to
a random-basis null.

  python -m scripts.basis_align --vienna outputs/transfer/vienna \
      --hongkong outputs/transfer/hongkong --joint outputs/transfer/joint

Output: outputs/figures/basis_alignment.png + outputs/sweep/basis_alignment.csv
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
from scipy.optimize import linear_sum_assignment

FIG = Path("outputs/figures"); FIG.mkdir(parents=True, exist_ok=True)


def _load(d):
    X = np.load(Path(d) / "road_landscape_basis.npy").astype(np.float64)
    if X.shape[0] > X.shape[1]:          # ensure (K, D)
        X = X.T
    return X / (np.linalg.norm(X, axis=1, keepdims=True) + 1e-12)


def align(Xa, Xb):
    """Hungarian matching on cosine; return matched cosines (sorted desc) + score."""
    S = Xa @ Xb.T                        # (K, K) cosine
    r, c = linear_sum_assignment(-S)
    matched = np.sort(S[r, c])[::-1]
    return matched, float(matched.mean()), S, (r, c)


def null_score(K, D, Xb, n=50, seed=42):
    rng = np.random.default_rng(seed)
    scores = []
    for _ in range(n):
        R = rng.standard_normal((K, D)); R /= np.linalg.norm(R, axis=1, keepdims=True)
        scores.append(align(R, Xb)[1])
    return np.array(scores)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--vienna", required=True)
    ap.add_argument("--hongkong", required=True)
    ap.add_argument("--joint", default=None)
    args = ap.parse_args()

    Xv, Xh = _load(args.vienna), _load(args.hongkong)
    K, D = Xv.shape
    matched_vh, s_vh, S_vh, (r, c) = align(Xv, Xh)
    null = null_score(K, D, Xh)
    z = (s_vh - null.mean()) / (null.std() + 1e-12)

    rows = [dict(pair="Vienna↔HongKong", align=round(s_vh, 4),
                 null_mean=round(float(null.mean()), 4), z=round(z, 1))]
    if args.joint:
        Xj = _load(args.joint)
        for nm, Xx in [("Vienna↔joint", Xv), ("HongKong↔joint", Xh)]:
            m, s, _, _ = align(Xx, Xj)
            rows.append(dict(pair=nm, align=round(s, 4),
                             null_mean=round(float(null.mean()), 4),
                             z=round((s - null.mean()) / (null.std() + 1e-12), 1)))
    df = pd.DataFrame(rows)
    df.to_csv("outputs/sweep/basis_alignment.csv", index=False)
    print(df.to_string(index=False))

    # ── figures ──────────────────────────────────────────────────────────────
    fig, ax = plt.subplots(1, 3, figsize=(16, 5))

    # (a) cosine-similarity matrix reordered so matches lie on the diagonal
    order = c[np.argsort(r)]
    im = ax[0].imshow((Xv @ Xh[order].T), cmap="RdBu_r", vmin=-1, vmax=1)
    ax[0].set_title("(a) Vienna vs HongKong basis cosine\n(reordered to diagonal)")
    ax[0].set_xlabel("HongKong basis (matched)"); ax[0].set_ylabel("Vienna basis")
    fig.colorbar(im, ax=ax[0], fraction=0.046, label="cosine")

    # (b) matched cosine per pair, sorted, vs null band
    ax[1].plot(matched_vh, "o-", color="#1f77b4", label="matched V↔HK")
    ax[1].axhspan(null.mean() - null.std(), null.mean() + null.std(),
                  color="gray", alpha=0.3, label="random null ±1σ")
    ax[1].axhline(null.mean(), color="gray", ls="--")
    ax[1].set_title(f"(b) matched basis cosine (mean={s_vh:.2f}, z={z:.0f})")
    ax[1].set_xlabel("matched basis rank"); ax[1].set_ylabel("cosine"); ax[1].legend()
    ax[1].grid(True, alpha=0.3)

    # (c) alignment score vs null (bars)
    ax[2].bar(df["pair"], df["align"], color="#1f77b4", alpha=0.8, label="aligned")
    ax[2].axhline(null.mean(), color="gray", ls="--", label="random null")
    ax[2].set_title("(c) alignment score vs random null")
    ax[2].set_ylabel("mean matched cosine")
    ax[2].tick_params(axis="x", rotation=20); ax[2].legend()

    fig.suptitle("Cross-city basis alignment — independently learned bases match",
                 fontsize=14)
    fig.tight_layout(rect=[0, 0, 1, 0.95])
    p = FIG / "basis_alignment.png"
    fig.savefig(p, dpi=130, bbox_inches="tight"); plt.close(fig)
    print("saved", p)


if __name__ == "__main__":
    main()
