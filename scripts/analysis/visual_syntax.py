"""Visual syntax: along-road basis transition matrix T (H4).

For each city, b_i = argmax_k a_{i,k} is the dominant visual basis of a covered
road node. T_{kl} = P(b_j=l | b_i=k) over road edges (i,j) measures how visual
bases follow one another along streets — the city's "visual syntax". Cities may
share a vocabulary (which bases) yet differ in syntax (how they are arranged).

Derived: self-transition rate, transition entropy, off-diagonal mass; cross-city
syntax distance (JSD of flattened T). Renders side-by-side T heatmaps.

  python -m scripts.analysis.visual_syntax --dirs outputs/sweep/Vienna_N2000 \
      outputs/sweep/HongKong_N2000 --labels Vienna HongKong

Output: outputs/figures/visual_syntax.png  +  outputs/sweep/visual_syntax.csv
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
from scipy.special import entr

from scripts.analysis.baseline_common import load_city

FIG = Path("outputs/figures"); FIG.mkdir(parents=True, exist_ok=True)


def transition_matrix(A, node_ids, edges):
    K = A.shape[1]
    b = A.argmax(axis=1)
    nid2idx = {n: i for i, n in enumerate(node_ids)}
    si = np.array([nid2idx.get(s, -1) for s in edges["src_node_id"]])
    di = np.array([nid2idx.get(d, -1) for d in edges["dst_node_id"]])
    ok = (si >= 0) & (di >= 0)
    bi, bj = b[si[ok]], b[di[ok]]
    C = np.zeros((K, K), dtype=np.float64)
    # undirected: count both directions
    np.add.at(C, (bi, bj), 1.0)
    np.add.at(C, (bj, bi), 1.0)
    row = C.sum(axis=1, keepdims=True)
    T = C / np.clip(row, 1, None)
    self_tr = float(np.trace(C) / max(C.sum(), 1))            # same basis both ends
    ent = float(np.mean([entr(T[k]).sum() for k in range(K) if row[k, 0] > 0]))
    usage = np.bincount(b, minlength=K) / len(b)
    return T, dict(self_transition=round(self_tr, 4),
                   transition_entropy=round(ent, 4),
                   off_diagonal_mass=round(1 - self_tr, 4),
                   n_covered=len(b)), usage


def jsd(p, q, eps=1e-12):
    p = p / p.sum(); q = q / q.sum(); m = 0.5 * (p + q)
    kl = lambda a, b: float(np.sum(a * np.log((a + eps) / (b + eps))))
    return 0.5 * kl(p, m) + 0.5 * kl(q, m)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dirs", nargs="+", required=True)
    ap.add_argument("--labels", nargs="+", required=True)
    args = ap.parse_args()

    Ts, usages, rows = {}, {}, []
    for d, lab in zip(args.dirs, args.labels):
        node_ids, Z, A, lat, lon, edges, n_panos = load_city(d, covered_only=True)
        T, m, usage = transition_matrix(A, node_ids, edges)
        Ts[lab] = T; usages[lab] = usage
        rows.append({"city": lab, **m})
        print(f"{lab}: self_transition={m['self_transition']} "
              f"entropy={m['transition_entropy']} "
              f"off_diag={m['off_diagonal_mass']} n={m['n_covered']}", flush=True)

    df = pd.DataFrame(rows)
    if len(args.labels) == 2:
        a, b = args.labels
        df.loc[0, "vocab_JSD_vs_other"] = round(jsd(usages[a], usages[b]), 4)
    df.to_csv("outputs/sweep/visual_syntax.csv", index=False)

    n = len(args.labels)
    fig, axes = plt.subplots(1, n, figsize=(5.2 * n, 5))
    axes = np.atleast_1d(axes)
    for ax, lab in zip(axes, args.labels):
        im = ax.imshow(Ts[lab], cmap="magma", vmin=0, vmax=min(1, Ts[lab].max()))
        ax.set_title(f"{lab}\nself-tr={df[df.city==lab].self_transition.iloc[0]:.2f}, "
                     f"H={df[df.city==lab].transition_entropy.iloc[0]:.2f}")
        ax.set_xlabel("next basis $l$"); ax.set_ylabel("basis $k$")
        fig.colorbar(im, ax=ax, fraction=0.046, label="$T_{kl}=P(b_j=l|b_i=k)$")
    fig.suptitle("Visual syntax — along-road basis transition matrix $T$", fontsize=14)
    fig.tight_layout(rect=[0, 0, 1, 0.95])
    p = FIG / "visual_syntax.png"
    fig.savefig(p, dpi=130, bbox_inches="tight"); plt.close(fig)
    print("\n", df.to_string(index=False))
    print("saved", p, "and outputs/sweep/visual_syntax.csv")


if __name__ == "__main__":
    main()
