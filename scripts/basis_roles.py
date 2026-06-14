"""Quantify basis roles: dominant vs combination-only vs dead (RQ4).

Per basis k over covered nodes:
  dominant_frac = P(argmax_l a_l = k)         how often it leads a location
  active_frac   = P(a_k > tau)                how often it participates at all
Classification:
  dominant        : dominant_frac >= min_dom (leads somewhere)
  combination-only: active but (almost) never dominant
  dead            : (almost) never active

Also reports the "effective dominant vocabulary" = #bases covering 90% of all
dominance assignments. Compares cities.

  python -m scripts.basis_roles --dirs outputs/sweep/Vienna_N2000 \
      outputs/sweep/HongKong_N2000 --labels Vienna HongKong

Output: outputs/figures/basis_roles.png + outputs/sweep/basis_roles.csv
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

from scripts.baseline_common import load_city

FIG = Path("outputs/figures"); FIG.mkdir(parents=True, exist_ok=True)
TAU = 0.01
MIN_DOM = 0.002          # >=0.2% of nodes -> counts as "dominant somewhere"


def roles(A):
    M, K = A.shape
    dom = A.argmax(axis=1)
    dom_frac = np.bincount(dom, minlength=K) / M
    act_frac = (A > TAU).mean(axis=0)
    cls = np.where(dom_frac >= MIN_DOM, "dominant",
                   np.where(act_frac >= 0.05, "combination", "dead"))
    # effective dominant vocabulary: #bases for 90% of dominance mass
    s = np.sort(dom_frac)[::-1]
    eff90 = int(np.searchsorted(np.cumsum(s), 0.90) + 1)
    return dom_frac, act_frac, cls, eff90


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dirs", nargs="+", required=True)
    ap.add_argument("--labels", nargs="+", required=True)
    args = ap.parse_args()

    data, rows = {}, []
    for d, lab in zip(args.dirs, args.labels):
        node_ids, Z, A, lat, lon, edges, n_panos = load_city(d, covered_only=True)
        df_, af, cls, eff90 = roles(A)
        data[lab] = (df_, af, cls)
        n_dom = int((cls == "dominant").sum())
        n_comb = int((cls == "combination").sum())
        n_dead = int((cls == "dead").sum())
        rows.append(dict(city=lab, K=A.shape[1], dominant=n_dom,
                         combination_only=n_comb, dead=n_dead,
                         eff_dominant_vocab_90=eff90,
                         median_active=int(np.median((A > TAU).sum(1)))))
        print(f"{lab}: dominant={n_dom} combination-only={n_comb} dead={n_dead} "
              f"| eff dominant vocab(90%)={eff90}", flush=True)

    pd.DataFrame(rows).to_csv("outputs/sweep/basis_roles.csv", index=False)

    n = len(args.labels)
    fig, ax = plt.subplots(2, n, figsize=(5.4 * n, 8))
    ax = np.atleast_2d(ax)
    cmap = {"dominant": "#1f77b4", "combination": "#ff7f0e", "dead": "#999999"}
    for j, lab in enumerate(args.labels):
        df_, af, cls = data[lab]
        order = np.argsort(-df_)
        # (top) sorted dominance fraction
        ax[0, j].bar(range(len(df_)), df_[order] * 100,
                     color=[cmap[c] for c in cls[order]])
        ax[0, j].axhline(MIN_DOM * 100, color="k", ls=":", lw=0.8)
        ax[0, j].set_title(f"{lab}: dominance per basis (sorted)\n"
                           f"dominant={int((cls=='dominant').sum())}, "
                           f"comb-only={int((cls=='combination').sum())}, "
                           f"dead={int((cls=='dead').sum())}")
        ax[0, j].set_xlabel("basis (sorted)"); ax[0, j].set_ylabel("dominance %")
        # (bottom) active vs dominant scatter
        for c in ["dominant", "combination", "dead"]:
            m = cls == c
            ax[1, j].scatter(af[m] * 100, df_[m] * 100, c=cmap[c], label=c, s=40)
        ax[1, j].set_xlabel("active frac (%)"); ax[1, j].set_ylabel("dominance frac (%)")
        ax[1, j].set_title(f"{lab}: role map"); ax[1, j].legend(fontsize=8)
        ax[1, j].grid(True, alpha=0.3)

    fig.suptitle("Basis roles — dominant (leads a place) vs combination-only "
                 "(used only in mixtures)", fontsize=14)
    fig.tight_layout(rect=[0, 0, 1, 0.95])
    p = FIG / "basis_roles.png"
    fig.savefig(p, dpi=130, bbox_inches="tight"); plt.close(fig)
    print("\n", pd.DataFrame(rows).to_string(index=False))
    print("saved", p)


if __name__ == "__main__":
    main()
