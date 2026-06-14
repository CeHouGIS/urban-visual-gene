"""Plot the P0-2 sampling-sensitivity sweep (outputs/sweep/summary.csv).

Four panels vs sample size N: coverage, real-signal edge fraction, unit counts
(full vs covered-only), and the sampling-invariant covered-unit density.

  python -m scripts.plot_sweep

Output: outputs/figures/sampling_sweep.png
"""
from __future__ import annotations

import scripts._env  # noqa: F401

from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

FIG = Path("outputs/figures"); FIG.mkdir(parents=True, exist_ok=True)
COLORS = {"Vienna": "#1f77b4", "HongKong": "#d62728"}


def main():
    df = pd.read_csv("outputs/sweep/summary.csv").sort_values(["city", "n_panos"])

    fig, ax = plt.subplots(2, 2, figsize=(13, 9.5))

    def plot(a, ycol, title, ylabel, pct=False, log=False):
        for city, g in df.groupby("city"):
            y = g[ycol] * 100 if pct else g[ycol]
            a.plot(g["n_panos"], y, "o-", color=COLORS.get(city, "k"),
                   label=city, lw=2, ms=7)
        a.set_title(title); a.set_xlabel("pano sample size N"); a.set_ylabel(ylabel)
        a.set_xscale("log"); a.set_xticks(sorted(df["n_panos"].unique()))
        a.get_xaxis().set_major_formatter(matplotlib.ticker.ScalarFormatter())
        if log:
            a.set_yscale("log")
        a.grid(True, alpha=0.3); a.legend()

    plot(ax[0, 0], "coverage_ratio",
         "(a) Road-node coverage by direct panos", "coverage (%)", pct=True)
    plot(ax[0, 1], "full_frac_pos_edges",
         "(b) Real-signal edges (full graph)\n(rest are interpolation, d_ij≈0)",
         "edges with d_ij>0 (%)", pct=True)
    # (c) unit counts: full vs covered
    a = ax[1, 0]
    for city, g in df.groupby("city"):
        c = COLORS.get(city, "k")
        a.plot(g["n_panos"], g["full_units"], "o-", color=c, lw=2, ms=7,
               label=f"{city} — full graph")
        a.plot(g["n_panos"], g["covered_units"], "s--", color=c, lw=2, ms=6,
               alpha=0.6, label=f"{city} — covered-only")
    a.set_title("(c) MRLU count vs N"); a.set_xlabel("pano sample size N")
    a.set_ylabel("# units"); a.set_xscale("log"); a.set_yscale("log")
    a.set_xticks(sorted(df["n_panos"].unique()))
    a.get_xaxis().set_major_formatter(matplotlib.ticker.ScalarFormatter())
    a.grid(True, alpha=0.3); a.legend(fontsize=8)

    plot(ax[1, 1], "covered_unit_density",
         "(d) Covered-unit density (units / covered node)\nsampling-invariant",
         "units per covered node")

    fig.suptitle("P0-2 sampling-sensitivity sweep — Vienna vs Hong Kong",
                 fontsize=15)
    fig.tight_layout(rect=[0, 0, 1, 0.96])
    p = FIG / "sampling_sweep.png"
    fig.savefig(p, dpi=130, bbox_inches="tight")
    plt.close(fig)
    print("saved", p)


if __name__ == "__main__":
    main()
