"""Visualise MRLU experiment results for Vienna and Hong Kong.

Outputs PNGs into outputs/figures/.
"""
from __future__ import annotations

from pathlib import Path

import geopandas as gpd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

CITIES = [
    ("Vienna",   "outputs/Austria/Vienna"),
    ("HongKong", "outputs/China/HongKong"),
]
FIG = Path("outputs/figures")
FIG.mkdir(parents=True, exist_ok=True)


def _load(out):
    out = Path(out)
    units = gpd.read_file(out / "minimum_road_landscape_units.geojson")
    stats = pd.read_csv(out / "unit_statistics.csv")
    return units, stats


def map_units(city, out):
    """Spatial map of MRLU units coloured by dominant basis."""
    units, _ = _load(out)
    fig, ax = plt.subplots(figsize=(11, 11))
    # colour by dominant_basis_id (categorical, tab20 repeated)
    ncol = 32
    cmap = plt.get_cmap("tab20")
    colors = [cmap(i % 20) for i in range(ncol)]
    for bid, grp in units.groupby("dominant_basis_id"):
        grp.plot(ax=ax, color=colors[int(bid) % ncol], linewidth=0.8)
    ax.set_title(f"{city} — Minimum Road Landscape Units (n={len(units)})\n"
                 f"coloured by dominant visual basis (K=32)", fontsize=13)
    ax.set_xlabel("longitude"); ax.set_ylabel("latitude")
    ax.set_aspect("equal")
    ax.grid(True, alpha=0.2)
    p = FIG / f"map_units_{city}.png"
    fig.savefig(p, dpi=130, bbox_inches="tight")
    plt.close(fig)
    return p


def map_units_by_length(city, out):
    """Spatial map coloured by unit road length (continuous)."""
    units, _ = _load(out)
    fig, ax = plt.subplots(figsize=(11, 11))
    units.plot(ax=ax, column="road_length_m", cmap="viridis",
               linewidth=0.9, legend=True,
               legend_kwds={"label": "unit road length (m)", "shrink": 0.5},
               norm=matplotlib.colors.LogNorm(
                   vmin=max(units["road_length_m"].min(), 1),
                   vmax=units["road_length_m"].max()))
    ax.set_title(f"{city} — MRLU by road length (log scale)", fontsize=13)
    ax.set_xlabel("longitude"); ax.set_ylabel("latitude")
    ax.set_aspect("equal")
    ax.grid(True, alpha=0.2)
    p = FIG / f"map_length_{city}.png"
    fig.savefig(p, dpi=130, bbox_inches="tight")
    plt.close(fig)
    return p


def comparison_charts():
    """Side-by-side distributions for both cities."""
    data = {c: _load(o)[1] for c, o in CITIES}
    colors = {"Vienna": "#1f77b4", "HongKong": "#d62728"}

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    # (a) unit count
    ax = axes[0, 0]
    counts = {c: len(s) for c, s in data.items()}
    ax.bar(counts.keys(), counts.values(),
           color=[colors[c] for c in counts])
    for i, (c, v) in enumerate(counts.items()):
        ax.text(i, v, str(v), ha="center", va="bottom", fontsize=12)
    ax.set_title("(a) Number of MRLU units"); ax.set_ylabel("units")

    # (b) road length distribution (log)
    ax = axes[0, 1]
    for c, s in data.items():
        ax.hist(s["road_length_m"], bins=np.logspace(1.6, 4.6, 40),
                alpha=0.55, label=c, color=colors[c])
    ax.set_xscale("log")
    ax.set_title("(b) Unit road length distribution")
    ax.set_xlabel("road length (m, log)"); ax.set_ylabel("# units"); ax.legend()

    # (c) nodes per unit
    ax = axes[1, 0]
    for c, s in data.items():
        ax.hist(s["n_road_nodes"], bins=np.logspace(0.4, 3.2, 40),
                alpha=0.55, label=c, color=colors[c])
    ax.set_xscale("log")
    ax.set_title("(c) Road nodes per unit")
    ax.set_xlabel("# road nodes (log)"); ax.set_ylabel("# units"); ax.legend()

    # (d) dominant basis usage
    ax = axes[1, 1]
    width = 0.4
    x = np.arange(32)
    for i, (c, s) in enumerate(data.items()):
        vc = s["dominant_basis_id"].value_counts().reindex(range(32), fill_value=0)
        ax.bar(x + (i - 0.5) * width, vc.values, width,
               label=c, color=colors[c], alpha=0.8)
    ax.set_title("(d) Units per dominant basis")
    ax.set_xlabel("basis id (0–31)"); ax.set_ylabel("# units"); ax.legend()

    fig.suptitle("MRLU experiment — Vienna vs Hong Kong (500 panos each)",
                 fontsize=15)
    fig.tight_layout(rect=[0, 0, 1, 0.97])
    p = FIG / "comparison_charts.png"
    fig.savefig(p, dpi=130, bbox_inches="tight")
    plt.close(fig)
    return p


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--task", default="all",
                    help="all | maps:<City> | comparison")
    args = ap.parse_args()
    cmap = dict(CITIES)

    produced = []
    if args.task.startswith("maps:"):
        city = args.task.split(":", 1)[1]
        produced.append(map_units(city, cmap[city]))
        produced.append(map_units_by_length(city, cmap[city]))
    elif args.task == "comparison":
        produced.append(comparison_charts())
    else:
        for city, out in CITIES:
            produced.append(map_units(city, out))
            produced.append(map_units_by_length(city, out))
        produced.append(comparison_charts())

    print("Saved figures:")
    for p in produced:
        print("  ", p)
