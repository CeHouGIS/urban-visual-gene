"""Re-colour the MRLU spatial maps using a PCA of the units' visual activations.

Each unit's mean K-dim basis activation is projected to 3 PCA components and
mapped to RGB. PCA is fit JOINTLY on both cities so visually similar units get
similar colours across Vienna and Hong Kong.

Two-phase to avoid the native segfault seen when running the heavy unit
extraction for both cities in one process:
  python -m scripts.plot_pca_maps --task extract:Vienna
  python -m scripts.plot_pca_maps --task extract:HongKong
  python -m scripts.plot_pca_maps --task plot

Outputs: outputs/figures/map_units_pca_<City>.png
"""
from __future__ import annotations

import argparse
from pathlib import Path

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
CACHE = FIG / "_pca_cache"
CACHE.mkdir(exist_ok=True)


def build_units(out):
    """Reproduce the run_experiment Stage 6 call, capturing per-unit activations."""
    import geopandas as gpd  # local import (heavy native libs)
    from scripts.stage6_extract_road_units import extract_road_units
    out = Path(out)
    act = pd.read_parquet(out / "road_basis_activation.parquet")
    nodes = pd.read_parquet(out / "road_graph_nodes.parquet")
    edges = pd.read_parquet(out / "road_graph_edges.parquet")
    ctx = pd.read_parquet(out / "road_context_features.parquet")
    npano = ctx.drop_duplicates("road_node_id").set_index("road_node_id")["n_panos"]
    nodes = nodes.copy()
    nodes["n_panos"] = nodes["road_node_id"].map(npano).fillna(0).astype(int)

    _, units_gdf, _, _, acts = extract_road_units(
        act, nodes, edges,
        boundary_quantile=0.90, min_road_nodes=3,
        min_road_length_m=50.0, min_panos=3,
        return_unit_activations=True,
    )
    return units_gdf, acts


def extract_city(city, out):
    """Phase A: extract units + activations for ONE city, cache to disk."""
    units_gdf, acts = build_units(out)
    np.save(CACHE / f"acts_{city}.npy", acts)
    units_gdf[["unit_id", "geometry"]].to_file(
        CACHE / f"units_{city}.geojson", driver="GeoJSON")
    print(f"{city}: cached {len(units_gdf)} units, acts {acts.shape}", flush=True)


def plot_all():
    """Phase B: joint PCA over both cities, render RGB maps."""
    import geopandas as gpd
    from sklearn.decomposition import PCA

    acts = {c: np.load(CACHE / f"acts_{c}.npy") for c, _ in CITIES}
    units = {c: gpd.read_file(CACHE / f"units_{c}.geojson") for c, _ in CITIES}

    stacked = np.vstack([acts[c] for c, _ in CITIES])
    pca = PCA(n_components=3, random_state=42).fit(stacked)
    proj = pca.transform(stacked)
    evr = pca.explained_variance_ratio_
    print("explained variance ratio:", np.round(evr, 3),
          "sum:", round(float(evr.sum()), 3))

    # Per-channel RANK (empirical-CDF) normalisation on the JOINT projection:
    # spreads colours across the full RGB cube even when a component is skewed,
    # giving far better visual contrast than linear min-max. Rank is computed
    # once on both cities together so colours stay comparable across maps.
    sorted_ch = [np.sort(proj[:, j]) for j in range(3)]
    ranks = np.linspace(0.0, 1.0, len(proj))

    def to_rgb(a):
        p = pca.transform(a)
        return np.stack([np.interp(p[:, j], sorted_ch[j], ranks)
                         for j in range(3)], axis=1)

    for city, _ in CITIES:
        u = units[city]
        color_list = [tuple(c) for c in to_rgb(acts[city])]
        fig, ax = plt.subplots(figsize=(11, 11))
        u.plot(ax=ax, color=color_list, linewidth=0.9)
        ax.set_title(
            f"{city} — MRLU coloured by PCA of visual activation (RGB)\n"
            f"joint 3-component PCA over both cities "
            f"(EVR={evr.sum():.2f}); n={len(u)}", fontsize=13)
        ax.set_xlabel("longitude"); ax.set_ylabel("latitude")
        ax.set_aspect("equal"); ax.grid(True, alpha=0.2)
        p = FIG / f"map_units_pca_{city}.png"
        fig.savefig(p, dpi=130, bbox_inches="tight")
        plt.close(fig)
        print("saved", p, flush=True)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--task", required=True,
                    help="extract:<City> | plot")
    args = ap.parse_args()
    cmap = dict(CITIES)
    if args.task.startswith("extract:"):
        city = args.task.split(":", 1)[1]
        extract_city(city, cmap[city])
    elif args.task == "plot":
        plot_all()
    else:
        raise SystemExit(f"unknown task {args.task}")
