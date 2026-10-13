"""Replot the replacement NAS cohort as a coherent Chapter 5 figure series."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import TwoSlopeNorm
import numpy as np
import pandas as pd
from scipy.cluster.hierarchy import dendrogram

from .nas_batch import OUTPUT

NAMES = (
    "Fig_NAS30_Study_Cities", "Fig_NAS30_Training_Loss",
    "Fig_NAS30_Image_UMAP", "Fig_NAS30_Decoder_Hierarchy",
    "Fig_NAS30_Shared_Distinctive", "Fig_NAS30_City_Composition",
    "Fig_NAS30_City_Similarity", "Fig_NAS30_Cooccurrence",
    "Fig_NAS30_Spatial_Structure", "Fig_NAS30_Activation_Examples",
)


def style() -> None:
    plt.rcParams.update({
        "font.family": "DejaVu Sans", "font.size": 9, "axes.titlesize": 11,
        "axes.labelsize": 9, "figure.facecolor": "white", "axes.facecolor": "white",
        "savefig.facecolor": "white", "pdf.fonttype": 42, "ps.fonttype": 42,
        "axes.spines.top": False, "axes.spines.right": False,
    })


def save(fig, root: Path, name: str) -> None:
    root.mkdir(parents=True, exist_ok=True)
    fig.savefig(root / f"{name}.pdf", bbox_inches="tight")
    fig.savefig(root / f"{name}.png", dpi=220, bbox_inches="tight")
    plt.close(fig)


def city_name(city: str) -> str:
    return city.split("/")[-1].replace("RiodeJaneiro", "Rio de Janeiro").replace(
        "BuenosAires", "Buenos Aires").replace("SaoPaulo", "São Paulo").replace(
        "CapeTown", "Cape Town").replace("NewYork", "New York").replace(
        "Yogyakarta", "Yogyakarta")


def cohort_map(output: Path, figures: Path, cities: list[str]) -> None:
    records = []
    for city in cities:
        frame = pd.read_parquet(output / "manifests" / f"{city.replace('/', '__')}.parquet",
                                columns=["panoid", "lat", "lon"])
        frame = frame.drop_duplicates("panoid")
        records.append((city, float(frame.lon.median()), float(frame.lat.median()), len(frame)))
    frame = pd.DataFrame(records, columns=["city", "lon", "lat", "panoramas"])
    frame.to_csv(output / "results" / "chapter5_city_locations.csv", index=False)
    fig = plt.figure(figsize=(12, 7.3), constrained_layout=True)
    grid = fig.add_gridspec(2, 1, height_ratios=[5.6, 1.2])
    ax = fig.add_subplot(grid[0])
    key_ax = fig.add_subplot(grid[1])
    try:
        import geopandas as gpd
        source = Path("/usr/local/lib/python3.11/site-packages/pyogrio/tests/fixtures/"
                      "naturalearth_lowres/naturalearth_lowres.shp")
        if source.is_file():
            world = gpd.read_file(source)
            world.plot(ax=ax, color="#edf1ef", edgecolor="#b9c6c4", linewidth=.4)
    except (ImportError, OSError, ValueError):
        pass
    ax.scatter(frame.lon, frame.lat, s=35, color="#bd5945", edgecolors="white",
               linewidth=.6, zorder=3)
    label_offsets = {6: (-12, -11), 13: (-14, 6), 14: (6, 7),
                     17: (6, -10), 20: (6, 12), 26: (-12, -10)}
    for i, row in enumerate(frame.itertuples(index=False), start=1):
        ax.annotate(str(i), (row.lon, row.lat), fontsize=7,
                    xytext=label_offsets.get(i, (4, 4)),
                    textcoords="offset points", zorder=4)
    ax.set(xlim=(-180, 180), ylim=(-60, 82), xlabel="Longitude", ylabel="Latitude",
           title="Independent NAS cohort: median location of sampled panoramas in 30 cities")
    ax.grid(color="#d9e3e1", linewidth=.4, alpha=.65)
    key_ax.axis("off")
    for i, city in enumerate(cities):
        col, row = divmod(i, 10)
        key_ax.text(col / 3 + .02, 1 - (row + .5) / 10,
                    f"{i + 1:02d}  {city_name(city)}", ha="left", va="center",
                    transform=key_ax.transAxes, fontsize=8)
    save(fig, figures, NAMES[0])


def training(output: Path, figures: Path) -> None:
    history = pd.read_csv(output / "history.csv")
    step = np.arange(1, len(history) + 1)
    epoch_end = history.val_squared_l2.notna()
    fig, ax = plt.subplots(figsize=(8.2, 3.8), constrained_layout=True)
    ax.plot(step, history.train_squared_l2.rolling(250, min_periods=1).mean(),
            color="#237c79", lw=1.2, label="Train, 250-step rolling mean")
    ax.plot(step[epoch_end], history.loc[epoch_end, "val_squared_l2"],
            color="#b95743", lw=1.3, marker="o", markersize=3.5,
            label="Fixed sampled validation set")
    ax.set(xlabel="Optimizer step", ylabel="Mean squared L2 reconstruction loss",
           title="BatchTopK SAE W4096 / K32 training")
    ax.legend(frameon=False)
    save(fig, figures, NAMES[1])


def image_umap(output: Path, figures: Path, cities: list[str]) -> None:
    frame = pd.read_parquet(output / "results" / "image_sample_umap.parquet")
    fig, ax = plt.subplots(figsize=(8.7, 6.8), constrained_layout=True)
    palette = plt.get_cmap("tab20", len(cities))
    for i, city in enumerate(cities):
        block = frame[frame.city == city]
        ax.scatter(block.umap_x, block.umap_y, s=3, alpha=.55, color=palette(i),
                   label=city_name(city), rasterized=True, linewidths=0)
    ax.set(xlabel="UMAP 1", ylabel="UMAP 2",
           title="Image-level SAE cluster composition; 200 direction images per city")
    ax.legend(loc="center left", bbox_to_anchor=(1.01, .5), ncol=2,
              fontsize=6.5, frameon=False, markerscale=3)
    save(fig, figures, NAMES[2])


def hierarchy(output: Path, figures: Path) -> None:
    tree = np.load(output / "hierarchy_linkage.npy")
    sizes = np.bincount(np.load(output / "feature_to_cluster64.npy"), minlength=64)
    fig, axes = plt.subplots(1, 2, figsize=(11.5, 4.4),
                             gridspec_kw={"width_ratios": [2.5, 1]},
                             constrained_layout=True)
    branches = dendrogram(tree, truncate_mode="lastp", p=64, no_labels=True,
                          color_threshold=0, above_threshold_color="#337e7a", ax=axes[0])
    merge_heights = np.asarray(branches["dcoord"])
    merge_heights = merge_heights[merge_heights > 0]
    margin = max(float(np.ptp(merge_heights)) * .06, 1e-3)
    axes[0].set_ylim(merge_heights.min() - margin, merge_heights.max() + margin)
    axes[0].set(xlabel="64 terminal branches", ylabel="Decoder cosine distance",
                title="Average-linkage hierarchy of 4,096 SAE decoder directions")
    axes[1].hist(sizes, bins=16, color="#be7654", edgecolor="white")
    axes[1].set(xlabel="Dictionary features per cluster", ylabel="Clusters",
                title="64-cluster size distribution")
    save(fig, figures, NAMES[3])


def pano_shares(output: Path, cities: list[str]) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    counts = np.load(output / "image_cluster64_patch_winners.u16.npy", mmap_mode="r")
    index = pd.read_parquet(output / "inference_image_index.parquet")
    shares = np.empty((len(cities), 64), dtype=np.float64)
    prevalence = np.empty((len(cities), 64), dtype=np.float64)
    city_sizes = np.empty(len(cities), dtype=np.int64)
    for i, city in enumerate(cities):
        ids = np.flatnonzero(index.city.to_numpy() == city)
        if len(ids) % 4 or not np.all(np.diff(index.source_image_index.iloc[ids].to_numpy().reshape(-1, 4), axis=1) == 1):
            raise ValueError(f"Non-four-view inference order: {city}")
        pano = np.asarray(counts[ids], dtype=np.float32).reshape(-1, 4, 64).sum(axis=1)
        totals = pano.sum(axis=1)
        pano /= totals[:, None].clip(min=1)
        shares[i] = pano.mean(axis=0)
        prevalence[i] = (pano >= .01).mean(axis=0)
        city_sizes[i] = len(pano)
    return shares, prevalence, city_sizes


def shared_distinctive(output: Path, figures: Path, cities: list[str]) -> dict:
    shares, prevalence, city_sizes = pano_shares(output, cities)
    shared = ((prevalence >= .10).sum(axis=0) >= 27) & (shares.mean(axis=0) >= .005)
    mean = shares.mean(axis=0)
    order = np.argsort(-mean)
    excess = 100 * (shares - (shares.sum(axis=0)[None, :] - shares) / 29)
    limit = float(np.quantile(np.abs(excess), .98))
    fig, axes = plt.subplots(2, 1, figsize=(12, 8.1), sharex=True,
                             gridspec_kw={"height_ratios": [1, 3.4]},
                             constrained_layout=True)
    axes[0].bar(np.arange(64), 100 * mean[order],
                color=np.where(shared[order], "#237c79", "#b7c2c0"), width=.88)
    axes[0].set(ylabel="Equal-city mean\npatch share (%)",
                title="Shared response and city-specific excess (independent NAS cohort)")
    im = axes[1].imshow(excess[:, order], aspect="auto", cmap="RdBu_r",
                        norm=TwoSlopeNorm(vmin=-limit, vcenter=0, vmax=limit))
    axes[1].set_yticks(range(30), [city_name(c) for c in cities], fontsize=7)
    axes[1].set_xticks(range(0, 64, 4), [f"F{order[j]:02d}" for j in range(0, 64, 4)],
                       rotation=90, fontsize=7)
    axes[1].set(xlabel="64 SAE clusters, ordered by equal-city mean share",
                ylabel="City")
    fig.colorbar(im, ax=axes[1], fraction=.02, label="City minus other-city mean (percentage points)",
                 extend="both")
    save(fig, figures, NAMES[4])
    table = pd.DataFrame(shares, index=cities, columns=[f"F{i:02d}" for i in range(64)])
    table.to_csv(output / "results" / "chapter5_panorama_city_composition.csv")
    pd.DataFrame(prevalence, index=cities, columns=table.columns).to_csv(
        output / "results" / "chapter5_pano_cluster_prevalence.csv")
    overlap = np.minimum(shares, (shares.sum(axis=0)[None, :] - shares) / 29).sum(axis=1)
    return {"panoramas": int(city_sizes.sum()), "shared_clusters": int(shared.sum()),
            "shared_mean_response_share": float(mean[shared].sum()),
            "city_reference_overlap_range": [float(overlap.min()), float(overlap.max())],
            "shared_rule": ">=1% patch share in >=10% panoramas of >=27 cities and >=0.5% equal-city mean share"}


def matrices(output: Path, figures: Path, cities: list[str]) -> None:
    names = [city_name(c) for c in cities]
    composition = pd.read_csv(output / "results" / "chapter5_panorama_city_composition.csv",
                              index_col=0).to_numpy()
    fig, ax = plt.subplots(figsize=(12.2, 6.2), constrained_layout=True)
    im = ax.imshow(composition * 100, aspect="auto", cmap="YlGnBu")
    ax.set_yticks(range(30), names, fontsize=7)
    ax.set_xticks(range(0, 64, 4), [f"F{j:02d}" for j in range(0, 64, 4)],
                  rotation=90, fontsize=7)
    ax.set(xlabel="SAE cluster", title="Mean panorama-level dominant-patch share by city")
    fig.colorbar(im, ax=ax, fraction=.025, label="Patch share (%)")
    save(fig, figures, NAMES[5])

    normalized = composition / np.linalg.norm(composition, axis=1, keepdims=True)
    similarity = normalized @ normalized.T
    pd.DataFrame(similarity, index=cities, columns=cities).to_csv(
        output / "results" / "chapter5_panorama_city_similarity.csv")
    fig, ax = plt.subplots(figsize=(7.8, 6.8), constrained_layout=True)
    im = ax.imshow(similarity, cmap="magma", vmin=0, vmax=1)
    ax.set_xticks(range(30), names, rotation=90, fontsize=6)
    ax.set_yticks(range(30), names, fontsize=6)
    ax.set_title("Cosine similarity of panorama-averaged SAE cluster shares")
    fig.colorbar(im, ax=ax, fraction=.035, label="Cosine similarity")
    save(fig, figures, NAMES[6])

    ppmi = pd.read_csv(output / "results" / "global_element_ppmi.csv",
                       index_col=0).to_numpy()
    fig, ax = plt.subplots(figsize=(7.1, 6.3), constrained_layout=True)
    im = ax.imshow(ppmi, cmap="magma", vmin=0)
    ax.set_xticks(range(0, 64, 8), [f"F{j:02d}" for j in range(0, 64, 8)])
    ax.set_yticks(range(0, 64, 8), [f"F{j:02d}" for j in range(0, 64, 8)])
    ax.set(xlabel="SAE cluster", ylabel="SAE cluster",
           title="Global PPMI of top-eight cluster presence within direction images")
    fig.colorbar(im, ax=ax, fraction=.04, label="PPMI (natural log)")
    save(fig, figures, NAMES[7])


def spatial(output: Path, figures: Path) -> None:
    frame = pd.read_csv(output / "results" / "city_spatial_structure_sae.csv")
    valid = frame[frame.eligible_for_plot].sort_values("visual_homogeneity")
    fig, ax = plt.subplots(figsize=(7.2, 5.2), constrained_layout=True)
    sizes = 25 + np.sqrt(valid.adjacent_grid_pairs.to_numpy()) * 5
    ax.scatter(valid.visual_homogeneity, valid.local_visual_continuity,
               s=sizes, color="#237c79", alpha=.78, edgecolors="white", linewidth=.4)
    highlights = set(valid.nsmallest(3, "visual_homogeneity").city) | set(
        valid.nlargest(3, "visual_homogeneity").city)
    label_positions = {"Brasília": (-6, 12, "right"),
                       "Cape Town": (8, 16, "left"),
                       "Monterrey": (8, -14, "left")}
    for row in valid.itertuples(index=False):
        if row.city in highlights:
            offset_x, offset_y, align = label_positions.get(
                city_name(row.city), (4, 4, "left"))
            ax.annotate(city_name(row.city),
                        (row.visual_homogeneity, row.local_visual_continuity),
                        fontsize=7, xytext=(offset_x, offset_y),
                        ha=align, textcoords="offset points")
    ax.set(xlabel="Mean similarity among all valid city grids",
           ylabel="Mean similarity among adjacent city grids",
           title="Within-city SAE composition (1 km, at least 5 panoramas/grid)")
    fig.text(.5, -.01, "21/30 cities have at least 20 adjacent valid grid pairs; "
             "all 30 remain in the data table.", ha="center", fontsize=7)
    save(fig, figures, NAMES[8])


def examples(output: Path, figures: Path) -> None:
    from PIL import Image
    original = output / "figures" / "Fig_BatchTopK_Panorama_Activation_Overlays.png"
    if not original.is_file():
        raise FileNotFoundError(original)
    with Image.open(original) as image:
        image.save(figures / f"{NAMES[9]}.png")
    # Keep raster examples as PNG to avoid embedding heavy bitmaps in a vector PDF.


def run(output: Path, figures: Path) -> dict:
    style()
    cohort = json.loads((output / "manifests" / "batch.json").read_text())
    audit = json.loads((output / "data_audit.json").read_text())
    inference = json.loads((output / "inference_report.json").read_text())
    if (cohort["images"], audit["actual_images"], inference["images"]) != (370768, 363700, 363700):
        raise ValueError("Unexpected NAS cohort")
    if inference["data_identity"] != audit["dataset_identity"]:
        raise ValueError("Inference and data audit do not match")
    cities = cohort["cities"]
    if len(cities) != 30:
        raise ValueError("Expected 30 cities")
    figures.mkdir(parents=True, exist_ok=True)
    cohort_map(output, figures, cities)
    training(output, figures)
    image_umap(output, figures, cities)
    hierarchy(output, figures)
    shared = shared_distinctive(output, figures, cities)
    matrices(output, figures, cities)
    spatial(output, figures)
    examples(output, figures)
    report = {"status": "complete", "source_experiment": str(output.resolve()),
              "dataset_identity": inference["data_identity"],
              "images": inference["images"], "panoramas": shared["panoramas"],
              "figures": list(NAMES), **{key: value for key, value in shared.items()
                                          if key != "panoramas"}}
    (figures / "figure_report.json").write_text(json.dumps(report, indent=2) + "\n")
    return report


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--output", type=Path, default=OUTPUT)
    p.add_argument("--figures", type=Path)
    args = p.parse_args()
    print(json.dumps(run(args.output, args.figures or args.output / "figures" / "chapter5"),
                     indent=2))


if __name__ == "__main__":
    main()
