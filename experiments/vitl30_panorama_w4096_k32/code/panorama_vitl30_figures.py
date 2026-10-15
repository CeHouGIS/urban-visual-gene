"""Draw Chapter 5 figures exclusively from the completed ViT-L panorama run."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.colors import TwoSlopeNorm
from scipy.cluster.hierarchy import dendrogram, linkage
from scipy.spatial.distance import pdist
from sklearn.metrics.pairwise import cosine_similarity

from sae_experiments.io_utils import write_json
from sae_experiments.panorama_vitl30 import OUTPUT, SOURCE, cohort, filtered_city
from sae_experiments.panorama_vitl30_analysis import (
    GROUPS, checked_groups, model_and_identity, analysis_cities, analysis_city_data,
)


def name(city: str) -> str:
    return city.split("/")[-1].replace("RiodeJaneiro", "Rio de Janeiro").replace(
        "BuenosAires", "Buenos Aires").replace("SaoPaulo", "Sao Paulo").replace(
        "CapeTown", "Cape Town").replace("NewYork", "New York")


def save(fig, folder: Path, stem: str):
    folder.mkdir(parents=True, exist_ok=True)
    fig.savefig(folder / f"{stem}.pdf", bbox_inches="tight")
    fig.savefig(folder / f"{stem}.png", dpi=180, bbox_inches="tight")
    plt.close(fig)


def load(args):
    model, model_hash = model_and_identity(args.output, "cpu")
    labels = checked_groups(args.output, model_hash)
    cities = analysis_cities(args)
    frames, shares, all_rows = {}, {}, []
    for city in cities:
        frame, _, _ = analysis_city_data(args, city)
        pano = frame.iloc[::4].reset_index(drop=True)
        path = args.output / "analysis/cities" / city.replace("/", "__") / "panorama_group_share.f32.npy"
        if not path.is_file():
            raise FileNotFoundError(f"New-model inference incomplete: {city}")
        values = np.load(path, mmap_mode="r")
        if values.shape != (len(pano), GROUPS):
            raise ValueError(f"Panorama shares do not match manifest: {city}")
        frames[city] = pano
        shares[city] = np.asarray(values, dtype=np.float32)
        all_rows.append({"city": city, "panoramas": len(pano), "latitude": float(pano.lat.median()),
                         "longitude": float(pano.lon.median())})
    return model, model_hash, labels, cities, frames, shares, pd.DataFrame(all_rows)


def city_map(figures, locations):
    locations = locations.dropna(subset=["latitude", "longitude"])
    fig, ax = plt.subplots(figsize=(11.5, 5.5), layout="constrained")
    try:
        import geopandas as gpd
        source = Path("/usr/local/lib/python3.11/site-packages/pyogrio/tests/fixtures/"
                      "naturalearth_lowres/naturalearth_lowres.shp")
        if source.is_file():
            gpd.read_file(source).plot(ax=ax, color="#e5ece7", edgecolor="#aabbb2", linewidth=.4)
    except (ImportError, OSError, ValueError):
        pass
    ax.scatter(locations.longitude, locations.latitude, s=34, c="#bc503d", zorder=3)
    for i, row in enumerate(locations.itertuples(index=False), 1):
        ax.annotate(str(i), (row.longitude, row.latitude), xytext=(4, 3),
                    textcoords="offset points", fontsize=7)
    ax.set(xlim=(-180, 180), ylim=(-60, 80), xlabel="Longitude", ylabel="Latitude",
           title=f"Mapped panorama samples in {len(locations)} study cities")
    key = "  ".join(f"{i+1} {name(city)}" for i, city in enumerate(locations.city))
    fig.text(.5, -.02, key, ha="center", va="top", fontsize=7, wrap=True)
    save(fig, figures, "Fig_ViTL30_Study_Cities")


def loss_curve(figures, output):
    history = pd.read_csv(output / "history.csv")
    if history.empty or not history.val_squared_l2.notna().any():
        raise ValueError("Training history lacks validation results")
    steps = int(json.loads((output / "experiment.json").read_text())["identity"]["steps_per_epoch"])
    x = (history.epoch - 1) * steps + history.step
    fig, ax = plt.subplots(figsize=(7.6, 3.7), layout="constrained")
    ax.plot(x, history.train_squared_l2, color="#247e77", lw=1.1, alpha=.9, label="Training")
    valid = history.val_squared_l2.notna()
    val_x = history.epoch[valid] * steps
    ax.plot(val_x, history.val_squared_l2[valid], color="#bd5945", marker="o", ms=3,
            label="Fixed sampled validation")
    best = history.loc[valid, "val_squared_l2"].idxmin()
    best_epoch = int(history.loc[best, "epoch"])
    ax.axvline(best_epoch * steps, color="#bd5945", lw=.8, ls="--", alpha=.6)
    ax.annotate(f"Selected epoch {best_epoch}",
                (best_epoch * steps, history.loc[best, "val_squared_l2"]),
                xytext=(-85, 14), textcoords="offset points", fontsize=8)
    ax.set(xlabel="Optimizer step", ylabel="Mean squared L2 reconstruction error",
           title="ViT-L panorama BatchTopK SAE, W=4,096 / k=32")
    ax.legend(frameon=False)
    save(fig, figures, "Fig_ViTL30_Training_Loss")


def hierarchy(figures, model, labels):
    directions = model.decoder.weight.detach().numpy().T
    tree = linkage(pdist(directions, metric="cosine"), method="average")
    fig, axes = plt.subplots(1, 2, figsize=(10.5, 4.2), layout="constrained",
                             gridspec_kw={"width_ratios": [2.5, 1]})
    den = dendrogram(tree, truncate_mode="lastp", p=GROUPS, no_labels=True, ax=axes[0],
                     color_threshold=0, above_threshold_color="#2b8279")
    heights = np.asarray(den["dcoord"])
    heights = heights[heights > 0]
    axes[0].set_ylim(max(0, heights.min() - .02), heights.max() + .02)
    axes[0].set(xlabel="64 terminal branches", ylabel="Decoder cosine distance",
                title="Average-linkage SAE dictionary hierarchy")
    axes[1].hist(np.bincount(labels, minlength=GROUPS), bins=14, color="#c16d50")
    axes[1].set(xlabel="Features per group", ylabel="Groups", title="64-group size distribution")
    save(fig, figures, "Fig_ViTL30_Decoder_Hierarchy")


def composition_figures(figures, cities, shares):
    comp = np.stack([shares[city].mean(axis=0) for city in cities])
    prevalence = np.stack([(shares[city] >= .01).mean(axis=0) for city in cities])
    shared = ((prevalence >= .1).sum(axis=0) >= 27) & (comp.mean(axis=0) >= .005)
    order = np.argsort(-comp.mean(axis=0))
    others = (comp.sum(axis=0)[None, :] - comp) / (len(cities) - 1)
    excess = (comp - others) * 100
    scale = max(np.quantile(np.abs(excess), .98), 1e-4)
    fig, axes = plt.subplots(2, 1, figsize=(11, 8), sharex=True, layout="constrained",
                             gridspec_kw={"height_ratios": [1, 3]})
    axes[0].bar(range(GROUPS), comp.mean(axis=0)[order] * 100,
                color=np.where(shared[order], "#247e77", "#b7c1be"))
    axes[0].set(ylabel="Equal-city mean (%)", title="Shared groups and city-specific excess")
    im = axes[1].imshow(excess[:, order], aspect="auto", cmap="RdBu_r",
                        norm=TwoSlopeNorm(vmin=-scale, vcenter=0, vmax=scale))
    axes[1].set_yticks(range(len(cities)), [name(c) for c in cities], fontsize=7)
    axes[1].set(xlabel="SAE groups, ordered by mean share", ylabel="City")
    fig.colorbar(im, ax=axes[1], label="Percentage-point excess", shrink=.8)
    save(fig, figures, "Fig_ViTL30_Shared_Distinctive")

    fig, ax = plt.subplots(figsize=(11, 6), layout="constrained")
    im = ax.imshow(comp * 100, aspect="auto", cmap="YlGnBu")
    ax.set_yticks(range(len(cities)), [name(c) for c in cities], fontsize=7)
    ax.set(xlabel="SAE group", title="Mean four-view panorama group share by city")
    fig.colorbar(im, ax=ax, label="Dominant-patch share (%)", shrink=.8)
    save(fig, figures, "Fig_ViTL30_City_Composition")

    similarity = cosine_similarity(comp)
    fig, ax = plt.subplots(figsize=(7.5, 6.7), layout="constrained")
    im = ax.imshow(similarity, vmin=0, vmax=1, cmap="magma")
    ax.set_xticks(range(len(cities)), [name(c) for c in cities], rotation=90, fontsize=6)
    ax.set_yticks(range(len(cities)), [name(c) for c in cities], fontsize=6)
    ax.set_title("Cosine similarity of city panorama compositions")
    fig.colorbar(im, ax=ax, label="Cosine similarity", shrink=.8)
    save(fig, figures, "Fig_ViTL30_City_Similarity")
    return comp, prevalence, shared, similarity


def cooccurrence(figures, cities, shares, output):
    occurrences = np.zeros((GROUPS, GROUPS), dtype=np.int64)
    marginals = np.zeros(GROUPS, dtype=np.int64)
    total = 0
    for city in cities:
        source = output / "analysis/cities" / city.replace("/", "__") / "image_group_share.f32.npy"
        presence = np.load(source).reshape(-1, GROUPS) >= .01
        marginals += presence.sum(axis=0)
        occurrences += presence.astype(np.int64).T @ presence.astype(np.int64)
        total += len(presence)
    smooth = .5
    pair_p = (occurrences + smooth) / (total + smooth)
    single_p = (marginals + smooth) / (total + smooth)
    ppmi = np.maximum(np.log(pair_p / np.outer(single_p, single_p)), 0)
    ppmi[occurrences < max(20, int(total * .0005))] = 0
    np.fill_diagonal(ppmi, 0)
    fig, ax = plt.subplots(figsize=(6.8, 6), layout="constrained")
    im = ax.imshow(ppmi, cmap="magma", vmin=0)
    ax.set(xlabel="SAE group", ylabel="SAE group", title="Directional-image group co-occurrence")
    fig.colorbar(im, ax=ax, label="PPMI (natural log)", shrink=.8)
    save(fig, figures, "Fig_ViTL30_Cooccurrence")
    return ppmi, occurrences, marginals


def spatial(figures, cities, frames, shares):
    rows = []
    for city in cities:
        pano = frames[city]
        mask = pano.lat.notna() & pano.lon.notna()
        pano = pano.loc[mask].reset_index(drop=True)
        if len(pano) < 2:
            continue
        lat = pano.lat.to_numpy()
        lon = pano.lon.to_numpy()
        ref_lat = np.deg2rad(np.median(lat))
        x = np.floor(lon * 111.32 * np.cos(ref_lat)).astype(np.int32)
        y = np.floor(lat * 110.57).astype(np.int32)
        coords = pd.DataFrame({"x": x, "y": y})
        counts = coords.groupby(["x", "y"]).size()
        valid = counts[counts >= 5].index
        if len(valid) < 2:
            continue
        grouped = pd.DataFrame(shares[city][mask.to_numpy()]).groupby([coords.x, coords.y]).mean()
        cells = grouped.loc[valid]
        sim = cosine_similarity(cells.to_numpy())
        global_mean = float(sim[np.triu_indices(len(cells), 1)].mean())
        lookup = {tuple(key): i for i, key in enumerate(cells.index)}
        adjacent = [(i, lookup[neighbor]) for (xx, yy), i in lookup.items()
                    for neighbor in ((xx + 1, yy), (xx, yy + 1)) if neighbor in lookup]
        local_mean = float(np.mean([sim[i, j] for i, j in adjacent])) if adjacent else np.nan
        rows.append({"city": city, "valid_cells": len(cells), "adjacent_pairs": len(adjacent),
                     "global_homogeneity": global_mean, "local_continuity": local_mean})
    table = pd.DataFrame(rows)
    eligible = table[table.adjacent_pairs >= 20]
    fig, ax = plt.subplots(figsize=(7.6, 5.2), layout="constrained")
    ax.scatter(eligible.global_homogeneity, eligible.local_continuity,
               s=30 + np.sqrt(eligible.adjacent_pairs) * 4, color="#247e77", alpha=.85)
    for row in eligible.itertuples():
        ax.annotate(name(row.city), (row.global_homogeneity, row.local_continuity),
                    xytext=(3, 3), textcoords="offset points", fontsize=7)
    ax.set(xlabel="Mean similarity among valid 1-km cells",
           ylabel="Mean similarity among adjacent cells",
           title="Within-city panorama group structure")
    save(fig, figures, "Fig_ViTL30_Spatial_Structure")
    return table


def umap_figure(figures, cities, shares):
    import umap
    samples, labels = [], []
    for city in cities:
        rng = np.random.default_rng(int.from_bytes(city.encode()[:4].ljust(4, b"0"), "little"))
        ids = rng.choice(len(shares[city]), min(100, len(shares[city])), replace=False)
        samples.append(shares[city][ids])
        labels.extend([city] * len(ids))
    vectors = np.concatenate(samples)
    coords = umap.UMAP(n_neighbors=30, min_dist=.1, metric="cosine", random_state=42).fit_transform(vectors)
    fig, ax = plt.subplots(figsize=(8.4, 6), layout="constrained")
    palette = plt.get_cmap("tab20", len(cities))
    labels = np.array(labels)
    for i, city in enumerate(cities):
        rows = labels == city
        ax.scatter(coords[rows, 0], coords[rows, 1], s=5, alpha=.55,
                   color=palette(i), label=name(city), rasterized=True)
    ax.set(xlabel="UMAP 1", ylabel="UMAP 2", title="Panorama-level SAE composition UMAP")
    ax.legend(loc="center left", bbox_to_anchor=(1.01, .5), fontsize=6.5, ncol=2,
              frameon=False, markerscale=2)
    save(fig, figures, "Fig_ViTL30_Panorama_UMAP")
    return coords, labels


def activation_examples(figures, output):
    import PIL.Image
    rows = json.loads((output / "analysis/top_examples.json").read_text())
    examples = []
    used = set()
    for row in rows:
        if row["group"] in used:
            continue
        folder = output / "analysis/evidence" / f"group_{row['group']:02d}"
        path = folder / f"{row['city'].replace('/', '__')}_{row['pano_index']}_{row['heading']}_panorama.jpg"
        if path.is_file():
            examples.append((row, path))
            used.add(row["group"])
        if len(examples) == 6:
            break
    if len(examples) < 4:
        raise ValueError("VLM evidence images are incomplete")
    fig, axes = plt.subplots(len(examples), 3, figsize=(14, len(examples) * 2.6),
                             gridspec_kw={"width_ratios": [1792 / 500, 1, 1]}, layout="constrained")
    for row_axes, (row, path) in zip(axes, examples):
        ax = row_axes[0]
        with PIL.Image.open(path) as image:
            ax.imshow(image)
        ax.set_title(f"Group {row['group']:02d}  |  {name(row['city'])}  |  peak heading {row['heading']} degrees",
                     fontsize=9, loc="left")
        ax.axis("off")
        for detail_ax, suffix, title in zip(row_axes[1:], ["crop", "activation_crop"],
                                            ["Original detail", "Activation detail"]):
            detail = path.with_name(path.name.replace("_panorama.jpg", f"_{suffix}.jpg"))
            with PIL.Image.open(detail) as image:
                detail_ax.imshow(image)
            detail_ax.set_title(title, fontsize=8)
            detail_ax.axis("off")
    save(fig, figures, "Fig_ViTL30_Activation_Examples")


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--source", type=Path, default=SOURCE)
    p.add_argument("--output", type=Path, default=OUTPUT)
    p.add_argument("--figures", type=Path, default=OUTPUT / "figures")
    args = p.parse_args()
    plt.rcParams.update({"font.size": 9, "axes.spines.top": False, "axes.spines.right": False,
                         "pdf.fonttype": 42, "figure.facecolor": "white"})
    model, model_hash, groups, cities, frames, shares, locations = load(args)
    results = args.output / "analysis/results"
    results.mkdir(parents=True, exist_ok=True)
    city_map(args.figures, locations)
    loss_curve(args.figures, args.output)
    hierarchy(args.figures, model, groups)
    comp, prevalence, shared, similarity = composition_figures(args.figures, cities, shares)
    ppmi, support, marginals = cooccurrence(args.figures, cities, shares, args.output)
    spatial_table = spatial(args.figures, cities, frames, shares)
    umap_figure(args.figures, cities, shares)
    activation_examples(args.figures, args.output)
    locations.to_csv(results / "city_locations.csv", index=False)
    pd.DataFrame(comp, index=cities).to_csv(results / "city_composition.csv")
    pd.DataFrame(prevalence, index=cities).to_csv(results / "city_prevalence.csv")
    pd.DataFrame(similarity, index=cities, columns=cities).to_csv(results / "city_similarity.csv")
    pd.DataFrame(ppmi).to_csv(results / "image_ppmi.csv", index=False)
    pd.DataFrame(support).to_csv(results / "image_pair_support.csv", index=False)
    spatial_table.to_csv(results / "spatial_structure.csv", index=False)
    write_json(results / "figure_report.json", {"status": "complete", "checkpoint_sha256": model_hash,
        "cities": len(cities), "panoramas": int(sum(len(shares[c]) for c in cities)),
        "shared_groups": int(shared.sum()), "shared_share": float(comp.mean(axis=0)[shared].sum()),
        "figures": sorted(path.name for path in args.figures.glob("Fig_ViTL30_*.pdf"))})
    print("figures complete", args.figures, flush=True)


if __name__ == "__main__":
    main()
