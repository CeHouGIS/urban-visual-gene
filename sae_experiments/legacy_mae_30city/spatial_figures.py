"""Recompute within-city spatial structure using SAE cluster patch counts."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from .figures import save
from .nas_batch import OUTPUT
from .train import write_json


def metrics(lat: np.ndarray, lon: np.ndarray, counts: np.ndarray,
            grid_m: int = 1000, minimum_panos: int = 5) -> dict:
    radius = 6378137.0
    x = radius * np.deg2rad(lon)
    y = radius * np.log(np.tan(np.pi / 4 + np.deg2rad(np.clip(lat, -85.0511, 85.0511)) / 2))
    cells = np.column_stack((np.floor(x / grid_m), np.floor(y / grid_m))).astype(np.int64)
    unique, inverse = np.unique(cells, axis=0, return_inverse=True)
    n = np.bincount(inverse)
    summed = np.zeros((len(unique), 64), dtype=np.float64)
    np.add.at(summed, inverse, counts)
    keep = n >= minimum_panos
    coords = unique[keep]
    values = summed[keep]
    values /= values.sum(axis=1, keepdims=True).clip(min=1)
    norm = values / np.linalg.norm(values, axis=1, keepdims=True).clip(min=1e-12)
    count = len(norm)
    homogeneity = (float((norm.sum(axis=0) ** 2).sum()) - count) / (count * (count - 1)) if count > 1 else np.nan
    lookup = {tuple(coord): i for i, coord in enumerate(coords)}
    edges = [(i, lookup[target]) for i, (gx, gy) in enumerate(coords)
             for target in ((gx + 1, gy), (gx, gy + 1)) if target in lookup]
    continuity = float(np.mean([norm[a] @ norm[b] for a, b in edges])) if edges else np.nan
    return {"grid_size_m": grid_m, "minimum_panoramas": minimum_panos,
            "valid_grids": count, "adjacent_grid_pairs": len(edges),
            "visual_homogeneity": homogeneity, "local_visual_continuity": continuity}


def run(output: Path, figures: Path) -> dict:
    counts = np.load(output / "image_cluster64_patch_winners.u16.npy", mmap_mode="r")
    index = pd.read_parquet(output / "inference_image_index.parquet")
    if len(counts) != len(index) or counts.shape[1] != 64:
        raise ValueError("Image index and patch winner counts do not match")
    rows = []
    for city, group in index.groupby("city", sort=False):
        source_ids = group.source_image_index.to_numpy(np.int64)
        frame = pd.read_parquet(output / "manifests" / (city.replace("/", "__") + ".parquet"))
        meta = frame.iloc[source_ids]
        if (len(meta) % 4 or meta.heading.to_numpy().reshape(-1, 4).tolist()
                != [[0, 90, 180, 270]] * (len(meta) // 4)
                or not np.all(meta.panoid.to_numpy().reshape(-1, 4)
                              == meta.panoid.to_numpy().reshape(-1, 4)[:, :1])):
            raise ValueError(f"Pano image order changed for {city}")
        pano = np.asarray(counts[group.index.to_numpy()], dtype=np.float64).reshape(-1, 4, 64).sum(axis=1)
        first = meta.iloc[::4]
        rows.append({"city": city, "panoramas": len(pano),
                     **metrics(first.lat.to_numpy(), first.lon.to_numpy(), pano)})
    summary = pd.DataFrame(rows)
    summary["eligible_for_plot"] = summary.adjacent_grid_pairs >= 20
    (output / "results").mkdir(exist_ok=True)
    summary.to_csv(output / "results" / "city_spatial_structure_sae.csv", index=False)
    figures.mkdir(parents=True, exist_ok=True)
    plotted = summary[summary.eligible_for_plot]
    fig, ax = plt.subplots(figsize=(9, 7))
    ax.scatter(plotted.visual_homogeneity, plotted.local_visual_continuity,
               s=35 + plotted.adjacent_grid_pairs.clip(upper=1000) / 12,
               c="#237a78", alpha=.8)
    labeled = {"Paris", "Dhaka", "Naples", "Monterrey", "Lagos", "Taipei"}
    for row in plotted.itertuples(index=False):
        if row.city.split("/")[-1] in labeled:
            ax.annotate(row.city.split("/")[-1],
                        (row.visual_homogeneity, row.local_visual_continuity),
                        fontsize=7, xytext=(3, 2), textcoords="offset points")
    ax.set(xlabel="Mean similarity between all valid city grids",
           ylabel="Mean similarity between adjacent grids",
           title="SAE composition: city homogeneity and local continuity (at least 20 adjacent grids)")
    save(fig, figures, "Fig_Homogeneity_vs_Local_Continuity")
    ordered = plotted.sort_values("local_visual_continuity")
    fig, ax = plt.subplots(figsize=(10, 9))
    y = np.arange(len(ordered))
    ax.barh(y - .18, ordered.visual_homogeneity, height=.36, label="All-grid homogeneity", color="#277d79")
    ax.barh(y + .18, ordered.local_visual_continuity, height=.36, label="Adjacent-grid continuity", color="#d0804d")
    ax.set_yticks(y, ordered.city.str.split("/").str[-1], fontsize=8)
    ax.set(xlabel="Mean SAE-cluster composition cosine similarity",
           title="Within-city visual structure (1 km grid, at least 5 panoramas per grid)")
    ax.legend(frameon=False)
    save(fig, figures, "Fig_Within_City_Visual_Structure")
    payload = json.loads((output / "figure_inventory.json").read_text())
    generated = ["Fig_Homogeneity_vs_Local_Continuity", "Fig_Within_City_Visual_Structure"]
    for row in payload["figures"]:
        if row["legacy_figure"] in generated:
            row.update(new_figure=row["legacy_figure"], status="recomputed_from_SAE",
                       reason="SAE analogue: same 1 km grid and five-panorama threshold, new 64-cluster composition")
    payload["generated"] = sorted(set(payload.get("generated", [])) | set(generated))
    payload["recomputed_count"] = sum(row["status"] == "recomputed_from_SAE" for row in payload["figures"])
    write_json(output / "figure_inventory.json", payload)
    return {"generated": generated, "recomputed": payload["recomputed_count"]}


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--output", type=Path, default=OUTPUT)
    p.add_argument("--figures", type=Path)
    args = p.parse_args()
    print(json.dumps(run(args.output, args.figures or args.output / "figures"), indent=2))


if __name__ == "__main__":
    main()
