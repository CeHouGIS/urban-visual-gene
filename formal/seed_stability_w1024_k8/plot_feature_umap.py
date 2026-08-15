#!/usr/bin/env python3
"""Joint UMAP of decoder features from the five SAE random-seed runs."""

import csv
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
import umap


ROOT = Path(__file__).resolve().parent
SEEDS = (11, 23, 37, 53, 71)
COLORS = ("#36d7ff", "#a78bfa", "#22e0a1", "#f6c945", "#ff7488")


def main():
    vectors, seed_ids, gene_ids = [], [], []
    for seed in SEEDS:
        checkpoint = torch.load(
            ROOT / f"seed_{seed:03d}" / "batch_topk_w1024_k8.pt",
            map_location="cpu",
            weights_only=False,
        )
        # nn.Linear stores decoder atoms as columns: [output_dim, latent_dim].
        decoder = checkpoint["state"]["decoder.weight"].float().numpy().T
        decoder /= np.maximum(np.linalg.norm(decoder, axis=1, keepdims=True), 1e-12)
        vectors.append(decoder)
        seed_ids.extend([seed] * decoder.shape[0])
        gene_ids.extend(range(decoder.shape[0]))

    vectors = np.concatenate(vectors, axis=0)
    seed_ids = np.asarray(seed_ids)
    gene_ids = np.asarray(gene_ids)
    reducer = umap.UMAP(
        n_neighbors=30,
        min_dist=0.08,
        n_components=2,
        metric="cosine",
        random_state=42,
        n_jobs=1,
    )
    xy = reducer.fit_transform(vectors)

    membership = {}
    with (ROOT / "results" / "stable_gene_members.csv").open() as f:
        for row in csv.DictReader(f):
            membership[(int(row["seed"]), int(row["gene_id"]))] = row["family_id"]
    stable = np.asarray([(s, g) in membership for s, g in zip(seed_ids, gene_ids)])

    out = ROOT / "results" / "feature_umap.csv"
    with out.open("w", newline="") as f:
        writer = csv.writer(f, lineterminator="\n")
        writer.writerow(("seed", "gene_id", "umap_1", "umap_2", "stable", "family_id"))
        for s, g, p, ok in zip(seed_ids, gene_ids, xy, stable):
            writer.writerow((s, g, f"{p[0]:.6f}", f"{p[1]:.6f}", int(ok), membership.get((s, g), "")))

    plt.style.use("dark_background")
    fig, axes = plt.subplots(1, 3, figsize=(16, 5.2), facecolor="#070b14")
    panels = ((np.ones(len(xy), bool), "All 5,120 decoder features"),
              (stable, "Strict-stable members (4,238)"),
              (~stable, "Outside strict families (882)"))
    for ax, (mask, title) in zip(axes, panels):
        ax.set_facecolor("#0d1626")
        for seed, color in zip(SEEDS, COLORS):
            sel = mask & (seed_ids == seed)
            ax.scatter(xy[sel, 0], xy[sel, 1], s=7, alpha=.62, c=color,
                       linewidths=0, label=f"Seed {seed}", rasterized=True)
        ax.set_title(title, fontsize=12, weight="bold", pad=12)
        ax.set_xlabel("UMAP 1", color="#8a9bb5")
        ax.set_ylabel("UMAP 2", color="#8a9bb5")
        ax.grid(color="#20314d", alpha=.28, linewidth=.5)
        ax.tick_params(colors="#8a9bb5", labelsize=8)
        for spine in ax.spines.values():
            spine.set_color("#20314d")
    axes[0].legend(loc="best", frameon=True, facecolor="#121f34", edgecolor="#20314d",
                   fontsize=8, markerscale=2)
    fig.suptitle("Joint UMAP of SAE feature directions across random seeds",
                 fontsize=17, weight="bold", y=1.02)
    fig.text(.5, -.02,
             "Cosine UMAP on normalized decoder columns · n_neighbors=30 · min_dist=0.08 · random_state=42",
             ha="center", color="#8a9bb5", fontsize=9)
    fig.tight_layout()
    fig.savefig(ROOT / "results" / "feature_umap.png", dpi=220, bbox_inches="tight",
                facecolor=fig.get_facecolor())
    print(json.dumps({"points": len(xy), "stable": int(stable.sum()),
                      "unmatched": int((~stable).sum()), "csv": str(out)}, indent=2))


if __name__ == "__main__":
    main()
