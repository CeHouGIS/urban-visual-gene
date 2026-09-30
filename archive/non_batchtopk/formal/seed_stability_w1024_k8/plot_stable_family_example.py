#!/usr/bin/env python3
"""Select and visualize a strong five-seed stable semantic family."""

import csv
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from PIL import Image


ROOT = Path(__file__).resolve().parent
REPO = ROOT.parents[1]
SEEDS = (11, 23, 37, 53, 71)
COLORS = ("#39d6ff", "#a58bff", "#22e0a1", "#f6c945", "#ff7488")
REFERENCE_ATOMS = REPO / "formal/formal_out_global3/genes/atoms.npy"
REFERENCE_THUMBS = REPO / "formal/formal_out_global3/genes/thumbs"
SITE = REPO / "formal/site"


def cosine_matrix(x):
    x = x / np.maximum(np.linalg.norm(x, axis=1, keepdims=True), 1e-12)
    return x @ x.T


def main():
    atoms = np.load(REFERENCE_ATOMS).astype("float32")
    atoms /= np.maximum(np.linalg.norm(atoms, axis=1, keepdims=True), 1e-12)
    decoders, nearest = {}, {}
    for seed in SEEDS:
        z = np.load(ROOT / f"seed_{seed:03d}" / "anchor_activations.npz")
        decoder = z["decoder"].T.astype("float32")
        decoder /= np.maximum(np.linalg.norm(decoder, axis=1, keepdims=True), 1e-12)
        decoders[seed] = decoder
        sim = decoder @ atoms.T
        nearest[seed] = (sim.argmax(axis=1), sim.max(axis=1))

    families = list(csv.DictReader((ROOT / "results/stable_gene_families.csv").open()))
    candidates = []
    for row in families:
        if int(row["seed_coverage"]) != 5:
            continue
        members = {int(x.split(":")[0]): int(x.split(":")[1]) for x in row["members"].split(";")}
        refs = [int(nearest[s][0][members[s]]) for s in SEEDS]
        sims = [float(nearest[s][1][members[s]]) for s in SEEDS]
        if len(set(refs)) == 1:
            candidates.append((min(sims), float(row["median_combined_similarity"]), row, members, refs[0], sims))
    _, _, family, members, ref_gene, prototype_sims = max(candidates, key=lambda x: (x[0], x[1]))

    support = {}
    with (ROOT / "results/stable_gene_members.csv").open() as f:
        for row in csv.DictReader(f):
            if row["family_id"] == family["family_id"]:
                support[int(row["seed"])] = int(row["support"])

    decoder_vectors = np.stack([decoders[s][members[s]] for s in SEEDS])
    decoder_sim = cosine_matrix(decoder_vectors)
    activation_vectors = []
    for seed in SEEDS:
        z = np.load(ROOT / f"seed_{seed:03d}" / "anchor_activations.npz")
        gene = members[seed]
        v = np.zeros(int(z["shape"][0]), dtype="float32")
        take = z["cols"] == gene
        v[z["rows"][take]] = z["vals"][take].astype("float32")
        activation_vectors.append(v)
    activation_sim = cosine_matrix(np.stack(activation_vectors))

    plt.style.use("dark_background")
    fig = plt.figure(figsize=(16, 8.8), facecolor="#070b14")
    gs = fig.add_gridspec(2, 7, height_ratios=[1, 1.18], hspace=.32, wspace=.35)
    proto_ax = fig.add_subplot(gs[0, :2])
    proto_ax.imshow(Image.open(REFERENCE_THUMBS / f"{ref_gene}.jpg").resize((420, 420)))
    proto_ax.set_title(f"Shared semantic prototype · Gene {ref_gene}", fontsize=13, weight="bold", pad=10)
    proto_ax.text(.5, -.09, "Vegetated roadside / street-edge greenery", transform=proto_ax.transAxes,
                  ha="center", color="#9badc5", fontsize=10)
    proto_ax.set_axis_off()

    for i, (seed, color) in enumerate(zip(SEEDS, COLORS)):
        ax = fig.add_subplot(gs[0, i+2])
        ax.set_facecolor("#101b2e")
        ax.text(.5, .80, f"Seed {seed}", ha="center", color=color, fontsize=13, weight="bold")
        ax.text(.5, .57, f"Gene {members[seed]}", ha="center", color="#eef5ff", fontsize=18, weight="bold")
        ax.text(.5, .35, f"prototype cosine\n{prototype_sims[i]:.3f}", ha="center", color="#9badc5", fontsize=10)
        ax.text(.5, .14, f"support {support[seed]:,}", ha="center", color="#9badc5", fontsize=9)
        ax.set_xticks([]); ax.set_yticks([])
        for spine in ax.spines.values(): spine.set_color(color); spine.set_linewidth(1.2)

    for col, (matrix, title) in enumerate(((activation_sim, "Activation-profile cosine"),
                                            (decoder_sim, "Decoder-direction cosine"))):
        ax = fig.add_subplot(gs[1, col*3:(col+1)*3])
        im = ax.imshow(matrix, vmin=.9, vmax=1, cmap="viridis")
        ax.set_xticks(range(5), [f"S{s}" for s in SEEDS])
        ax.set_yticks(range(5), [f"S{s}" for s in SEEDS])
        ax.set_title(title, fontsize=13, weight="bold", pad=10)
        for i in range(5):
            for j in range(5):
                ax.text(j, i, f"{matrix[i,j]:.3f}", ha="center", va="center",
                        color="white" if matrix[i,j] < .975 else "#07101c", fontsize=9, weight="bold")
        fig.colorbar(im, ax=ax, fraction=.046, pad=.04)

    note = fig.add_subplot(gs[1, 6]); note.set_axis_off()
    note.text(0, .95, "Why this is a strong case", color="#22e0a1", fontsize=12, weight="bold", va="top")
    note.text(0, .79,
              f"5 / 5 seeds\nSame reference Gene {ref_gene}\n"
              f"Min prototype cosine {min(prototype_sims):.3f}\n"
              f"Median family similarity {float(family['median_combined_similarity']):.3f}\n"
              f"Min activation similarity {float(family['min_activation_similarity']):.3f}\n"
              f"Min decoder similarity {float(family['min_decoder_similarity']):.3f}",
              color="#c9d5e5", fontsize=10, linespacing=1.7, va="top")

    fig.suptitle(f"One Reproducible Visual-Gene Family Across Five Random Seeds · {family['family_id']}",
                 fontsize=20, weight="bold", y=.98)
    fig.text(.5, .02,
             "Independent gene IDs converge to the same reference direction and nearly identical activation behavior.",
             ha="center", color="#9badc5", fontsize=10)
    fig.savefig(SITE / "seed_stability_family_example.png", dpi=220, bbox_inches="tight", facecolor=fig.get_facecolor())

    result = {
        "family_id": family["family_id"], "reference_gene": ref_gene,
        "semantic_label": "vegetated roadside / street-edge greenery",
        "members": [{"seed": s, "gene": members[s], "support": support[s],
                     "prototype_cosine": round(prototype_sims[i], 6)} for i, s in enumerate(SEEDS)],
        "activation_similarity": activation_sim.round(6).tolist(),
        "decoder_similarity": decoder_sim.round(6).tolist(),
    }
    (SITE / "seed_stability_family_example.json").write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
