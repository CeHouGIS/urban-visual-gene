"""Plot 12-city visual-gene similarity as a dendrogram and clustered matrix."""
from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy.cluster.hierarchy import dendrogram, linkage
from scipy.spatial.distance import squareform


ROOT = Path(__file__).resolve().parents[2]
ACTIVATIONS = ROOT / "formal" / "formal_out_global3" / "genes" / "sparse_acts.npz"
OUTPUT = ROOT / "formal" / "figures" / "city_similarity_hierarchical_clustering.png"
CITIES = [
    "Hong Kong", "Singapore", "Amsterdam", "Cape Town", "Paris", "São Paulo",
    "Mexico City", "Sydney", "Jakarta", "Dhaka", "New Delhi", "Manila",
]
DATA_NAMES = [name.replace(" ", "") for name in CITIES]
DATA_NAMES[CITIES.index("São Paulo")] = "SaoPaulo"


def city_profiles() -> np.ndarray:
    with np.load(ACTIVATIONS) as payload:
        idx = payload["idx"]
        city = payload["city"].astype(str)
        top_gene = idx[:, :, 0].astype(np.int64)
        profiles = np.zeros((len(CITIES), 512), dtype=np.float64)
        for row, name in enumerate(DATA_NAMES):
            counts = np.bincount(top_gene[city == name].ravel(), minlength=512)
            profiles[row] = counts / counts.sum()
    return profiles


def js_distance(a: np.ndarray, b: np.ndarray) -> float:
    a = a + 1e-12
    b = b + 1e-12
    midpoint = 0.5 * (a + b)
    divergence = 0.5 * np.sum(a * np.log2(a / midpoint)) + 0.5 * np.sum(b * np.log2(b / midpoint))
    return float(np.sqrt(max(divergence, 0)))


def main() -> None:
    profiles = city_profiles()
    distances = np.zeros((len(CITIES), len(CITIES)), dtype=np.float64)
    for i in range(len(CITIES)):
        for j in range(i + 1, len(CITIES)):
            distances[i, j] = distances[j, i] = js_distance(profiles[i], profiles[j])
    tree = linkage(squareform(distances, checks=False), method="average")
    leaf_order = dendrogram(tree, no_plot=True)["leaves"]
    similarity = 1 - distances / distances.max()
    # A left-oriented scipy dendrogram draws its first leaf at the bottom.
    # Reverse the matrix order so both panels read identically from top to bottom.
    display_order = leaf_order[::-1]
    ordered = similarity[np.ix_(display_order, display_order)]
    ordered_names = [CITIES[i] for i in display_order]

    fig = plt.figure(figsize=(17, 8.5), facecolor="white")
    grid = fig.add_gridspec(1, 2, width_ratios=(1.05, 1), wspace=0.27, left=0.065, right=0.95, bottom=0.13, top=0.78)
    tree_ax = fig.add_subplot(grid[0, 0])
    matrix_ax = fig.add_subplot(grid[0, 1])

    dendrogram(
        tree,
        labels=CITIES,
        orientation="left",
        ax=tree_ax,
        color_threshold=0,
        above_threshold_color="#3976d3",
        link_color_func=lambda _: "#3976d3",
    )
    tree_ax.set_xlabel("Jensen–Shannon distance (lower = more similar)", color="#475569")
    tree_ax.tick_params(axis="both", colors="#334155", labelsize=10)
    tree_ax.spines[["top", "right", "left"]].set_visible(False)
    tree_ax.spines["bottom"].set_color("#cbd5e1")
    tree_ax.grid(axis="x", color="#e2e8f0", linewidth=0.8)
    tree_ax.set_title("Hierarchical clustering", loc="left", fontsize=14, weight="bold", color="#172033", pad=12)

    image = matrix_ax.imshow(ordered, cmap="YlGnBu", vmin=0, vmax=1, aspect="equal")
    ticks = np.arange(len(CITIES))
    matrix_ax.set_xticks(ticks, ordered_names, rotation=45, ha="right", fontsize=9)
    matrix_ax.set_yticks(ticks, ordered_names, fontsize=9)
    matrix_ax.tick_params(length=0, colors="#334155")
    matrix_ax.set_title("Similarity matrix in cluster order", loc="left", fontsize=14, weight="bold", color="#172033", pad=12)
    for i in range(len(CITIES)):
        for j in range(len(CITIES)):
            matrix_ax.text(j, i, f"{ordered[i, j]:.2f}", ha="center", va="center", fontsize=6.5,
                           color="white" if ordered[i, j] > 0.62 else "#172033")
    for spine in matrix_ax.spines.values():
        spine.set_visible(False)
    colorbar = fig.colorbar(image, ax=matrix_ax, fraction=0.045, pad=0.03)
    colorbar.set_label("Normalized similarity (1 − distance / max distance)", color="#475569", fontsize=9)
    colorbar.outline.set_edgecolor("#cbd5e1")

    fig.suptitle("Visual-gene similarity across 12 cities", x=0.065, y=0.96, ha="left", fontsize=23, weight="bold", color="#172033")
    fig.text(0.065, 0.895, "City profiles are the frequencies of 512 dominant SAE visual genes across a balanced Street View sample.", fontsize=11, color="#64748b")
    fig.text(0.065, 0.835, "Distance: Jensen–Shannon · Linkage: average", fontsize=9.5, color="#64748b")
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUTPUT, dpi=220, facecolor="white")
    plt.close(fig)
    print(f"wrote {OUTPUT}")
    print("leaf order:", " -> ".join(ordered_names))


if __name__ == "__main__":
    main()
