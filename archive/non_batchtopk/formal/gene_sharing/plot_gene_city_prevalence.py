#!/usr/bin/env python3
"""Plot how many SAE genes are shared across N of the 12 study cities."""

import csv
from collections import Counter
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.patches import Patch


REPO = Path(__file__).resolve().parents[2]
INPUT = REPO / "formal/site/citygenome/gene_city_specificity_batchtopk_w1024_k8.csv"
OUT = Path(__file__).resolve().parent

COLORS = {
    0: "#69778d", 1: "#ff7488", 2: "#ff9e64",
    3: "#f6c945", 4: "#f6c945", 5: "#f6c945",
    6: "#a58bff", 7: "#a58bff", 8: "#a58bff",
    9: "#39d6ff", 10: "#39d6ff", 11: "#39d6ff", 12: "#22e0a1",
}


def main():
    with INPUT.open() as f:
        rows = list(csv.DictReader(f))
    counts = Counter(int(float(r["prevalence_n_cities"])) for r in rows)
    x = list(range(13))
    y = [counts[i] for i in x]

    with (OUT / "gene_city_prevalence_counts.csv").open("w", newline="") as f:
        w = csv.writer(f, lineterminator="\n")
        w.writerow(("n_cities", "n_genes", "share_of_dictionary"))
        for n, value in zip(x, y):
            w.writerow((n, value, f"{value / len(rows):.6f}"))

    plt.style.use("dark_background")
    fig, ax = plt.subplots(figsize=(14, 7.4), facecolor="#070b14")
    ax.set_facecolor("#0d1626")
    bars = ax.bar(x, y, width=.72, color=[COLORS[i] for i in x],
                  edgecolor="#dbe4f0", linewidth=.35)
    for bar, value in zip(bars, y):
        ax.text(bar.get_x() + bar.get_width()/2, value + 3, str(value),
                ha="center", va="bottom", color="#eaf2fc", fontsize=10, weight="bold")

    ax.set_title("How Widely Are Visual Genes Shared Across Cities?", loc="left",
                 fontsize=22, weight="bold", pad=28, color="#eef5ff")
    ax.text(0, 1.025,
            "BatchTopK SAE · W1024/K8 · 12-city occurrence threshold ≥ 0.05% of city patches",
            transform=ax.transAxes, fontsize=10, color="#94a7c1")
    ax.set_xlabel("Number of cities in which a gene is present", fontsize=11, color="#b9c7d9", labelpad=12)
    ax.set_ylabel("Number of SAE genes", fontsize=11, color="#b9c7d9", labelpad=10)
    ax.set_xticks(x)
    ax.set_xlim(-.65, 12.65)
    ax.set_ylim(0, max(y) * 1.22)
    ax.grid(axis="y", color="#2b3c57", alpha=.5, linewidth=.6)
    ax.set_axisbelow(True)
    ax.tick_params(colors="#9badc5")
    for spine in ax.spines.values(): spine.set_color("#2b3c57")

    legend = [
        Patch(facecolor="#69778d", label="Unused / below threshold (0)"),
        Patch(facecolor="#ff7488", label="City-unique (1)"),
        Patch(facecolor="#ff9e64", label="Pair-specific (2)"),
        Patch(facecolor="#f6c945", label="Regional (3–5)"),
        Patch(facecolor="#a58bff", label="Accessory (6–8)"),
        Patch(facecolor="#39d6ff", label="Near-core (9–11)"),
        Patch(facecolor="#22e0a1", label="Universal core (12)"),
    ]
    ax.legend(handles=legend, ncol=4, loc="upper center", bbox_to_anchor=(.5, -.16),
              frameon=False, fontsize=9, labelcolor="#c9d5e5")

    used = len(rows) - counts[0]
    fig.text(.985, .94, f"{used} used genes\n{counts[12]} universal core",
             ha="right", va="top", color="#22e0a1", fontsize=11, weight="bold")
    fig.tight_layout(rect=(0, .06, 1, 1))
    fig.savefig(OUT / "gene_city_prevalence_bar.png", dpi=240, bbox_inches="tight", facecolor=fig.get_facecolor())
    fig.savefig(OUT / "gene_city_prevalence_bar.svg", bbox_inches="tight", facecolor=fig.get_facecolor())
    print(dict(sorted(counts.items())))


if __name__ == "__main__":
    OUT.mkdir(parents=True, exist_ok=True)
    main()
