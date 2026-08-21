"""Summarize cross-seed activation coverage and render representative families."""
from __future__ import annotations

import csv
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


ROOT = Path(__file__).resolve().parents[2]
RUN_ROOT = ROOT / "formal" / "seed_stability_w1024_k8"
RESULTS = RUN_ROOT / "results"
FIGURES = ROOT / "formal" / "figures"
SEEDS = (11, 23, 37, 53, 71)
N_ANCHORS = 250_000
COLORS = {11: "#39d6ff", 23: "#a58bff", 37: "#22e0a1", 53: "#f6c945", 71: "#ff7488"}


def load_members() -> dict[str, dict[int, int]]:
    families: dict[str, dict[int, int]] = {}
    with (RESULTS / "stable_gene_members.csv").open(newline="") as handle:
        for row in csv.DictReader(handle):
            families.setdefault(row["family_id"], {})[int(row["seed"])] = int(row["gene_id"])
    return families


def load_seed(seed: int) -> dict[str, np.ndarray]:
    with np.load(RUN_ROOT / f"seed_{seed:03d}" / "anchor_activations.npz") as z:
        return {key: z[key].copy() for key in ("rows", "cols", "vals", "support")}


def activation_values(payload: dict[str, np.ndarray], gene: int) -> np.ndarray:
    return payload["vals"][payload["cols"] == gene].astype(np.float32)


def main() -> None:
    FIGURES.mkdir(parents=True, exist_ok=True)
    families = load_members()
    seed_data = {seed: load_seed(seed) for seed in SEEDS}

    records = []
    for family, members in families.items():
        supports = [int(seed_data[s]["support"][members[s]]) if s in members else 0 for s in SEEDS]
        rates = np.asarray(supports, dtype=float) / N_ANCHORS * 100
        records.append(
            {
                "family": family,
                "coverage": len(members),
                "supports": supports,
                "rates": rates,
                "range_pp": float(rates.max() - rates.min()),
                "mean_pct": float(rates.mean()),
                "cv": float(np.std(rates) / np.mean(rates)) if np.mean(rates) else np.nan,
            }
        )
    records.sort(key=lambda r: r["range_pp"])

    stats_path = RESULTS / "activation_range_by_family.csv"
    with stats_path.open("w", newline="") as handle:
        fields = ["family_id", "seed_coverage", *[f"seed_{s}_activation_pct" for s in SEEDS], "mean_activation_pct", "range_pp", "coefficient_of_variation"]
        writer = csv.DictWriter(handle, fieldnames=fields, lineterminator="\n")
        writer.writeheader()
        for r in records:
            writer.writerow(
                {
                    "family_id": r["family"],
                    "seed_coverage": r["coverage"],
                    **{f"seed_{s}_activation_pct": f"{r['rates'][i]:.6f}" for i, s in enumerate(SEEDS)},
                    "mean_activation_pct": f"{r['mean_pct']:.6f}",
                    "range_pp": f"{r['range_pp']:.6f}",
                    "coefficient_of_variation": f"{r['cv']:.6f}",
                }
            )

    five = [r for r in records if r["coverage"] == 5]
    chosen = {
        "Most consistent": five[0],
        "Typical (median range)": five[len(five) // 2],
        "Largest range (5/5)": five[-1],
        "Roadside-greenery example": next(r for r in records if r["family"] == "stable_0122"),
    }
    selected_ids = {r["family"] for r in chosen.values()}

    fig, ax = plt.subplots(figsize=(18, 7))
    fig.subplots_adjust(left=0.07, right=0.985, bottom=0.12, top=0.82)
    x = np.arange(len(records))
    bars = ax.bar(
        x,
        [r["range_pp"] for r in records],
        width=1.0,
        color=["#39d6ff" if r["coverage"] == 5 else "#f6c945" for r in records],
        alpha=0.84,
        linewidth=0,
    )
    for i, r in enumerate(records):
        if r["family"] in selected_ids:
            bars[i].set_color("#ff7488")
            bars[i].set_alpha(1)
            ax.annotate(r["family"], (i, r["range_pp"]), xytext=(0, 7), textcoords="offset points", rotation=65, ha="left", fontsize=8)
    fig.suptitle("Cross-seed activation-coverage range for every stable visual-gene family", x=0.07, y=0.965, ha="left", fontsize=17, weight="bold")
    fig.text(0.07, 0.915, "Range = max(seed activation rate) − min(seed activation rate), evaluated on the same 250,000 anchor patches", color="#58677c", fontsize=10)
    ax.set_xlabel("Stable families sorted by activation-rate range")
    ax.set_ylabel("Activation-rate range (percentage points)")
    ax.set_xticks([])
    ax.spines[["top", "right"]].set_visible(False)
    ax.grid(axis="y", alpha=0.22)
    ax.scatter([], [], color="#39d6ff", label=f"5/5 seeds (n={sum(r['coverage']==5 for r in records)})")
    ax.scatter([], [], color="#f6c945", label=f"4/5 seeds (n={sum(r['coverage']==4 for r in records)})")
    ax.scatter([], [], color="#ff7488", label="Families used in montage")
    ax.legend(frameon=False, ncol=3, loc="upper left")
    fig.savefig(FIGURES / "sae_seed_activation_range_bars.png", dpi=220, facecolor="white")
    plt.close(fig)

    fig, axes = plt.subplots(4, 5, figsize=(18, 12), sharex=True, sharey=True, constrained_layout=True)
    for row_idx, (role, record) in enumerate(chosen.items()):
        family = record["family"]
        members = families[family]
        all_values = [activation_values(seed_data[s], members[s]) for s in SEEDS if s in members]
        upper = max(float(np.quantile(v, 0.995)) for v in all_values if len(v))
        bins = np.linspace(0, upper, 32)
        for col_idx, seed in enumerate(SEEDS):
            ax = axes[row_idx, col_idx]
            gene = members.get(seed)
            ax.set_facecolor("#f7f9fc")
            if gene is None:
                ax.text(0.5, 0.52, "No strict match", ha="center", va="center", transform=ax.transAxes, color="#c44e52", weight="bold")
                support = 0
            else:
                vals = activation_values(seed_data[seed], gene)
                support = len(vals)
                ax.hist(vals, bins=bins, density=True, color=COLORS[seed], alpha=0.86, edgecolor="none")
                ax.axvline(float(np.median(vals)), color="#202b3c", lw=1, ls="--")
                ax.text(0.97, 0.95, f"Gene {gene}\n{support:,} patches\n{100*support/N_ANCHORS:.3f}%", ha="right", va="top", transform=ax.transAxes, fontsize=8)
            if row_idx == 0:
                ax.set_title(f"Seed {seed}", color=COLORS[seed], weight="bold")
            if col_idx == 0:
                ax.set_ylabel(f"{role}\n{family}\nDensity", fontsize=9, weight="bold")
            if row_idx == 3:
                ax.set_xlabel("Activation magnitude")
            ax.spines[["top", "right"]].set_visible(False)
            ax.grid(axis="y", alpha=0.15)
        axes[row_idx, 4].text(
            0.97,
            0.58,
            f"coverage {record['coverage']}/5\nmean rate {record['mean_pct']:.3f}%\nrange {record['range_pp']:.3f} pp",
            ha="right",
            va="top",
            transform=axes[row_idx, 4].transAxes,
            fontsize=8,
            color="#58677c",
        )
    fig.suptitle("Four representative stable families: activation distributions across five SAE seeds", x=0.01, ha="left", fontsize=18, weight="bold")
    fig.savefig(FIGURES / "sae_seed_representative_gene_montage.png", dpi=220, facecolor="white")
    plt.close(fig)

    print(f"wrote {stats_path}")
    print(f"wrote {FIGURES / 'sae_seed_activation_range_bars.png'}")
    print(f"wrote {FIGURES / 'sae_seed_representative_gene_montage.png'}")
    for role, r in chosen.items():
        print(role, r["family"], r["coverage"], f"range={r['range_pp']:.6f} pp")


if __name__ == "__main__":
    main()
