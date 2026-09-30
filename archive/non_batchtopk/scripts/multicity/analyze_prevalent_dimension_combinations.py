#!/usr/bin/env python3
"""Find the most prevalent within-image pairs and triples of latent dimensions."""
from __future__ import annotations

import scripts._env  # noqa: F401

import argparse
import heapq
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from scripts.multicity.config import CITIES


ROOT = Path("outputs/experiments/dinov3_multicity/feature_mae_n30x12800_qc")
DEFAULT_TOP = ROOT / "mae/hierarchy_32_64/top1_dimensions.npy"
DEFAULT_CITY = ROOT / "mae/hierarchy_32_64/city_indices.npy"
DEFAULT_HIERARCHY = ROOT / "mae/hierarchy_edp_32_64/hierarchy_arrays.npz"


def configure_plotting() -> None:
    plt.rcParams.update(
        {
            "font.family": "sans-serif",
            "font.sans-serif": ["Arial", "Helvetica", "Liberation Sans", "DejaVu Sans"],
            "font.size": 7,
            "axes.labelsize": 7,
            "axes.titlesize": 8,
            "xtick.labelsize": 6,
            "ytick.labelsize": 6,
            "axes.linewidth": 0.5,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
        }
    )


def palette32() -> np.ndarray:
    colours = []
    for name in ("tab20", "tab20b", "tab20c"):
        colours.extend(plt.get_cmap(name)(np.linspace(0, 1, 20))[:, :3])
    return np.asarray(colours[:32])


def top_pairs(cooccurrence: np.ndarray, number: int) -> list[tuple[int, ...]]:
    left, right = np.triu_indices(len(cooccurrence), 1)
    values = cooccurrence[left, right]
    selected = np.argsort(values)[-number:][::-1]
    return [(int(left[index]), int(right[index])) for index in selected]


def candidate_presence(
    top: np.ndarray,
    ranked_dimensions: np.ndarray,
    chunk_size: int,
) -> np.ndarray:
    presence = np.empty((len(top), len(ranked_dimensions)), dtype=bool)
    lookup = np.full(512, -1, dtype=np.int16)
    lookup[ranked_dimensions] = np.arange(len(ranked_dimensions), dtype=np.int16)
    for start in range(0, len(top), chunk_size):
        winners = np.asarray(top[start : start + chunk_size], dtype=np.int64)
        rows = np.repeat(np.arange(len(winners), dtype=np.int64), winners.shape[1])
        counts = np.bincount(
            rows * 512 + winners.ravel(), minlength=len(winners) * 512
        ).reshape(len(winners), 512)
        presence[start : start + len(winners)] = counts[:, ranked_dimensions] >= 4
    return presence


def bitsets(presence: np.ndarray) -> list[int]:
    values = []
    for column in range(presence.shape[1]):
        packed = np.packbits(presence[:, column], bitorder="little")
        values.append(int.from_bytes(packed.tobytes(), byteorder="little", signed=False))
    return values


def integer_popcount(value: int) -> int:
    """Return the population count on both Python 3.8 and newer runtimes."""
    method = getattr(value, "bit_count", None)
    return method() if method is not None else bin(value).count("1")


def best_triples(
    presence: np.ndarray,
    ranked_dimensions: np.ndarray,
    occurrence: np.ndarray,
    number: int,
) -> tuple[list[tuple[int, ...]], int, bool]:
    packed = bitsets(presence)
    maximum = min(64, len(ranked_dimensions))
    verified = False
    best: list[tuple[int, tuple[int, int, int]]] = []
    while True:
        heap: list[tuple[int, tuple[int, int, int]]] = []
        for first in range(maximum - 2):
            first_bits = packed[first]
            for second in range(first + 1, maximum - 1):
                pair_bits = first_bits & packed[second]
                for third in range(second + 1, maximum):
                    support = integer_popcount(pair_bits & packed[third])
                    item = (support, (first, second, third))
                    if len(heap) < number:
                        heapq.heappush(heap, item)
                    elif item > heap[0]:
                        heapq.heapreplace(heap, item)
        best = sorted(heap, reverse=True)
        cutoff = best[-1][0]
        # Any triple containing a dimension outside the candidate set has
        # support no greater than that dimension's marginal occurrence.
        next_occurrence = (
            occurrence[ranked_dimensions[maximum]] if maximum < len(ranked_dimensions) else -1
        )
        if next_occurrence < cutoff:
            verified = True
            break
        if maximum >= min(128, len(ranked_dimensions)):
            break
        maximum = min(maximum + 16, 128, len(ranked_dimensions))
    triples = [tuple(int(ranked_dimensions[x]) for x in local) for _, local in best]
    return triples, maximum, verified


def combination_statistics(
    top: np.ndarray,
    city_ids: np.ndarray,
    combinations: list[tuple[int, ...]],
    fine: np.ndarray,
    coarse: np.ndarray,
    chunk_size: int,
) -> pd.DataFrame:
    city_counts = np.bincount(city_ids, minlength=len(CITIES)).astype(np.float64)
    city_weight = len(top) / (len(CITIES) * city_counts)
    supports = np.zeros(len(combinations), dtype=np.int64)
    weighted_supports = np.zeros(len(combinations), dtype=np.float64)
    patch_sums = [np.zeros(len(combo), dtype=np.float64) for combo in combinations]
    city_support = np.zeros((len(combinations), len(CITIES)), dtype=np.int64)

    for start in range(0, len(top), chunk_size):
        winners = np.asarray(top[start : start + chunk_size], dtype=np.int64)
        rows = np.repeat(np.arange(len(winners), dtype=np.int64), winners.shape[1])
        counts = np.bincount(
            rows * 512 + winners.ravel(), minlength=len(winners) * 512
        ).reshape(len(winners), 512)
        block_cities = np.asarray(city_ids[start : start + len(winners)], dtype=np.int64)
        weights = city_weight[block_cities]
        for index, combo in enumerate(combinations):
            chosen = counts[:, combo]
            mask = np.all(chosen >= 4, axis=1)
            if not np.any(mask):
                continue
            supports[index] += int(mask.sum())
            weighted_supports[index] += float(weights[mask].sum())
            patch_sums[index] += chosen[mask].sum(axis=0)
            city_support[index] += np.bincount(block_cities[mask], minlength=len(CITIES))

    rows = []
    for index, combo in enumerate(combinations):
        support = int(supports[index])
        shares = patch_sums[index] / max(support * 196, 1)
        relative = shares / max(shares.sum(), 1e-12)
        row = {
            "combination_size": len(combo),
            "dimensions": "|".join(f"D{x:03d}" for x in combo),
            "fine_clusters": "|".join(f"F{fine[x]:03d}" for x in combo),
            "coarse_clusters": "|".join(f"C{coarse[x]:03d}" for x in combo),
            "cooccurrence_count": support,
            "image_prevalence": support / len(top),
            "city_balanced_prevalence": weighted_supports[index] / len(top),
            "combined_mean_patch_share": float(shares.sum()),
            "cities_with_at_least_10_images": int(np.sum(city_support[index] >= 10)),
        }
        for member in range(3):
            row[f"dimension_{member + 1}"] = (
                f"D{combo[member]:03d}" if member < len(combo) else ""
            )
            row[f"mean_patch_share_{member + 1}"] = (
                float(shares[member]) if member < len(combo) else np.nan
            )
            row[f"relative_share_{member + 1}"] = (
                float(relative[member]) if member < len(combo) else np.nan
            )
        rows.append(row)
    frame = pd.DataFrame(rows)
    frame["rank_within_size"] = frame.groupby("combination_size")["cooccurrence_count"].rank(
        method="first", ascending=False
    ).astype(int)
    return frame.sort_values(["combination_size", "rank_within_size"])


def plot(frame: pd.DataFrame, coarse: np.ndarray, output: Path) -> None:
    colours = palette32()
    fig, axes = plt.subplots(2, 2, figsize=(7.2, 6.0), constrained_layout=True)
    for column, size in enumerate((2, 3)):
        part = frame[frame.combination_size == size].sort_values("cooccurrence_count").tail(10)
        labels = part.dimensions.str.replace("|", " + ", regex=False)
        y = np.arange(len(part))
        prevalence = 100 * part.city_balanced_prevalence.to_numpy()
        axes[0, column].barh(y, prevalence, color="#0072B2" if size == 2 else "#009E73")
        axes[0, column].set(
            yticks=y,
            yticklabels=labels,
            xlabel="Images containing combination (%)",
            title="Pairs" if size == 2 else "Triples",
        )
        for yy, value in zip(y, prevalence):
            axes[0, column].text(value + 0.25, yy, f"{value:.1f}%", va="center", fontsize=5.2)
        axes[0, column].set_xlim(0, prevalence.max() * 1.18)

        left = np.zeros(len(part), dtype=np.float64)
        for member in range(size):
            widths = 100 * part[f"mean_patch_share_{member + 1}"].to_numpy()
            dimension_indices = part[f"dimension_{member + 1}"].str[1:].astype(int).to_numpy()
            bar_colours = colours[coarse[dimension_indices]]
            axes[1, column].barh(y, widths, left=left, color=bar_colours,
                                 edgecolor="white", linewidth=0.25)
            for row_index, (start, width, dimension) in enumerate(
                zip(left, widths, part[f"dimension_{member + 1}"])
            ):
                if width >= 1.25:
                    axes[1, column].text(
                        start + width / 2, row_index, dimension,
                        ha="center", va="center", fontsize=4.2, color="white",
                    )
            left += widths
        axes[1, column].set(
            yticks=y,
            yticklabels=labels,
            xlabel="Mean image area represented by combination (%)",
        )
    for label, ax in zip("abcd", axes.flat):
        ax.text(-0.16, 1.05, label, transform=ax.transAxes, fontweight="bold", fontsize=9)
        ax.spines[["top", "right"]].set_visible(False)
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, dpi=400, facecolor="white")
    fig.savefig(output.with_suffix(".pdf"), facecolor="white")
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--top", type=Path, default=DEFAULT_TOP)
    parser.add_argument("--city-indices", type=Path, default=DEFAULT_CITY)
    parser.add_argument("--hierarchy", type=Path, default=DEFAULT_HIERARCHY)
    parser.add_argument("--cooccurrence", type=Path, default=Path("results/dimension_cooccurrence.csv"))
    parser.add_argument("--dimension-statistics", type=Path,
                        default=Path("results/dimension_image_statistics.csv"))
    parser.add_argument("--semantic-labels", type=Path,
                        default=Path("results/semantic_labels_fine.json"))
    parser.add_argument("--output-table", type=Path,
                        default=Path("results/prevalent_dimension_combinations.csv"))
    parser.add_argument("--output-figure", type=Path,
                        default=Path("paper/figures/supplementary/Fig_Prevalent_Dimension_Combinations.png"))
    parser.add_argument("--number", type=int, default=10)
    parser.add_argument("--candidate-dimensions", type=int, default=128)
    parser.add_argument("--chunk-size", type=int, default=4000)
    args = parser.parse_args()

    configure_plotting()
    top = np.load(args.top, mmap_mode="r")
    city_ids = np.load(args.city_indices, mmap_mode="r")
    with np.load(args.hierarchy) as hierarchy:
        fine = hierarchy["fine_labels"].astype(np.int64)
        coarse = hierarchy["coarse_labels"].astype(np.int64)
    cooccurrence = pd.read_csv(args.cooccurrence, index_col=0).to_numpy(np.int64)
    dimension_stats = pd.read_csv(args.dimension_statistics)
    occurrence = dimension_stats.image_occurrence_count.to_numpy(np.int64)
    ranked = np.argsort(occurrence)[::-1]

    pairs = top_pairs(cooccurrence, args.number)
    candidate_count = min(args.candidate_dimensions, len(ranked))
    presence = candidate_presence(top, ranked[:candidate_count], args.chunk_size)
    triples, searched, verified = best_triples(
        presence, ranked[:candidate_count], occurrence, args.number
    )
    combinations = pairs + triples
    frame = combination_statistics(
        top, city_ids, combinations, fine, coarse, args.chunk_size
    )
    if args.semantic_labels.is_file():
        payload = json.loads(args.semantic_labels.read_text())
        semantic_map = {
            item["category_id"]: item.get("name_en", item.get("short_label", ""))
            for item in payload.get("categories", [])
        }
        frame.insert(
            frame.columns.get_loc("fine_clusters") + 1,
            "fine_semantic_labels",
            [
                "|".join(semantic_map.get(value, "") for value in values.split("|"))
                for values in frame.fine_clusters
            ],
        )
    args.output_table.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(args.output_table, index=False)
    plot(frame, coarse, args.output_figure)

    report = {
        "images": len(top),
        "presence_definition": "dimension wins at least 4 of 196 patches",
        "pairs_reported": len(pairs),
        "triples_reported": len(triples),
        "candidate_dimensions_loaded": candidate_count,
        "candidate_dimensions_searched": searched,
        "top_triple_search_complete": verified,
        "triple_completeness_rule": (
            "the next excluded dimension occurs less often than the tenth triple"
        ),
    }
    args.output_table.with_suffix(".report.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2), flush=True)


if __name__ == "__main__":
    main()
