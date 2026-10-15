"""Draw comparable Sankey panels for held-out dictionary-width relations."""
from __future__ import annotations

from sae_experiments.paths import REPO_ROOT

import argparse
import csv
import json
from collections import Counter
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.patches import PathPatch, Rectangle
from matplotlib.path import Path as MplPath


DATA_DIR = REPO_ROOT / "reports/data/scaling_heldout"
OUT_DIR = REPO_ROOT / "reports/figures/scaling_heldout"
CATEGORIES = ("stable", "split_child", "no_close_match", "related_or_reconfigured", "insufficient_support")
LABELS = {
    "stable": "Stable",
    "split_child": "Split descendant",
    "no_close_match": "No close match",
    "related_or_reconfigured": "Related / reorganized",
    "insufficient_support": "Insufficient support",
    "stable_only_parent": "Stable-only parent",
    "split_parent": "Parent with split",
    "unattributed": "No assigned parent",
}
COLORS = {
    "stable": "#218579",
    "split_child": "#d99b36",
    "no_close_match": "#b64b5e",
    "related_or_reconfigured": "#6885a8",
    "insufficient_support": "#b3bac0",
    "stable_only_parent": "#218579",
    "split_parent": "#d99b36",
    "unattributed": "#d2d7da",
}


def load_rows(data_dir: Path, left: int, right: int) -> list[dict]:
    path = data_dir / f"feature_relations_w{left}_to_w{right}.csv"
    with path.open(newline="") as source:
        return list(csv.DictReader(source))


def assigned_parents(relations: dict) -> dict[int, int]:
    """Credit each matched child to one documented parent, favoring stable links."""
    assignment = {row["child"]: row["parent"] for row in relations["stable"]}
    split_scores = {}
    for row in relations["split_candidates"]:
        for child, recall, cosine in zip(row["children"], row["parent_recalls"], row["decoder_cosines"]):
            if child in assignment:
                continue
            score = recall * cosine
            if child not in split_scores or score > split_scores[child][0]:
                split_scores[child] = (score, row["parent"])
    assignment.update({child: parent for child, (_, parent) in split_scores.items()})
    return assignment


def flow_counts(data_dir: Path, left: int, right: int, previous: list[dict] | None = None) -> Counter:
    rows = load_rows(data_dir, left, right)
    relations = json.loads((data_dir / f"relations_w{left}_to_w{right}.json").read_text())
    parents = assigned_parents(relations)
    split_parents = {row["parent"] for row in relations["split_candidates"]}
    flows = Counter()
    for row in rows:
        child, category = int(row["feature"]), row["category"]
        if category in ("stable", "split_child"):
            if child not in parents:
                raise ValueError(f"missing documented parent for W{right} feature {child}")
            parent = parents[child]
            source = previous[parent]["category"] if previous is not None else (
                "split_parent" if parent in split_parents else "stable_only_parent"
            )
        else:
            source = "unattributed"
        flows[source, category] += 1
    if sum(flows.values()) != right:
        raise ValueError(f"W{right} flow does not conserve child counts")
    return flows


def positions(keys: list[str], totals: Counter, unit: float, gap: float):
    full_height = sum(totals[key] * unit for key in keys) + gap * (len(keys) - 1)
    top = .50 + full_height / 2
    result = {}
    for key in keys:
        height = totals[key] * unit
        result[key] = (top - height, top)
        top -= height + gap
    return result


def ribbon(ax, x0, x1, source_bottom, source_top, target_bottom, target_top, color):
    delta = (x1 - x0) * .48
    vertices = [
        (x0, source_top), (x0 + delta, source_top), (x1 - delta, target_top), (x1, target_top),
        (x1, target_bottom), (x1 - delta, target_bottom), (x0 + delta, source_bottom),
        (x0, source_bottom), (x0, source_top),
    ]
    codes = [MplPath.MOVETO, MplPath.CURVE4, MplPath.CURVE4, MplPath.CURVE4,
             MplPath.LINETO, MplPath.CURVE4, MplPath.CURVE4, MplPath.CURVE4, MplPath.CLOSEPOLY]
    ax.add_patch(PathPatch(MplPath(vertices, codes), facecolor=color, edgecolor="none", alpha=.74))


def spaced_labels(centers: list[float], minimum_gap: float = .071) -> list[float]:
    ordered = sorted(enumerate(centers), key=lambda pair: pair[1], reverse=True)
    placed = []
    prior = .96
    for index, center in ordered:
        value = min(center, prior - minimum_gap)
        placed.append((index, value))
        prior = value
    low = min(value for _, value in placed)
    if low < .055:
        placed = [(index, value + .055 - low) for index, value in placed]
    return [value for _, value in sorted(placed)]


def draw_panel(ax, flows: Counter, sources: list[str], target_width: int, left_title: str, right_title: str):
    destinations = list(CATEGORIES)
    source_totals = Counter({key: sum(count for (source, _), count in flows.items() if source == key) for key in sources})
    target_totals = Counter({key: sum(count for (_, target), count in flows.items() if target == key) for key in destinations})
    sources = [key for key in sources if source_totals[key]]
    destinations = [key for key in destinations if target_totals[key]]
    gap = .021
    unit = (.82 - gap * (max(len(sources), len(destinations)) - 1)) / target_width
    left = positions(sources, source_totals, unit, gap)
    right = positions(destinations, target_totals, unit, gap)
    left_top = {key: left[key][1] for key in sources}
    right_top = {key: right[key][1] for key in destinations}
    x0, x1, bar_width = .24, .77, .017
    for source in sources:
        for target in destinations:
            count = flows[source, target]
            if not count:
                continue
            height = count * unit
            source_top = left_top[source]
            target_top = right_top[target]
            ribbon(ax, x0 + bar_width, x1, source_top - height, source_top,
                   target_top - height, target_top, COLORS[target])
            left_top[source] -= height
            right_top[target] -= height
    for keys, bounds, totals, x, align, label_x in (
        (sources, left, source_totals, x0, "right", x0 - .015),
        (destinations, right, target_totals, x1, "left", x1 + bar_width + .015),
    ):
        centers = [(bounds[key][0] + bounds[key][1]) / 2 for key in keys]
        label_ys = spaced_labels(centers)
        for key, center, label_y in zip(keys, centers, label_ys):
            bottom, top = bounds[key]
            ax.add_patch(Rectangle((x, bottom), bar_width, top - bottom,
                                   facecolor=COLORS[key], edgecolor="white", linewidth=.4, zorder=4))
            if abs(center - label_y) > .012:
                start = x if align == "right" else x + bar_width
                ax.plot([start, label_x], [center, label_y], color="#7d8587", lw=.7)
            ax.text(label_x, label_y, f"{LABELS[key]}  {totals[key]:,}",
                    ha=align, va="center", fontsize=10, color="#263536")
    ax.text(.25, .965, left_title, ha="center", va="bottom", fontsize=11, weight="bold", color="#263536")
    ax.text(.78, .965, right_title, ha="center", va="bottom", fontsize=11, weight="bold", color="#263536")
    ax.set_xlim(0, 1.17)
    ax.set_ylim(0, 1.06)
    ax.axis("off")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, default=DATA_DIR)
    parser.add_argument("--outdir", type=Path, default=OUT_DIR)
    args = parser.parse_args()
    args.outdir.mkdir(parents=True, exist_ok=True)
    first_rows = load_rows(args.data_dir, 512, 1024)
    first = flow_counts(args.data_dir, 512, 1024)
    second = flow_counts(args.data_dir, 1024, 2048, first_rows)
    summary = json.loads((args.data_dir / "summary.json").read_text())
    for flows, left, right in ((first, 512, 1024), (second, 1024, 2048)):
        expected = dict(summary["pairs"][f"w{left}_to_w{right}"]["child_counts"])
        expected["split_child"] = expected.pop("split_child_not_stable")
        for category, count in expected.items():
            assert sum(value for (_, target), value in flows.items() if target == category) == count
    assert second["stable", "stable"] == summary["stable_three_width_chains"]

    figure = plt.figure(figsize=(16, 12), facecolor="white")
    figure.suptitle("How street-view features change as BatchTopK dictionaries widen",
                     x=.50, y=.987, fontsize=19, weight="bold", color="#263536")
    figure.text(.50, .947, "Ribbon width = number of features in the larger dictionary; each panel is normalized separately.",
                ha="center", fontsize=11, color="#5a6768")
    top = figure.add_axes((.065, .53, .88, .36))
    bottom = figure.add_axes((.065, .105, .88, .36))
    draw_panel(top, first, ["stable_only_parent", "split_parent", "unattributed"], 1024,
               "W=512  |  parent class of W=1024 features", "W=1024  |  feature class")
    draw_panel(bottom, second, list(CATEGORIES) + ["unattributed"], 2048,
               "W=1024  |  parent class of W=2048 features", "W=2048  |  feature class")
    figure.text(.08, .508, "01  |  512  →  1,024", fontsize=15, weight="bold", color="#263536")
    figure.text(.08, .083, "02  |  1,024  →  2,048", fontsize=15, weight="bold", color="#263536")
    figure.text(.50, .042,
                f"{summary['stable_three_width_chains']} stable W512→W1024→W2048 chains. Split parents may produce multiple children. "
                "'No assigned parent' includes all features without a verified single lineage.",
                ha="center", fontsize=10, color="#5a6768")
    png = args.outdir / "dictionary_scaling_sankey.png"
    svg = args.outdir / "dictionary_scaling_sankey.svg"
    figure.savefig(png, dpi=200, facecolor="white")
    figure.savefig(svg, facecolor="white")
    plt.close(figure)
    path = args.data_dir / "dictionary_scaling_sankey_flows.csv"
    with path.open("w", newline="") as output:
        writer = csv.writer(output)
        writer.writerow(("comparison", "source", "destination", "larger_dictionary_features"))
        for comparison, flows in (("512_to_1024", first), ("1024_to_2048", second)):
            for (source, target), count in sorted(flows.items()):
                writer.writerow((comparison, source, target, count))
    print(f"Saved {png}, {svg}, and {path}")


if __name__ == "__main__":
    main()
