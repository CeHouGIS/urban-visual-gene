"""Plot concrete dictionary-splitting candidates from scaling analysis."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


def load_examples(result_dir: Path, left: int, right: int, n: int):
    summary = json.loads((result_dir / "scaling_summary.json").read_text())
    rows = summary["pairs"][f"w{left}_to_w{right}"]["split"]
    stats_left = np.load(result_dir / f"feature_stats_w{left}.npz")
    stats_right = np.load(result_dir / f"feature_stats_w{right}.npz")
    pair = np.load(result_dir / f"pair_metrics_w{left}_to_w{right}.npz")
    # Prefer high coverage with exactly two children: these are easiest to
    # inspect visually, while still representing the intended split pattern.
    scored = []
    for row in rows:
        targets = [int(x) for x in row["targets"]]
        if len(targets) != 2 or row["coverage"] < 0.45:
            continue
        parent = set(stats_left["top_rows"][int(row["source"])].tolist()) - {-1}
        child_sets = [set(stats_right["top_rows"][target].tolist()) - {-1} for target in targets]
        child_cover = [len(parent & child) / max(1, len(parent)) for child in child_sets]
        overlap = len(parent & child_sets[0] & child_sets[1]) / max(1, len(parent))
        # A convincing split has two substantial, mostly distinct child
        # supports.  A child covering 90% while the other covers 5% is a
        # near-duplicate candidate and is deliberately ranked lower.
        balance_score = min(child_cover) * (1.0 - overlap) * float(row["coverage"])
        scored.append((balance_score, row, child_cover, overlap))
    scored.sort(key=lambda item: item[0], reverse=True)
    chosen = [item[1] for item in scored[:n]]
    return chosen, stats_left, stats_right, pair


def add_example(ax, row, left_stats, right_stats, pair, left_width, right_width):
    source = int(row["source"])
    targets = [int(x) for x in row["targets"]]
    parent_rows = left_stats["top_rows"][source]
    parent_rows = parent_rows[parent_rows >= 0]
    matrix = np.zeros((len(parent_rows), len(targets)), dtype=np.float32)
    for col, target in enumerate(targets):
        child_rows = right_stats["top_rows"][target]
        matrix[:, col] = np.isin(parent_rows, child_rows).astype(np.float32)
    im = ax.imshow(matrix, aspect="auto", interpolation="nearest", cmap="Blues", vmin=0, vmax=1)
    ax.set_xticks(range(len(targets)))
    ax.set_xticklabels([f"child {x}" for x in targets], fontsize=8)
    ax.set_yticks(range(len(parent_rows)))
    ax.set_yticklabels([str(x) for x in parent_rows], fontsize=6)
    ax.set_xlabel("larger dictionary feature", fontsize=8)
    ax.set_ylabel("parent top-activation row", fontsize=8)
    coverage = float(row["coverage"])
    ax.set_title(f"parent {source} -> {len(targets)} children | union coverage {coverage:.0%}", fontsize=10, loc="left")
    for col, target in enumerate(targets):
        c = float(pair["decoder_cosine"][source, target])
        o = float(pair["top_row_overlap"][source, target])
        ax.text(col, len(parent_rows) + 0.7, f"cos {c:.2f}\noverlap {o:.0%}", ha="center", va="top", fontsize=7)
    ax.set_ylim(len(parent_rows) + 2.8, -0.5)
    return im


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--result-dir", required=True)
    p.add_argument("--out", required=True)
    p.add_argument("--left", type=int, default=512)
    p.add_argument("--right", type=int, default=1024)
    p.add_argument("--examples", type=int, default=3)
    args = p.parse_args()
    result_dir = Path(args.result_dir)
    examples, left_stats, right_stats, pair = load_examples(result_dir, args.left, args.right, args.examples)
    if not examples:
        raise RuntimeError("no split candidates passed the coverage threshold")
    fig, axes = plt.subplots(len(examples), 1, figsize=(10, max(3.6, 3.1 * len(examples))), squeeze=False)
    for ax, row in zip(axes[:, 0], examples):
        add_example(ax, row, left_stats, right_stats, pair, args.left, args.right)
    fig.suptitle(
        f"Dictionary split candidates: W={args.left} -> W={args.right}\n"
        "blue cells show which child feature also ranks each parent high-activation row",
        fontsize=12,
    )
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=180, bbox_inches="tight")
    print(f"saved {out}")
    for row in examples:
        print(json.dumps(row, sort_keys=True))


if __name__ == "__main__":
    main()
