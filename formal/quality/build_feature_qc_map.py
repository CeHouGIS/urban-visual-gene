"""Recover patch-row filters from the deterministic formal pano sample.

The original feature shards did not store pano IDs.  Their row order is,
however, deterministic: retained pano order, four headings, then 784 patches.
This tool verifies that order using the old gate decision and only emits a
row filter when the reconstructed retained pano count equals the cached shard.
"""
from __future__ import annotations

import csv
import json
from collections import defaultdict
from pathlib import Path

import numpy as np

HEADINGS = {"0", "90", "180", "270"}
ROWS_PER_PANO = 4 * 784


def pano_key(path: str) -> str:
    stem = Path(path).stem
    tail = stem.rsplit("_", 1)
    return tail[0] if len(tail) == 2 and tail[1] in HEADINGS else stem


def main() -> None:
    ap = __import__("argparse").ArgumentParser()
    ap.add_argument("--sample-dir", required=True)
    ap.add_argument("--feature-dir", required=True)
    ap.add_argument("--output", required=True)
    args = ap.parse_args()
    sample_dir, feature_dir = Path(args.sample_dir), Path(args.feature_dir)

    # Keep image-level flags from the completed sharded QC pass.
    flags = defaultdict(lambda: defaultdict(lambda: [False, 0]))
    for path in sorted(sample_dir.glob("qc_part*/images.part0000.csv")):
        with path.open(newline="") as source:
            for row in csv.DictReader(source):
                key = (row["dataset"], pano_key(row["path"]))
                state = flags[key[0]][key[1]]
                state[0] |= row["black"] == "1"
                state[1] += int(row["tunnel"] == "1")

    # Reconstruct the exact path order used for feature extraction.
    order = defaultdict(list)
    for path in sorted(sample_dir.glob("formal_sample_images.part*.tsv")):
        with path.open() as source:
            for line in source:
                parts = line.rstrip("\n").split("\t", 1)
                if len(parts) == 2:
                    city, image = parts
                    pano = pano_key(image)
                    if not order[city] or order[city][-1] != pano:
                        order[city].append(pano)

    result = {"version": 1, "rows_per_pano": ROWS_PER_PANO, "cities": {}, "unsafe": []}
    for city, panos in sorted(order.items()):
        feature = feature_dir / f"dict_{city}_p784.f16.npy"
        if not feature.exists():
            continue
        cached_panos = int(np.load(feature, mmap_mode="r").shape[0] // ROWS_PER_PANO)
        # The old gate rejected black in any view and tunnel only when at
        # least two views were tunnel-like.
        old_kept = [p for p in panos if not flags[city][p][0] and flags[city][p][1] < 2]
        new_bad = {p for p in panos if flags[city][p][0] or flags[city][p][1] > 0}
        if len(old_kept) != cached_panos:
            result["unsafe"].append(city)
            result["cities"][city] = {"verified": False, "sample_panos": len(panos),
                                       "old_kept_panos": len(old_kept),
                                       "cached_panos": cached_panos,
                                       "new_bad_panos": len(new_bad)}
            continue
        kept_positions = {p: i for i, p in enumerate(old_kept)}
        bad_positions = sorted(kept_positions[p] for p in new_bad if p in kept_positions)
        result["cities"][city] = {"verified": True, "sample_panos": len(panos),
                                   "old_kept_panos": len(old_kept),
                                   "cached_panos": cached_panos,
                                   "new_bad_panos": len(bad_positions),
                                   "bad_kept_positions": bad_positions}
    result["verified_cities"] = sum(v.get("verified", False) for v in result["cities"].values())
    result["unsafe"] = sorted(result["unsafe"])
    Path(args.output).write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n")
    print(json.dumps({"cities": len(result["cities"]), "verified": result["verified_cities"],
                      "unsafe": result["unsafe"]}, ensure_ascii=False))


if __name__ == "__main__":
    main()
