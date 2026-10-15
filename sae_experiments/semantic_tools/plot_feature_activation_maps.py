"""Project dictionary feature responses back onto original street-view images.

The training cache contains contextual 28x28 patch features rather than image
paths.  This script reconstructs the deterministic city-stripe cache mapping,
recovers the source pano and heading, and evaluates ReLU pre-activations on all
784 patches of that image.  Pre-activations are used instead of ``encode`` so
the visualization is independent of BatchTopK's batch-level selection.
"""
from __future__ import annotations

import argparse
import bisect
import csv
import json
from collections import defaultdict
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
from PIL import Image

from sae_experiments.models.base_sae import build_sae


ROWS_PER_PANO = 4 * 28 * 28
PATCHES_PER_VIEW = 28 * 28
HEADINGS = (0, 90, 180, 270)


def load_checkpoint(path: Path, device: torch.device):
    meta = torch.load(path, map_location="cpu", weights_only=False)
    model = build_sae(
        input_dim=int(meta["D"]),
        config={
            "type": meta.get("model_type", "batch_topk"),
            "latent_dim": int(meta["K"]),
            "k": int(meta.get("k", meta.get("topk", 8))),
            "decoder_unit_norm": True,
        },
    ).to(device)
    model.load_state_dict(meta["state"])
    model.eval()
    return model


def pano_id(path: str) -> tuple[str, int | None]:
    stem = Path(path).stem
    if "_" not in stem:
        return stem, None
    value, suffix = stem.rsplit("_", 1)
    try:
        return value, int(suffix)
    except ValueError:
        return stem, None


def load_image_catalog(qc_root: Path):
    orders: dict[str, list[str]] = defaultdict(list)
    paths: dict[str, dict[str, dict[int | None, str]]] = defaultdict(dict)
    seen: dict[str, set[str]] = defaultdict(set)
    for path in sorted(qc_root.glob("formal_sample_images.part*.tsv")):
        with path.open() as source:
            for line in source:
                city, image = line.rstrip("\n").split("\t", 1)
                pid, heading = pano_id(image)
                if pid not in seen[city]:
                    seen[city].add(pid)
                    orders[city].append(pid)
                paths[city].setdefault(pid, {})[heading] = image

    # Reconstruct the old feature-extraction gate.  Feature shards retain this
    # order, then the newer QC map optionally removes whole pano blocks.
    flags = defaultdict(lambda: defaultdict(lambda: [False, 0]))
    for path in sorted(qc_root.glob("qc_part*/images.part*.csv")):
        with path.open(newline="") as source:
            for row in csv.DictReader(source):
                city = row["dataset"]
                pid, _ = pano_id(row["path"])
                state = flags[city][pid]
                state[0] |= row.get("black", "0") == "1"
                state[1] += int(row.get("tunnel", "0") == "1")
    old_kept = {
        city: [pid for pid in ids if not flags[city][pid][0] and flags[city][pid][1] < 2]
        for city, ids in orders.items()
    }
    return old_kept, paths


class CacheRowMapper:
    def __init__(self, manifest_path: Path, qc_map_path: Path, qc_root: Path, rows: int):
        manifest = json.loads(manifest_path.read_text())
        qc_map = json.loads(qc_map_path.read_text())
        self.old_kept, self.paths = load_image_catalog(qc_root)
        self.cities = list(manifest["cities"])
        self.source_out = Path(manifest["source_out"])
        self.qc_cities = qc_map["cities"]
        base, extra = divmod(int(rows), len(self.cities))
        self.selected = []
        for index, city in enumerate(self.cities):
            filtered_rows = int(manifest["cities"][city]["filtered_rows"])
            take = min(filtered_rows, base + (index < extra))
            available = filtered_rows - take
            start = (index * 104729) % (available + 1) if available else 0
            self.selected.append((city, start, take))
        self.segments = []
        offsets = [0] * len(self.selected)
        global_offset = 0
        while global_offset < rows:
            advanced = False
            for city_index, (_city, _start, take) in enumerate(self.selected):
                remaining = take - offsets[city_index]
                if remaining <= 0:
                    continue
                count = min(1024, remaining)
                self.segments.append((global_offset, global_offset + count, city_index, offsets[city_index]))
                offsets[city_index] += count
                global_offset += count
                advanced = True
            if not advanced:
                break
        self.segment_starts = [row[0] for row in self.segments]

    def resolve(self, cache_row: int):
        index = bisect.bisect_right(self.segment_starts, int(cache_row)) - 1
        if index < 0:
            raise ValueError(f"cache row outside mapping: {cache_row}")
        global_start, global_stop, city_index, local_offset = self.segments[index]
        city, selected_start, _take = self.selected[city_index]
        filtered_row = selected_start + local_offset + int(cache_row) - global_start
        qc = self.qc_cities[city]
        bad = qc.get("bad_kept_positions", []) if qc.get("verified") else []
        filtered_pano = filtered_row // ROWS_PER_PANO
        old_position = filtered_pano + bisect.bisect_right(bad, filtered_pano)
        local = filtered_row % ROWS_PER_PANO
        heading_index, patch_index = divmod(local, PATCHES_PER_VIEW)
        heading = HEADINGS[heading_index]
        ids = self.old_kept.get(city, [])
        if old_position >= len(ids):
            raise IndexError(f"{city}: pano position {old_position} outside {len(ids)}")
        pid = ids[old_position]
        image = self.paths[city].get(pid, {}).get(heading)
        if image is None:
            # Equirectangular Baidu entries have no heading suffix and are not
            # used for this square-image overlay, but keep the fallback clear.
            image = self.paths[city].get(pid, {}).get(None)
        source_row = old_position * ROWS_PER_PANO + heading_index * PATCHES_PER_VIEW
        return {
            "city": city,
            "pano": pid,
            "heading": heading,
            "patch": int(patch_index),
            "image": image,
            "source_row": int(source_row),
            "feature_path": str(self.source_out / f"dict_{city}_p784.f16.npy"),
        }


def choose_example_row(stats: dict[str, np.ndarray], feature: int, mapper: CacheRowMapper):
    values = stats["top_values"][feature]
    rows = stats["top_rows"][feature]
    order = np.argsort(values)[::-1]
    for pos in order:
        row = int(rows[pos])
        if row < 0:
            continue
        info = mapper.resolve(row)
        if info["image"] and "_" in Path(info["image"]).stem:
            info["cache_row"] = row
            info["activation"] = float(values[pos])
            return info
    raise RuntimeError(f"no square GoogleSV image found for feature {feature}")


def feature_maps(info, feature_ids, model, device: torch.device):
    source = np.load(info["feature_path"], mmap_mode="r")
    start = int(info["source_row"])
    x = torch.from_numpy(np.asarray(source[start:start + PATCHES_PER_VIEW], dtype=np.float32)).to(device)
    with torch.no_grad():
        # Dense positive preactivation is stable across batch composition.
        acts = torch.relu(model._preactivate(x)).cpu().numpy()
    return [acts[:, int(feature)].reshape(28, 28) for feature in feature_ids]


def draw_pair(result_dir: Path, mapper: CacheRowMapper, left: int, right: int, examples: list[dict], model_left, model_right, out: Path, device: torch.device):
    stats_left = dict(np.load(result_dir / f"feature_stats_w{left}.npz"))
    stats_right = dict(np.load(result_dir / f"feature_stats_w{right}.npz"))
    fig, axes = plt.subplots(len(examples), 4, figsize=(13, max(3.8, 3.8 * len(examples))), squeeze=False)
    for row_index, example in enumerate(examples):
        source = int(example["source"])
        targets = [int(x) for x in example["targets"]]
        info = choose_example_row(stats_left, source, mapper)
        image = Image.open(info["image"]).convert("RGB").resize((448, 448), Image.BILINEAR)
        parent_map, = feature_maps(info, [source], model_left, device)
        child_maps = feature_maps(info, targets, model_right, device)
        maps = [parent_map, *child_maps]
        vmax = max(float(np.max(x)) for x in maps) or 1.0
        labels = [f"parent {source}", f"child {targets[0]}", f"child {targets[1]}"]
        ax = axes[row_index]
        ax[0].imshow(image)
        ax[0].set_title(f"original\n{info['city']} / {info['heading']} deg", fontsize=9)
        ax[0].axis("off")
        for panel, heatmap, label in zip(ax[1:], maps, labels):
            panel.imshow(image)
            panel.imshow(heatmap, cmap="magma", alpha=0.62, vmin=0, vmax=vmax, interpolation="bilinear", extent=(0, 448, 448, 0))
            panel.set_title(f"{label}\nReLU preactivation", fontsize=9)
            panel.axis("off")
        ax[0].text(0, -0.06, f"pano {info['pano'][:12]}... | cache row {info['cache_row']} | patch {info['patch']}", transform=ax[0].transAxes, fontsize=7)
    fig.suptitle(f"Activation maps for split candidates: W={left} -> W={right}\nDense ReLU preactivation on the same original street-view image", fontsize=12)
    fig.tight_layout(rect=(0, 0, 1, 0.95), h_pad=2.0)
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=180, bbox_inches="tight")
    print(f"saved {out}")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--result-dir", required=True)
    p.add_argument("--manifest", required=True)
    p.add_argument("--qc-map", required=True)
    p.add_argument("--qc-root", required=True)
    p.add_argument("--left-checkpoint", required=True)
    p.add_argument("--right-checkpoint", required=True)
    p.add_argument("--left", type=int, required=True)
    p.add_argument("--right", type=int, required=True)
    p.add_argument("--out", required=True)
    p.add_argument("--device", default="cuda:0")
    args = p.parse_args()
    result_dir = Path(args.result_dir)
    summary = json.loads((result_dir / "scaling_summary.json").read_text())
    split = summary["pairs"][f"w{args.left}_to_w{args.right}"]["split"]
    # Reuse the balanced examples used by the split heatmap plot.
    left_stats = dict(np.load(result_dir / f"feature_stats_w{args.left}.npz"))
    right_stats = dict(np.load(result_dir / f"feature_stats_w{args.right}.npz"))
    scored = []
    for row in split:
        targets = [int(x) for x in row["targets"]]
        if len(targets) != 2:
            continue
        parent = set(left_stats["top_rows"][int(row["source"])].tolist()) - {-1}
        child_sets = [set(right_stats["top_rows"][target].tolist()) - {-1} for target in targets]
        covers = [len(parent & child) / max(1, len(parent)) for child in child_sets]
        overlap = len(parent & child_sets[0] & child_sets[1]) / max(1, len(parent))
        score = min(covers) * (1 - overlap) * float(row["coverage"])
        scored.append((score, row))
    selected = [row for _score, row in sorted(scored, key=lambda item: item[0], reverse=True)[:3]]
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    mapper = CacheRowMapper(Path(args.manifest), Path(args.qc_map), Path(args.qc_root), 30_000_000)
    model_left = load_checkpoint(Path(args.left_checkpoint), device)
    model_right = load_checkpoint(Path(args.right_checkpoint), device)
    draw_pair(result_dir, mapper, args.left, args.right, selected, model_left, model_right, Path(args.out), device)
    for row in selected:
        print(json.dumps(row, sort_keys=True))


if __name__ == "__main__":
    main()
