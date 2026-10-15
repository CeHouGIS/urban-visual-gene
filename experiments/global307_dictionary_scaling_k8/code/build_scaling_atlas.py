"""Build an image atlas for every W512 feature and unassigned W1024 feature.

Lineage comes from sparse BatchTopK activations on the held-out anchors. The
overlays show dense positive preactivation on all four 28x28 patch grids,
so they do not depend on a viewer's browser or batch.
"""
from __future__ import annotations

from sae_experiments.paths import REPO_ROOT

import argparse
import hashlib
import json
from collections import Counter, defaultdict
from pathlib import Path

import matplotlib
import numpy as np
import torch
from PIL import Image

from formal.analysis.plot_feature_activation_maps import load_checkpoint
from formal.analysis.plot_heldout_scaling import recover_selected_images


RESULT_DIR = Path("/host/root/mnt/nas/huangyj/GoogleSV/urban_visual_gene/formal_batchtopk_scaling_heldout_295city_40pano")
QC_ROOT = Path("/host/root/mnt/nas/huangyj/GoogleSV/urban_visual_gene/formal_city_qc")
OUT_DIR = REPO_ROOT / "reports/scaling"
BAIDU = {"Shanghai", "Hangzhou", "Nanjing", "Beijing", "Tianjin", "Guangzhou", "Shenzhen"}


def top_rows(values: np.ndarray, count: int = 16) -> list[int]:
    positive = np.flatnonzero(values > 0)
    if not len(positive):
        return []
    ranked = positive[np.argsort(values[positive])[-count:][::-1]]
    return [int(row) for row in ranked]


def google_rows(values: np.ndarray, anchors: list[dict], count: int = 24) -> list[int]:
    """Rank panorama maxima, rather than filling the quota with nearby patches."""
    if len(values) % 128 == 0 and len(values) >= 256:
        first = (anchors[0]["city"], anchors[0].get("source_pano"))
        last = (anchors[127]["city"], anchors[127].get("source_pano"))
        next_one = (anchors[128]["city"], anchors[128].get("source_pano"))
        layout_valid = first == last and first != next_one and all(
            (anchors[start]["city"], anchors[start].get("source_pano")) ==
            (anchors[start + 127]["city"], anchors[start + 127].get("source_pano"))
            for start in range(0, len(anchors), 128)
        ) and len({(anchors[start]["city"], anchors[start].get("source_pano")) for start in range(0, len(anchors), 128)}) == len(anchors) // 128
        if layout_valid:
            blocks = np.asarray(values).reshape(-1, 128)
            best_patch = blocks.argmax(axis=1)
            best_value = blocks[np.arange(len(blocks)), best_patch]
            ranked = np.argsort(best_value)[::-1]
            rows = []
            for block in ranked:
                if best_value[block] <= 0:
                    break
                index = int(block * 128 + best_patch[block])
                if anchors[index]["city"] not in BAIDU:
                    rows.append(index)
                    if len(rows) == count:
                        break
            return rows
    selected = set()
    rows = []
    positive = np.flatnonzero(values > 0)
    for raw_index in positive[np.argsort(values[positive])[::-1]]:
        index = int(raw_index)
        anchor = anchors[index]
        key = (anchor["city"], anchor.get("source_pano", index))
        if anchor["city"] in BAIDU or key in selected:
            continue
        selected.add(key)
        rows.append(index)
        if len(rows) == count:
            break
    return rows


def inactive_probe_rows(model, features: np.ndarray, inactive: list[int], anchors: list[dict], device: torch.device, count: int = 24):
    """Find the strongest dense probe for features never selected by TopK."""
    if not inactive:
        return {}
    dense = np.zeros((len(features), len(inactive)), dtype=np.float32)
    selected = torch.as_tensor(inactive, device=device)
    with torch.inference_mode():
        for start in range(0, len(features), 4096):
            x = torch.from_numpy(np.asarray(features[start:start + 4096], dtype=np.float32)).to(device)
            dense[start:start + len(x)] = torch.relu(model._preactivate(x).index_select(1, selected)).cpu().numpy()
    probes = {}
    for column, feature in enumerate(inactive):
        probes[feature] = [(i, float(dense[i, column])) for i in google_rows(dense[:, column], anchors, count)]
    return probes


def edges_for_parent(parent: int, stable: dict, splits: dict, metrics: dict) -> list[dict]:
    children = {}
    if parent in stable:
        children[stable[parent]["child"]] = "stable"
    if parent in splits:
        for child in splits[parent]["children"]:
            children[child] = "stable_split" if children.get(child) == "stable" else "split"
    edges = []
    for child, kind in children.items():
        edges.append({
            "child": child,
            "kind": kind,
            "decoder_cosine": round(float(metrics["decoder_cosine"][parent, child]), 3),
            "activation_cosine": round(float(metrics["activation_cosine"][parent, child]), 3),
            "support_jaccard": round(float(metrics["support_jaccard"][parent, child]), 3),
            "parent_recall": round(float(metrics["parent_recall"][parent, child]), 3),
            "child_precision": round(float(metrics["child_precision"][parent, child]), 3),
        })
    return sorted(edges, key=lambda edge: (edge["kind"] not in ("stable", "stable_split"), edge["child"]))


def build_rows(anchors: list[dict], left: np.ndarray, right: np.ndarray, relations: dict, metrics: dict, probes: dict):
    stable = {row["parent"]: row for row in relations["stable"]}
    splits = {row["parent"]: row for row in relations["split_candidates"]}
    rows = []
    for parent in range(left.shape[1]):
        support = int(metrics["left_frequency"][parent])
        kind = "split" if parent in splits else "stable" if parent in stable else "unresolved" if support else "inactive"
        edges = edges_for_parent(parent, stable, splits, metrics)
        nearest = int((.5 * metrics["decoder_cosine"][parent] + .5 * metrics["activation_cosine"][parent]).argmax())
        references = []
        if not edges:
            references = [{"child": nearest, "kind": "reference",
                           "decoder_cosine": round(float(metrics["decoder_cosine"][parent, nearest]), 3),
                           "activation_cosine": round(float(metrics["activation_cosine"][parent, nearest]), 3)}]
        candidates = []
        if parent in splits:
            children = splits[parent]["children"]
            for position, child in enumerate(children):
                other = children[1 - position]
                mask = (left[:, parent] > 0) & (right[:, child] > 0) & (right[:, other] == 0)
                chosen = google_rows(np.where(mask, right[:, child], 0), anchors)
                candidates.append({"focus_child": child, "rows": chosen})
        elif parent in stable:
            child = stable[parent]["child"]
            candidates.append({"focus_child": child, "rows": google_rows(left[:, parent].astype(np.float32) * right[:, child].astype(np.float32), anchors)})
        elif support:
            candidates.append({"focus_child": nearest, "rows": google_rows(left[:, parent], anchors)})
        else:
            candidates.append({"focus_child": nearest, "rows": [i for i, _ in probes[parent]],
                               "probe_value": probes[parent][0][1] if probes[parent] else None})
        rows.append({"id": parent, "kind": kind, "support": support,
                     "stable_child": stable[parent]["child"] if parent in stable else None,
                     "split_coverage": round(float(splits[parent]["parent_coverage"]), 3) if parent in splits else None,
                     "edges": edges or references, "candidates": candidates})
    orphan_rows = []
    for child in relations["no_close_match"]:
        child = int(child)
        nearest = int((.5 * metrics["decoder_cosine"][:, child] + .5 * metrics["activation_cosine"][:, child]).argmax())
        orphan_rows.append({"id": child, "kind": "orphan", "support": int(metrics["right_frequency"][child]),
                            "nearest_parent": nearest,
                            "max_decoder_cosine": round(float(metrics["decoder_cosine"][:, child].max()), 3),
                            "max_activation_cosine": round(float(metrics["activation_cosine"][:, child].max()), 3),
                            "candidates": [{"focus_child": child, "rows": google_rows(right[:, child], anchors)}]})
    return rows, orphan_rows


def select_images(rows: list[dict], orphans: list[dict], anchors: list[dict], resolved: dict):
    for entry in rows + orphans:
        selected = set()
        cities = set()
        examples = []
        candidates = entry.pop("candidates")
        quota = 3 if entry["kind"] == "split" else 6
        pools = []
        for candidate in candidates:
            viable = []
            for index in candidate["rows"]:
                info = resolved.get(index)
                if info and all(h in info.get("images", {}) and Path(info["images"][h]).is_file() for h in (0, 90, 180, 270)):
                    viable.append((index, info))
            pools.append(viable)
        # Round robin ensures one child cannot consume every shared pano first.
        for _ in range(quota):
            for candidate, pool in zip(candidates, pools):
                distinct = [(i, info) for i, info in pool if (info["city"], info["source_pano"]) not in selected]
                if not distinct:
                    continue
                index, info = next(((i, info) for i, info in distinct if info["city"] not in cities), distinct[0])
                selected.add((info["city"], info["source_pano"]))
                cities.add(info["city"])
                examples.append({"anchor": index, "city": info["city"], "heading": info["heading"],
                                 "source_pano": info["source_pano"], "pano_id": info.get("pano_id"),
                                 "image_sources": [info["images"][h] for h in (0, 90, 180, 270)],
                                 "focus_child": candidate["focus_child"], "probe_value": candidate.get("probe_value")})
        entry["examples"] = examples
        entry["example_target"] = quota * len(candidates)
        entry["available_candidate_panos"] = len({(info["city"], info["source_pano"]) for pool in pools for _, info in pool})


def write_assets(rows: list[dict], orphans: list[dict], source_root: Path, models: dict, device: torch.device, outdir: Path):
    scenes = defaultdict(lambda: {"features": set(), "examples": []})
    for entry in rows + orphans:
        for example in entry["examples"]:
            key = (example["city"], example["source_pano"])
            scene = scenes[key]
            scene["examples"].append(example)
            if entry["kind"] == "orphan":
                scene["features"].update(((512, entry["nearest_parent"]), (1024, entry["id"])))
            else:
                scene["features"].add((512, entry["id"]))
                scene["features"].update((1024, edge["child"]) for edge in entry["edges"])
    (outdir / "assets/panoramas").mkdir(parents=True, exist_ok=True)
    (outdir / "assets/panorama_maps").mkdir(parents=True, exist_ok=True)
    cmap = matplotlib.colormaps["magma"]
    sources = {}
    for number, ((city, source_pano), scene) in enumerate(scenes.items(), 1):
        digest = hashlib.sha1(f"panorama-v1:{city}:{source_pano}".encode()).hexdigest()[:14]
        original = outdir / f"assets/panoramas/{digest}.jpg"
        if not original.exists():
            panorama = Image.new("RGB", (1792, 448))
            for position, path in enumerate(scene["examples"][0]["image_sources"]):
                with Image.open(path) as image:
                    panorama.paste(image.convert("RGB").resize((448, 448), Image.Resampling.LANCZOS), (position * 448, 0))
            panorama.save(original, quality=82, optimize=True)
        if city not in sources:
            sources[city] = np.load(source_root / f"dict_{city}_p784.f16.npy", mmap_mode="r")
        source = sources[city]
        start = source_pano * 3136
        x = torch.from_numpy(np.asarray(source[start:start + 3136], dtype=np.float32).copy()).to(device)
        maps = {}
        with torch.inference_mode():
            for width in (512, 1024):
                ids = sorted(feature for model_width, feature in scene["features"] if model_width == width)
                if not ids:
                    continue
                pre = torch.relu(models[width]._preactivate(x)).index_select(1, torch.as_tensor(ids, device=device)).cpu().numpy()
                maps.update({(width, feature): np.concatenate(pre[:, position].reshape(4, 28, 28), axis=1) for position, feature in enumerate(ids)})
        maximum = max((float(value.max()) for value in maps.values()), default=0.0)
        for (width, feature), value in maps.items():
            path = outdir / f"assets/panorama_maps/{digest}_w{width}_f{feature}.png"
            fraction = np.clip(value / maximum, 0, 1) if maximum > 0 else np.zeros_like(value)
            rgba = (cmap(fraction) * 255).astype(np.uint8)
            rgba[:, :, 3] = np.round(205 * fraction ** .8).astype(np.uint8)
            Image.fromarray(rgba, "RGBA").save(path, optimize=True)
        for example in scene["examples"]:
            example["image"] = f"assets/panoramas/{digest}.jpg"
            example["maps"] = {f"{width}:{feature}": f"assets/panorama_maps/{digest}_w{width}_f{feature}.png" for width, feature in maps}
            del example["image_sources"]
        if number % 50 == 0 or number == len(scenes):
            print(f"Rendered scenes: {number}/{len(scenes)}", flush=True)
    return len(scenes)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--result-dir", type=Path, default=RESULT_DIR)
    parser.add_argument("--qc-root", type=Path, default=QC_ROOT)
    parser.add_argument("--outdir", type=Path, default=OUT_DIR)
    parser.add_argument("--device", default="cuda:1")
    args = parser.parse_args()
    summary = json.loads((args.result_dir / "summary.json").read_text())
    anchors = json.loads((args.result_dir / "anchor_manifest.json").read_text())
    relations = json.loads((args.result_dir / "relations_w512_to_w1024.json").read_text())
    with np.load(args.result_dir / "pair_metrics_w512_to_w1024.npz") as values:
        metrics = {key: values[key] for key in values.files}
    left = np.load(args.result_dir / "activations_w512.f16.npy", mmap_mode="r")
    right = np.load(args.result_dir / "activations_w1024.f16.npy", mmap_mode="r")
    features = np.load(summary["local_feature_cache"], mmap_mode="r")
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    models = {width: load_checkpoint(Path(summary["checkpoints"][str(width)]), device) for width in (512, 1024)}
    inactive = [i for i, count in enumerate(metrics["left_frequency"]) if count == 0]
    probes = inactive_probe_rows(models[512], features, inactive, anchors, device)
    rows, orphans = build_rows(anchors, left, right, relations, metrics, probes)
    wanted = {index for row in rows + orphans for candidate in row["candidates"] for index in candidate["rows"]}
    qc = json.loads((args.qc_root / "feature_qc_map.json").read_text())
    print(f"Recovering {len(wanted)} candidate image anchors", flush=True)
    resolved = recover_selected_images(args.qc_root, anchors, wanted, qc)
    select_images(rows, orphans, anchors, resolved)
    missing = [(row["kind"], row["id"]) for row in rows + orphans
               if not row["examples"]]
    if missing:
        raise RuntimeError(f"missing image evidence for {len(missing)} atlas rows: {missing[:12]}")
    manifest = json.loads((args.qc_root.parent / "formal_batchtopk_307city_qc_filtered/feature_qc_training_manifest.json").read_text())
    scene_count = write_assets(rows, orphans, Path(manifest["source_out"]), models, device, args.outdir)
    counts = Counter(row["kind"] for row in rows)
    assert len(rows) == 512 and len({row["id"] for row in rows}) == 512
    assert len(orphans) == len(relations["no_close_match"])
    atlas = {"sample": summary["sample"], "pair": "512_to_1024", "counts": dict(counts),
             "headings": [0, 90, 180, 270], "example_target": 6,
             "overlay": "dense positive preactivation on four full 28x28 grids; shared color maximum within each panorama across directions and features",
             "relation": "sparse BatchTopK on held-out anchors", "scene_count": scene_count,
             "analysis": {key: summary[key] for key in ("pairs", "dictionary_activity", "heldout_cosine_loss", "stable_three_width_chains", "thresholds")},
             "parents": rows, "unassigned_children": orphans}
    path = args.outdir / "atlas_512_1024.json"
    path.write_text(json.dumps(atlas, ensure_ascii=False, separators=(",", ":")))
    print(f"Saved {len(rows)} parent rows, {len(orphans)} unassigned children and {scene_count} scenes to {path}", flush=True)


if __name__ == "__main__":
    main()
