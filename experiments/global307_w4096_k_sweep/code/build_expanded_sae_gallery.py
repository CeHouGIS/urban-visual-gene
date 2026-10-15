"""Map a trained BatchTopK dictionary back to QC-verified four-view images.

Candidate panoramas are ranked on SAE-training-excluded patch anchors. Each
displayed heatmap is recomputed with one complete panorama (3136 patches) as
the BatchTopK batch, then aligned with its four original 28x28 image grids.
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
from torch.nn import functional as F
from PIL import Image

from formal.analysis.plot_feature_activation_maps import load_checkpoint
from formal.analysis.plot_heldout_scaling import recover_selected_images


ROOT = Path("/host/root/mnt/nas/huangyj/GoogleSV/urban_visual_gene")
RESULT_DIR = ROOT / "formal_batchtopk_scaling_heldout_295city_40pano"
QC_ROOT = ROOT / "formal_city_qc"
MANIFEST = ROOT / "formal_batchtopk_307city_ablation_k2048/feature_qc_training_manifest.json"
CHECKPOINT = ROOT / "formal_batchtopk_307city_w4096_k8/sae_448_k4096.pt"
OUTDIR = REPO_ROOT / "reports/sae_4096"
HEADINGS = (0, 90, 180, 270)
BAIDU = {"Shanghai", "Hangzhou", "Nanjing", "Beijing", "Tianjin", "Guangzhou", "Shenzhen"}


def select_feature_ids(support: np.ndarray, per_band: int, seed: int) -> list[int]:
    """Deterministic spread over active-feature frequency, without cherry-picking labels."""
    eligible = np.flatnonzero(support >= 25)
    ordered = eligible[np.argsort(support[eligible])]
    rng = np.random.default_rng(seed)
    chosen = []
    for band in np.array_split(ordered, 3):
        if len(band):
            chosen.extend(int(i) for i in rng.choice(band, min(per_band, len(band)), replace=False))
    return sorted(chosen)


def scan_candidates(model, features: np.ndarray, device: torch.device,
                    batch_size: int, keep: int, max_panos: int = 0):
    if batch_size % 128:
        raise ValueError("batch_size must be a multiple of 128 patches per pano")
    total = min(len(features), max_panos * 128) if max_panos else len(features)
    if total % 128:
        raise ValueError("held-out features must contain complete 128-patch pano blocks")
    width = model.latent_dim
    support = np.zeros(width, dtype=np.int64)
    loss_sum = 0.0
    best_values = np.zeros((keep, width), dtype=np.float32)
    best_rows = np.full((keep, width), -1, dtype=np.int32)
    columns = np.arange(width)[None, :]
    with torch.inference_mode():
        for start in range(0, total, batch_size):
            stop = min(start + batch_size, total)
            x = torch.from_numpy(np.asarray(features[start:stop], dtype=np.float32)).to(device)
            acts = model.encode(x)
            loss_sum += (1 - F.cosine_similarity(x, model.decode(acts), dim=1)).sum().item()
            support += torch.count_nonzero(acts, dim=0).cpu().numpy()
            values, offsets = acts.reshape(-1, 128, width).max(dim=1)
            values = values.cpu().numpy()
            rows = start + np.arange(len(values), dtype=np.int32)[:, None] * 128 + offsets.cpu().numpy()
            combined_values = np.concatenate((best_values, values), axis=0)
            combined_rows = np.concatenate((best_rows, rows), axis=0)
            chosen = np.argpartition(combined_values, -keep, axis=0)[-keep:]
            best_values = combined_values[chosen, columns]
            best_rows = combined_rows[chosen, columns]
            if stop == total or (start // batch_size) % 50 == 0:
                print(f"Scanned {stop:,}/{total:,} held-out anchors", flush=True)
    return support, best_values, best_rows, loss_sum / total


def render_panorama(paths: dict, target: Path) -> None:
    panorama = Image.new("RGB", (1792, 448))
    for position, heading in enumerate(HEADINGS):
        with Image.open(paths[heading]) as image:
            panorama.paste(image.convert("RGB").resize((448, 448), Image.Resampling.LANCZOS), (position * 448, 0))
    panorama.save(target, quality=83, optimize=True)


def save_sparse_map(values: np.ndarray, target: Path) -> None:
    peak = float(values.max())
    fraction = np.clip(values / peak, 0, 1) if peak > 0 else np.zeros_like(values)
    rgba = (matplotlib.colormaps["magma"](fraction) * 255).astype(np.uint8)
    rgba[:, :, 3] = np.round(220 * fraction ** .65).astype(np.uint8)
    Image.fromarray(rgba, "RGBA").save(target, optimize=True)


def build_gallery(model, source_root: Path, anchors: list[dict], support: np.ndarray,
                  best_values: np.ndarray, best_rows: np.ndarray, resolved: dict,
                  selected: list[int], per_feature: int, device: torch.device, outdir: Path):
    (outdir / "assets/panoramas").mkdir(parents=True, exist_ok=True)
    (outdir / "assets/maps").mkdir(parents=True, exist_ok=True)
    selected_examples = defaultdict(list)
    for feature in selected:
        order = np.argsort(best_values[:, feature])[::-1]
        used = set()
        candidates = []
        for rank in order:
            index = int(best_rows[rank, feature])
            if index < 0 or best_values[rank, feature] <= 0 or index not in resolved:
                continue
            info = resolved[index]
            if info["city"] in BAIDU:
                continue
            if not all(h in info["images"] and Path(info["images"][h]).is_file() for h in HEADINGS):
                continue
            key = (info["city"], info["source_pano"])
            if key not in used:
                used.add(key)
                candidates.append((index, info))
        distinct = []
        cities = set()
        for index, info in candidates:
            if info["city"] not in cities:
                distinct.append((index, info))
                cities.add(info["city"])
        selected_examples[feature] = distinct + [(i, info) for i, info in candidates if info["city"] in cities and (i, info) not in distinct]

    pano_features = defaultdict(set)
    pano_info = {}
    for feature, candidates in selected_examples.items():
        for _, info in candidates:
            key = (info["city"], info["source_pano"])
            pano_features[key].add(feature)
            pano_info[key] = info
    map_paths = {}
    source_arrays = {}
    for number, (key, feature_ids) in enumerate(pano_features.items(), 1):
        city, source_pano = key
        info = pano_info[key]
        digest = hashlib.sha1(f"w4096:{city}:{source_pano}".encode()).hexdigest()[:14]
        image_rel = f"assets/panoramas/{digest}.jpg"
        render_panorama(info["images"], outdir / image_rel)
        if city not in source_arrays:
            source_arrays[city] = np.load(source_root / f"dict_{city}_p784.f16.npy", mmap_mode="r")
        source = source_arrays[city]
        offset = source_pano * 3136
        x = torch.from_numpy(np.asarray(source[offset:offset + 3136], dtype=np.float32).copy()).to(device)
        if x.shape != (3136, model.input_dim):
            raise ValueError(f"incomplete feature block for {city} pano {source_pano}")
        with torch.inference_mode():
            acts = model.encode(x).cpu().numpy()
        for feature in feature_ids:
            grid = np.concatenate(acts[:, feature].reshape(4, 28, 28), axis=1)
            if grid.max() <= 0:
                continue
            map_rel = f"assets/maps/{digest}_f{feature}.png"
            save_sparse_map(grid, outdir / map_rel)
            map_paths[key, feature] = (image_rel, map_rel, int(np.count_nonzero(grid)), float(grid.max()))
        if number % 20 == 0 or number == len(pano_features):
            print(f"Mapped {number}/{len(pano_features)} complete panoramas", flush=True)

    gallery = []
    for feature in selected:
        examples = []
        used = set()
        for anchor, info in selected_examples[feature]:
            key = (info["city"], info["source_pano"])
            if key in used or (key, feature) not in map_paths:
                continue
            image, mask, active, peak = map_paths[key, feature]
            examples.append({"city": info["city"], "source_pano": info["source_pano"],
                             "pano_id": info["pano_id"], "anchor": anchor,
                             "image": image, "mask": mask, "active_patches": active,
                             "peak_activation": round(peak, 4)})
            used.add(key)
            if len(examples) == per_feature:
                break
        gallery.append({"feature": feature, "anchor_support": int(support[feature]),
                        "examples": examples})
    return gallery


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, default=CHECKPOINT)
    parser.add_argument("--result-dir", type=Path, default=RESULT_DIR)
    parser.add_argument("--qc-root", type=Path, default=QC_ROOT)
    parser.add_argument("--manifest", type=Path, default=MANIFEST)
    parser.add_argument("--outdir", type=Path, default=OUTDIR)
    parser.add_argument("--device", default="cuda:2")
    parser.add_argument("--feature", type=int, action="append", help="Select specific feature IDs; repeat as needed")
    parser.add_argument("--per-band", type=int, default=8, help="Example features in each support tercile")
    parser.add_argument("--per-feature", type=int, default=4)
    parser.add_argument("--candidate-panos", type=int, default=24)
    parser.add_argument("--scan-panos", type=int, default=0, help="Limit scanned panos for a smoke test")
    parser.add_argument("--batch-size", type=int, default=8192)
    args = parser.parse_args()
    device = torch.device(args.device)
    model = load_checkpoint(args.checkpoint, device)
    if model.latent_dim <= model.input_dim:
        parser.error("checkpoint dictionary must be wider than the DINOv3 input")
    sample = json.loads((args.result_dir / "summary.json").read_text())["sample"]
    anchors = json.loads((args.result_dir / "anchor_manifest.json").read_text())
    features = np.load(json.loads((args.result_dir / "summary.json").read_text())["local_feature_cache"], mmap_mode="r")
    if len(anchors) != len(features):
        raise ValueError("anchor manifest and feature cache have different lengths")
    support, best_values, best_rows, heldout_loss = scan_candidates(
        model, features, device, args.batch_size, args.candidate_panos, args.scan_panos)
    selected = sorted(set(args.feature)) if args.feature else select_feature_ids(support, args.per_band, sample["seed"])
    if not selected or any(feature < 0 or feature >= model.latent_dim for feature in selected):
        parser.error("feature selection is empty or outside the dictionary")
    wanted = {int(row) for feature in selected for row in best_rows[:, feature] if row >= 0}
    qc = json.loads((args.qc_root / "feature_qc_map.json").read_text())
    print(f"Recovering images for {len(wanted)} candidate anchors", flush=True)
    resolved = recover_selected_images(args.qc_root, anchors, wanted, qc)
    source_root = Path(json.loads(args.manifest.read_text())["source_out"])
    gallery = build_gallery(model, source_root, anchors, support, best_values, best_rows,
                            resolved, selected, args.per_feature, device, args.outdir)
    payload = {"checkpoint": str(args.checkpoint), "dictionary_width": model.latent_dim,
               "input_dim": model.input_dim, "batch_topk_k": model.k,
               "scanned_panos": min(args.scan_panos or sample["panoramas"], sample["panoramas"]),
               "anchor_patches": min((args.scan_panos or sample["panoramas"]) * 128, len(features)),
               "heldout_cosine_loss": heldout_loss,
               "active_features": int(np.count_nonzero(support)),
               "map_batch_patches": 3136, "headings": list(HEADINGS),
               "selection": "8 deterministic features per support tercile; strongest distinct held-out panos",
               "features": gallery}
    args.outdir.mkdir(parents=True, exist_ok=True)
    (args.outdir / "gallery.json").write_text(json.dumps(payload, ensure_ascii=False, separators=(",", ":")))
    print(f"Saved {len(gallery)} features and {sum(len(row['examples']) for row in gallery)} activation maps", flush=True)


if __name__ == "__main__":
    main()
