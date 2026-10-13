"""Build a replacement 30-city, four-heading NAS batch and extract DINOv3-B tokens.

This is a new image cohort, not the original Feature-MAE image set. Every selected
panorama has all four JPEGs and a matching entry in the NAS city metadata.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from PIL import Image

from .extract import load_model, transform
from .train import ROOT, digest, write_json

NAS_ROOT = Path("/host/root/mnt/nas/huangyj/GoogleSV")
MODEL_DIR = Path("/root/.cache/modelscope/models/facebook--dinov3-vitb16-pretrain-lvd1689m/snapshots/master")
OUTPUT = ROOT / "outputs/experiments/nas_30city_batchtopk_w4096_k32"
CITIES = (
    "Argentina/BuenosAires", "Australia/Melbourne", "Australia/Sydney",
    "Austria/Vienna", "Bangladesh/Dhaka", "Belgium/Antwerp",
    "SouthAfrica/CapeTown", "Brazil/SaoPaulo", "US/Phoenix",
    "China/Taipei", "Brazil/RiodeJaneiro", "Australia/Perth",
    "France/Paris", "Germany/Berlin", "Brazil/Brasília",
    "Indonesia/Yogyakarta", "Italy/Naples", "Korea/Seoul",
    "Mexico/Monterrey", "Netherlands/Amsterdam", "Nigeria/Lagos",
    "Portugal/Lisboa", "Russia/Moscow", "Singapore/Singapore",
    "SouthAfrica/Johannesburg", "Spain/Barcelona",
    "Sweden/Stockholm", "UAE/Dubai", "US/Chicago",
    "US/NewYork",
)
HEADINGS = (0, 90, 180, 270)
HEADING_BITS = {str(h): 1 << i for i, h in enumerate(HEADINGS)}


def stable_seed(seed: int, city: str) -> int:
    return int.from_bytes(hashlib.sha256(f"{seed}/{city}".encode()).digest()[:8], "little")


def candidate_panos(city_root: Path, seed: int, count: int) -> dict[str, Path]:
    """Spread candidates across randomized pano-ID shards."""
    shards = [leaf for top in city_root.iterdir() if top.is_dir()
              for leaf in top.iterdir() if leaf.is_dir()]
    if not shards:
        raise FileNotFoundError(f"No JPEG shards under {city_root}")
    rng = np.random.default_rng(seed)
    rng.shuffle(shards)
    selected: dict[str, Path] = {}
    for leaf in shards:
        masks: dict[str, int] = {}
        with os.scandir(leaf) as entries:
            for entry in entries:
                if not entry.is_file() or not entry.name.endswith(".jpg"):
                    continue
                stem, sep, suffix = entry.name[:-4].rpartition("_")
                if sep and suffix in HEADING_BITS:
                    masks[stem] = masks.get(stem, 0) | HEADING_BITS[suffix]
        complete = [panoid for panoid, mask in masks.items() if mask == 15]
        if not complete:
            continue
        rng.shuffle(complete)
        for panoid in complete[: min(32, count - len(selected))]:
            selected[panoid] = leaf
        if len(selected) >= count:
            break
    return selected


def metadata_for(panoids: set[str], metadata_csv: Path) -> dict[str, tuple[float, float]]:
    if not metadata_csv.is_file():
        raise FileNotFoundError(metadata_csv)
    found: dict[str, tuple[float, float]] = {}
    for chunk in pd.read_csv(metadata_csv, usecols=["panoid", "lat", "lon"],
                             dtype={"panoid": str}, chunksize=200_000):
        hits = chunk[chunk.panoid.isin(panoids) & chunk.lat.notna() & chunk.lon.notna()]
        for row in hits.itertuples(index=False):
            found[row.panoid] = (float(row.lat), float(row.lon))
        if len(found) == len(panoids):
            break
    return found


def build(nas_root: Path, output: Path, panoramas_per_city: int, seed: int) -> dict:
    manifest_root = output / "manifests"
    manifest_root.mkdir(parents=True, exist_ok=True)
    rows = []
    for city in CITIES:
        manifest = manifest_root / (city.replace("/", "__") + ".parquet")
        if manifest.is_file():
            old = pd.read_parquet(manifest, columns=["source_image_index", "panoid", "heading", "image_path"])
            n = old.panoid.nunique()
            if (len(old) != n * 4 or n < 512 or n > panoramas_per_city
                    or set(old.heading.unique()) != set(HEADINGS)
                    or not np.array_equal(old.source_image_index.to_numpy(), np.arange(len(old)))):
                raise ValueError(f"Existing city manifest is incompatible: {manifest}")
            rows.append({"city": city, "panoramas": n, "images": len(old),
                         "metadata_matched_candidates": None, "complete_candidates": None,
                         "manifest": str(manifest), "manifest_sha256": digest(manifest)})
            print("manifest existing", city, len(old), "images", flush=True)
            continue
        city_root = nas_root / "images" / city
        if not city_root.is_dir():
            raise FileNotFoundError(city_root)
        rng = np.random.default_rng(stable_seed(seed, city))
        if city == "US/NewYork":
            # The New York NAS directory has millions of files. Indexed metadata
            # IDs avoid a full directory scan while still verifying all four JPEGs.
            metadata_file = nas_root / "metadata" / city / "panoids.csv"
            candidates = {}
            for chunk in pd.read_csv(metadata_file, usecols=["panoid"], dtype={"panoid": str},
                                     chunksize=200_000):
                for panoid in chunk.panoid.dropna():
                    if len(candidates) >= panoramas_per_city * 2:
                        break
                    if len(panoid) < 2:
                        continue
                    leaf = city_root / panoid[0] / panoid[1]
                    if all((leaf / f"{panoid}_{heading}.jpg").is_file() for heading in HEADINGS):
                        candidates[panoid] = leaf
                if len(candidates) >= panoramas_per_city * 2:
                    break
        else:
            candidates = candidate_panos(city_root, stable_seed(seed, city), panoramas_per_city * 2)
        matched = metadata_for(set(candidates), nas_root / "metadata" / city / "panoids.csv")
        valid = []
        for panoid in matched:
            leaf = candidates[panoid]
            paths = [leaf / f"{panoid}_{heading}.jpg" for heading in HEADINGS]
            if all(p.is_file() and p.stat().st_size > 0 for p in paths):
                valid.append(panoid)
        if len(valid) < 512:
            raise ValueError(f"{city}: only {len(valid)} complete panoramas with metadata")
        rng.shuffle(valid)
        records = []
        for panoid in valid[:panoramas_per_city]:
            lat, lon = matched[panoid]
            for heading in HEADINGS:
                records.append({"source_image_index": len(records), "city": city,
                                "panoid": panoid, "heading": heading, "lat": lat, "lon": lon,
                                "image_path": str(candidates[panoid] / f"{panoid}_{heading}.jpg")})
        partial = manifest.with_suffix(".partial.parquet")
        pd.DataFrame.from_records(records).to_parquet(partial, index=False)
        os.replace(partial, manifest)
        rows.append({"city": city, "panoramas": len(records) // 4, "images": len(records),
                     "metadata_matched_candidates": len(matched), "complete_candidates": len(candidates),
                     "manifest": str(manifest), "manifest_sha256": digest(manifest)})
        print("manifest", city, len(records), "images", flush=True)
    report = {"batch_id": "nas_30city_four_heading_seed42_v1", "source": str(nas_root),
              "replaces_original_feature_mae_images": True,
              "images_are_original_feature_mae_images": False,
              "feature_contract": "DINOv3 ViT-B/16 224px, 14x14x768 L2-normalized patches",
              "cities": list(CITIES), "panoramas_per_city_target": panoramas_per_city,
              "images": sum(row["images"] for row in rows), "seed": seed, "detail": rows}
    write_json(manifest_root / "batch.json", report)
    return report


def extract_one(city: str, manifest_root: Path, token_root: Path, model, device: str,
                batch_size: int, workers: int) -> None:
    slug = city.replace("/", "__")
    manifest = manifest_root / (slug + ".parquet")
    frame = pd.read_parquet(manifest)
    manifest_sha256 = digest(manifest)
    target = token_root / "patch_tokens" / slug / "tokens.f16.npy"
    target.parent.mkdir(parents=True, exist_ok=True)
    shape = (len(frame), 196, 768)
    report_path = target.parent / "token_report.json"
    if target.exists():
        arr = np.load(target, mmap_mode="r")
        existing_report = json.loads(report_path.read_text()) if report_path.is_file() else {}
        if (arr.shape != shape or arr.dtype != np.float16
                or existing_report.get("manifest_sha256") != manifest_sha256):
            raise ValueError(f"Existing token cache has wrong shape/manifest: {target}")
        print("existing", city, len(frame), flush=True)
        return
    partial = target.with_name("tokens.partial.npy")
    progress = target.with_name("tokens.progress.json")
    if partial.exists() and progress.exists():
        arr = np.load(partial, mmap_mode="r+")
        resume = json.loads(progress.read_text())
        completed = int(resume["completed"])
        if (arr.shape != shape or completed < 0 or completed > len(frame)
                or resume.get("manifest_sha256") != manifest_sha256):
            raise ValueError(f"Invalid token resume state: {city}")
    elif not partial.exists() and not progress.exists():
        arr = np.lib.format.open_memmap(partial, mode="w+", dtype=np.float16, shape=shape)
        completed = 0
    else:
        raise ValueError(f"Partial/progress state mismatch: {city}")
    preprocess = transform()
    prefix = 1 + int(getattr(model.config, "num_register_tokens", 0))

    def read_image(path: str) -> torch.Tensor:
        with Image.open(path) as image:
            return preprocess(image.convert("RGB"))

    with ThreadPoolExecutor(max_workers=workers) as pool, torch.inference_mode():
        for start in range(completed, len(frame), batch_size):
            stop = min(start + batch_size, len(frame))
            paths = frame.image_path.iloc[start:stop].tolist()
            pixels = torch.stack(list(pool.map(read_image, paths))).to(device)
            if device.startswith("cuda"):
                pixels = pixels.to(torch.bfloat16)
            hidden = model(pixel_values=pixels).last_hidden_state[:, prefix:].float()
            if hidden.shape[1:] != (196, 768):
                raise ValueError(f"DINO output mismatch: {hidden.shape}")
            hidden = torch.nn.functional.normalize(hidden, dim=-1)
            arr[start:stop] = hidden.cpu().numpy().astype(np.float16)
            if stop % (batch_size * 25) == 0 or stop == len(frame):
                arr.flush()
                write_json(progress, {"city": city, "completed": stop, "total": len(frame),
                                      "manifest_sha256": manifest_sha256})
                print("extract", city, stop, "/", len(frame), flush=True)
    arr.flush()
    os.replace(partial, target)
    progress.unlink(missing_ok=True)
    write_json(report_path, {"city": city, "images": len(frame),
               "shape": list(shape), "model": str(MODEL_DIR), "manifest": str(manifest),
               "manifest_sha256": manifest_sha256})


def screen_city(city: str, manifest_root: Path, output: Path, workers: int) -> dict:
    from formal.quality.tiny_qc import QualityGate

    slug = city.replace("/", "__")
    manifest = manifest_root / (slug + ".parquet")
    frame = pd.read_parquet(manifest)
    if len(frame) % 4 or frame.heading.tolist()[:4] != list(HEADINGS):
        raise ValueError(f"Non-four-view manifest: {city}")
    target = output / "quality" / (slug + ".parquet")
    report_path = target.with_suffix(".json")
    target.parent.mkdir(parents=True, exist_ok=True)
    identity = digest(manifest)
    if target.exists() and report_path.exists():
        existing = json.loads(report_path.read_text())
        if existing.get("manifest_sha256") != identity:
            raise ValueError(f"Quality cache and manifest mismatch: {city}")
        return existing

    gate = QualityGate()
    paths = frame.image_path.tolist()
    panoids = frame.panoid.tolist()

    def judge(i: int) -> tuple[str, bool, bool, bool]:
        views = []
        for path in paths[i:i + 4]:
            with Image.open(path) as image:
                views.append(image.convert("RGB"))
        result = gate.score_views(views)
        return panoids[i], result["is_bad"], result["black"], result["tunnel"]

    with ThreadPoolExecutor(max_workers=workers) as pool:
        rows = list(pool.map(judge, range(0, len(frame), 4)))
    table = pd.DataFrame(rows, columns=["panoid", "is_bad", "black", "tunnel"])
    partial = target.with_suffix(".partial.parquet")
    table.to_parquet(partial, index=False)
    os.replace(partial, target)
    report = {"city": city, "panoramas": len(table), "rejected": int(table.is_bad.sum()),
              "black": int(table.black.sum()), "tunnel": int(table.tunnel.sum()),
              "model_loaded": gate.model_loaded, "manifest_sha256": identity,
              "quality_parquet": str(target)}
    write_json(report_path, report)
    print("quality", city, "rejected", report["rejected"], "/", len(table), flush=True)
    return report


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("command", choices=["build", "extract", "screen"])
    p.add_argument("--nas-root", type=Path, default=NAS_ROOT)
    p.add_argument("--output", type=Path, default=OUTPUT)
    p.add_argument("--panoramas-per-city", type=int, default=3200)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--model-dir", type=Path, default=MODEL_DIR)
    p.add_argument("--device", default="cuda:1")
    p.add_argument("--batch-size", type=int, default=8)
    p.add_argument("--workers", type=int, default=8)
    p.add_argument("--city-start", type=int, default=0)
    p.add_argument("--city-stop", type=int, default=len(CITIES))
    p.add_argument("--partial-build", action="store_true",
                   help="wait for each selected city manifest while the batch is built")
    args = p.parse_args()
    if min(args.panoramas_per_city, args.batch_size, args.workers) < 1:
        p.error("sample, batch and worker counts must be positive")
    if args.command == "build":
        build(args.nas_root, args.output, args.panoramas_per_city, args.seed)
        return
    manifest_root = args.output / "manifests"
    if not 0 <= args.city_start < args.city_stop <= len(CITIES):
        p.error("city slice must fall within the thirty selected cities")
    batch_path = manifest_root / "batch.json"
    if batch_path.is_file():
        report = json.loads(batch_path.read_text())
        if report["cities"] != list(CITIES):
            raise ValueError("Manifest city order differs from extractor config")
    elif not args.partial_build:
        raise FileNotFoundError(f"Batch manifest still building: {batch_path}")
    if not args.partial_build:
        for city in CITIES[args.city_start:args.city_stop]:
            if not (manifest_root / (city.replace("/", "__") + ".parquet")).is_file():
                raise FileNotFoundError(f"City manifest missing: {city}")
    model = load_model(args.model_dir, args.device) if args.command == "extract" else None
    selected = CITIES[args.city_start:args.city_stop]
    for city in selected:
        manifest = manifest_root / (city.replace("/", "__") + ".parquet")
        if args.partial_build:
            import time
            for _ in range(7200):
                if manifest.is_file():
                    break
                time.sleep(1)
            else:
                raise TimeoutError(f"City manifest was not built within two hours: {city}")
        if args.command == "extract":
            extract_one(city, manifest_root, args.output / "tokens", model, args.device,
                        args.batch_size, args.workers)
        else:
            screen_city(city, manifest_root, args.output, args.workers)


if __name__ == "__main__":
    main()
