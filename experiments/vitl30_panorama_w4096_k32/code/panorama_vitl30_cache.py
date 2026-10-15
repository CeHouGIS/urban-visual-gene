"""Use existing panorama caches and recover only documented image mappings."""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
import pandas as pd

from sae_experiments.io_utils import metadata_for, digest, write_json
from sae_experiments.panorama_vitl30 import DIM, OUTPUT, PATCHES

QC_ROOT = Path("/host/root/mnt/nas/huangyj/GoogleSV/urban_visual_gene/formal_city_qc")
DATA_ROOT = QC_ROOT.parents[1]


def items(output):
    return json.loads((output / "cached_feature_map.json").read_text())["cities"]


def names(output):
    return [item["city"] for item in items(output)]


def cache_city(output, city):
    item = next(item for item in items(output) if item["city"] == city)
    path = Path(item["feature_path"])
    tokens = np.load(path, mmap_mode="r").reshape(-1, 4, PATCHES, DIM)
    if tokens.shape[0] != item["panoramas"]:
        raise ValueError(f"Cached panorama count changed: {city}")
    audit = json.loads((output / "cached_feature_audit.json").read_text())
    expected = next(row["identity"] for row in audit["cities"] if row["city"] == city)
    stat = path.stat()
    identity = hashlib.sha256(json.dumps([str(path), stat.st_size, stat.st_mtime_ns,
                                          (len(tokens) * 4 * PATCHES, DIM), .25]).encode()).hexdigest()
    if identity != expected:
        raise ValueError(f"Training feature identity changed: {city}")
    manifest = output / "cache_manifests" / f"{city.replace('/', '__')}.parquet"
    if not manifest.is_file():
        raise FileNotFoundError(f"Run panorama_vitl30_cache first: {manifest}")
    frame = pd.read_parquet(manifest)
    if len(frame) != len(tokens) * 4:
        raise ValueError(f"Image mapping count differs from cache: {city}")
    return frame, tokens, identity + "/" + digest(manifest)


def pano_key(path):
    pid, heading = Path(path).stem.rsplit("_", 1)
    if heading not in {"0", "90", "180", "270"}:
        raise ValueError(f"Not a directional Google image: {path}")
    return pid, int(heading)


def recover_catalog(selected):
    """Read bounded saved sample metadata, never walk street-image directories."""
    orders, paths = defaultdict(list), defaultdict(dict)
    seen = defaultdict(set)
    for part in sorted(QC_ROOT.glob("formal_sample_images.part*.tsv")):
        with part.open() as source:
            for line in source:
                key, image = line.rstrip("\n").split("\t", 1)
                if key not in selected:
                    continue
                pid, heading = pano_key(image)
                if pid not in seen[key]:
                    seen[key].add(pid)
                    orders[key].append(pid)
                paths[key].setdefault(pid, {})[heading] = image
    flags = defaultdict(lambda: defaultdict(lambda: [False, 0]))
    for part in sorted(QC_ROOT.glob("qc_part*/images.part*.csv")):
        with part.open(newline="") as source:
            for row in csv.DictReader(source):
                key = row["dataset"]
                if key not in selected:
                    continue
                pid, _ = pano_key(row["path"])
                flag = flags[key][pid]
                flag[0] |= row.get("black", "0") == "1"
                flag[1] += int(row.get("tunnel", "0") == "1")
    kept = {key: [pid for pid in ids if not flags[key][pid][0] and flags[key][pid][1] < 2]
            for key, ids in orders.items()}
    return kept, paths


def prepare(output):
    selected_items = items(output)
    qc_path = QC_ROOT / "feature_qc_map.json"
    qc = json.loads(qc_path.read_text())
    selected = {item["city"].split("/")[-1] for item in selected_items
                if Path(item["feature_path"]).name.startswith("dict_")}
    kept, paths = recover_catalog(selected)
    target = output / "cache_manifests"
    target.mkdir(parents=True, exist_ok=True)

    def one(item):
        city = item["city"]
        key = city.split("/")[-1]
        slug = city.replace("/", "__")
        n = item["panoramas"]
        source = Path(item["feature_path"])
        if source.name == "context.f16.npy":
            frame = pd.read_parquet(output / "manifests" / f"{slug}.parquet")
            if len(frame) != n * 4:
                raise ValueError(f"Existing ID manifest count differs: {city}")
            frame["image_mapping_verified"] = True
            method = "recorded extraction manifest"
        elif qc["cities"].get(key, {}).get("verified") and len(kept.get(key, [])) == n:
            records = []
            coords = metadata_for(set(kept[key]), DATA_ROOT / "metadata" / city / "panoids.csv")
            for i, pid in enumerate(kept[key]):
                views = paths[key][pid]
                if set(views) != {0, 90, 180, 270}:
                    raise ValueError(f"Incomplete mapped panorama: {city}/{pid}")
                lat, lon = coords.get(pid, (np.nan, np.nan))
                for heading in (0, 90, 180, 270):
                    records.append({"panoid": pid, "heading": heading, "image_path": views[heading],
                                    "lat": lat, "lon": lon, "context_pano_index": i,
                                    "image_mapping_verified": True})
            frame = pd.DataFrame(records)
            method = "saved sample order and reconstructed extraction quality gate; count verified"
        else:
            frame = pd.DataFrame({"panoid": [""] * (n * 4), "heading": np.tile([0, 90, 180, 270], n),
                                  "image_path": [""] * (n * 4), "lat": np.nan, "lon": np.nan,
                                  "context_pano_index": np.repeat(np.arange(n), 4),
                                  "image_mapping_verified": False})
            method = "unverified image mapping; feature summaries only"
        dest = target / f"{slug}.parquet"
        frame.to_parquet(dest, index=False)
        result = {"city": city, "panoramas": n, "method": method,
                  "mapped_images": int(frame.image_mapping_verified.sum()),
                  "geolocated_panoramas": int(frame.iloc[::4].lat.notna().sum()),
                  "manifest_sha256": digest(dest)}
        print("mapped", result, flush=True)
        return result

    with ThreadPoolExecutor(max_workers=8) as pool:
        rows = list(pool.map(one, selected_items))
    write_json(target / "report.json", {"qc_map_sha256": digest(qc_path), "cities": rows,
                                        "image_directory_scan": False})


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=OUTPUT)
    prepare(parser.parse_args().output)
