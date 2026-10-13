"""Re-extract original 14x14x768 DINOv3-B tokens from Feature-MAE TAR shards.

The archived image table supplies the exact QC image IDs and their original
source_image_index. Each selected (panoid, heading) is joined to its original
TAR sidecar; missing or ambiguous matches stop extraction rather than silently
substituting a different street-view image.
"""
from __future__ import annotations

import argparse
import io
import json
import os
from collections import OrderedDict
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow.parquet as pq
import torch
from PIL import Image
from torchvision.transforms import v2
from transformers import AutoModel

from .train import CITIES, DEFAULT_OUTPUT, DEFAULT_TOKENS, EXPECTED_IMAGES, write_json

OLD_IMAGES = Path("/nas_data_24T/GSV/packages_main/cities")
OLD_MODEL = Path("/workplace/models/modelscope/models/facebook--dinov3-vitb16-pretrain-lvd1689m/snapshots/master")


def exact_image_paths(rows: pd.DataFrame, city_dir: Path) -> pd.DataFrame:
    if not city_dir.is_dir():
        raise FileNotFoundError(city_dir)
    want = rows[["panoid", "heading"]].copy()
    want["panoid"] = want["panoid"].astype(str)
    want["heading"] = want["heading"].astype(int)
    want["join_key"] = want.panoid + ":" + want.heading.astype(str)
    wanted = set(want.join_key)
    found = []
    sidecars = sorted(city_dir.glob("shard-*.parquet"))
    if not sidecars:
        raise FileNotFoundError("No TAR sidecars in " + str(city_dir))
    for sidecar in sidecars:
        tar = sidecar.with_suffix(".tar")
        if not tar.is_file():
            continue
        table = pq.read_table(sidecar, columns=["panoid", "heading", "jpg_offset", "jpg_size"])
        frame = table.to_pandas()
        frame = frame[frame.panoid.notna() & frame.heading.notna()].copy()
        frame["panoid"] = frame.panoid.astype(str)
        frame["heading"] = frame.heading.astype(int)
        frame["join_key"] = frame.panoid + ":" + frame.heading.astype(str)
        matches = frame[frame.join_key.isin(wanted)].copy()
        if len(matches):
            matches["tar_path"] = str(tar)
            found.append(matches)
    if not found:
        raise ValueError("No archived image IDs matched " + str(city_dir))
    lookup = pd.concat(found, ignore_index=True)
    # A shard may repeat a pano heading. Identical byte identities are harmless.
    lookup = lookup.drop_duplicates(["join_key", "tar_path", "jpg_offset", "jpg_size"])
    ambiguous = lookup.join_key.value_counts()
    if (ambiguous > 1).any():
        raise ValueError(f"{city_dir}: {int((ambiguous > 1).sum())} ambiguous image identities")
    joined = want.merge(lookup[["join_key", "tar_path", "jpg_offset", "jpg_size"]],
                        on="join_key", how="left", validate="one_to_one", sort=False)
    if joined.tar_path.isna().any():
        sample = joined.loc[joined.tar_path.isna(), "join_key"].head(5).tolist()
        raise ValueError(f"{city_dir}: {int(joined.tar_path.isna().sum())} missing QC images, e.g. {sample}")
    joined["source_image_index"] = rows.source_image_index.to_numpy(np.int64)
    return joined


def preflight(images_root: Path, manifest_root: Path, output: Path) -> list[tuple[str, pd.DataFrame]]:
    missing = []
    frames = []
    for city in CITIES:
        manifest = manifest_root / (city.replace("/", "__") + ".parquet")
        source = images_root / city
        if not manifest.is_file() or not source.is_dir():
            missing.append({"city": city, "manifest": str(manifest) if not manifest.is_file() else None,
                            "source_dir": str(source) if not source.is_dir() else None})
            continue
        frame = pd.read_parquet(manifest, columns=["source_image_index", "panoid", "heading"])
        frames.append((city, frame))
    report = {"status": "source_missing" if missing else "source_present",
              "expected_cities": 30, "found_cities": len(frames),
              "manifest_images": sum(len(frame) for _, frame in frames),
              "expected_images": EXPECTED_IMAGES, "missing": missing,
              "images_root": str(images_root), "manifest_root": str(manifest_root)}
    if not missing and report["manifest_images"] != EXPECTED_IMAGES:
        report["status"] = "wrong_image_count"
    write_json(output / "extraction_preflight.json", report)
    if report["status"] != "source_present":
        raise RuntimeError("Original Feature-MAE TAR source is unavailable or incomplete; see extraction_preflight.json")
    return frames


class TarReader:
    def __init__(self, max_open: int = 16):
        self.handles: OrderedDict[str, int] = OrderedDict()
        self.max_open = max_open

    def get(self, path: str) -> int:
        if path in self.handles:
            handle = self.handles.pop(path)
            self.handles[path] = handle
            return handle
        handle = os.open(path, os.O_RDONLY)
        self.handles[path] = handle
        if len(self.handles) > self.max_open:
            os.close(self.handles.popitem(last=False)[1])
        return handle

    def image(self, path: str, offset: int, size: int) -> Image.Image:
        payload = os.pread(self.get(path), size, offset)
        if len(payload) != size:
            raise IOError(f"short read: {path} {offset} {size}")
        with Image.open(io.BytesIO(payload)) as image:
            return image.convert("RGB")

    def close(self) -> None:
        for handle in self.handles.values():
            os.close(handle)
        self.handles.clear()


def load_model(model_dir: Path, device: str):
    if not (model_dir / "config.json").is_file():
        raise FileNotFoundError("DINOv3 ViT-B/16 weights unavailable: " + str(model_dir))
    model = AutoModel.from_pretrained(model_dir, local_files_only=True).eval().requires_grad_(False).to(device)
    hidden_size = int(model.config.hidden_size)
    if hidden_size != 768:
        raise ValueError(f"expected DINOv3-B/16 hidden_size=768, got {hidden_size}")
    if device.startswith("cuda"):
        model = model.to(torch.bfloat16)
    return model


def transform():
    return v2.Compose([
        v2.Resize((224, 224), interpolation=v2.InterpolationMode.BICUBIC, antialias=True),
        v2.ToImage(), v2.ToDtype(torch.float32, scale=True),
        v2.Normalize(mean=(.485, .456, .406), std=(.229, .224, .225)),
    ])


def extract_city(city: str, rows: pd.DataFrame, images_root: Path, token_root: Path,
                 model, device: str, batch_size: int) -> dict:
    matched = exact_image_paths(rows, images_root / city)
    target = token_root / "patch_tokens" / city.replace("/", "__") / "tokens.f16.npy"
    target.parent.mkdir(parents=True, exist_ok=True)
    count = max(12800, int(rows.source_image_index.max()) + 1)
    if count != 12800:
        raise ValueError(f"{city}: legacy source array should have 12,800 rows, got {count}")
    if target.exists():
        existing = np.load(target, mmap_mode="r")
        if existing.shape != (12800, 196, 768) or existing.dtype != np.float16:
            raise ValueError(str(target) + ": wrong existing array shape")
        return {"city": city, "status": "existing", "qc_images": len(rows), "tokens": str(target)}
    partial = target.with_suffix(".partial.npy")
    progress_path = target.with_suffix(".progress.json")
    if partial.exists() and progress_path.exists():
        tokens = np.load(partial, mmap_mode="r+")
        completed = int(json.loads(progress_path.read_text())["completed"])
        if tokens.shape != (12800, 196, 768):
            raise ValueError("resume array has wrong shape")
    elif not partial.exists() and not progress_path.exists():
        tokens = np.lib.format.open_memmap(partial, mode="w+", dtype=np.float16,
                                           shape=(12800, 196, 768))
        completed = 0
    else:
        raise ValueError("partial array and progress file disagree")
    preprocess = transform()
    reader = TarReader()
    prefix = 1 + int(getattr(model.config, "num_register_tokens", 0))
    try:
        for start in range(completed, len(matched), batch_size):
            subset = matched.iloc[start:start + batch_size]
            pixels = torch.stack([preprocess(reader.image(str(row.tar_path), int(row.jpg_offset), int(row.jpg_size)))
                                  for row in subset.itertuples(index=False)]).to(device)
            if device.startswith("cuda"):
                pixels = pixels.to(torch.bfloat16)
            with torch.inference_mode():
                hidden = model(pixel_values=pixels).last_hidden_state[:, prefix:].float()
                if hidden.shape[1:] != (196, 768):
                    raise ValueError("DINOv3 token shape mismatch: " + str(hidden.shape))
                hidden = torch.nn.functional.normalize(hidden, dim=-1)
            indices = subset.source_image_index.to_numpy(np.int64)
            tokens[indices] = hidden.cpu().numpy().astype(np.float16)
            tokens.flush()
            write_json(progress_path, {"city": city, "completed": start + len(subset),
                                       "total": len(matched)})
            if (start // batch_size + 1) % 100 == 0:
                print(city, start + len(subset), "/", len(matched), flush=True)
    finally:
        reader.close()
    tokens.flush()
    os.replace(partial, target)
    progress_path.unlink(missing_ok=True)
    result = {"city": city, "status": "complete", "qc_images": len(rows),
              "tokens": str(target), "shape": [12800, 196, 768]}
    write_json(target.with_suffix(".report.json"), result)
    return result


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--images-root", type=Path, default=OLD_IMAGES)
    p.add_argument("--model-dir", type=Path, default=OLD_MODEL)
    p.add_argument("--manifest-root", type=Path, default=DEFAULT_OUTPUT / "recovered_manifests")
    p.add_argument("--token-root", type=Path, default=DEFAULT_TOKENS)
    p.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    p.add_argument("--device", default="cuda:1")
    p.add_argument("--batch-size", type=int, default=8)
    p.add_argument("--preflight-only", action="store_true")
    args = p.parse_args()
    if args.batch_size < 1:
        p.error("batch-size must be positive")
    frames = preflight(args.images_root, args.manifest_root, args.output)
    if args.preflight_only:
        return
    model = load_model(args.model_dir, args.device)
    results = []
    for city, rows in frames:
        results.append(extract_city(city, rows, args.images_root, args.token_root,
                                    model, args.device, args.batch_size))
    write_json(args.output / "extraction_report.json", {"status": "complete",
                "cities": results, "images": sum(r["qc_images"] for r in results),
                "model_dir": str(args.model_dir.resolve()), "images_root": str(args.images_root.resolve())})


if __name__ == "__main__":
    main()
