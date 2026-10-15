"""SAE dimensions represented by VLM embeddings of top-activating image crops.

Input: an explicit JSONL image/feature mapping and a checkpoint, optionally an
existing sparse activation archive. Ranking is by maximum *sparse* activation
per distinct image, not by patch count. No spatial coordinates or decoder atoms
are supplied to UMAP. Run each stage separately or use ``--stage all``.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import heapq
import json
import math
import os
from collections import defaultdict
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor
from contextlib import nullcontext
import multiprocessing
import textwrap
from pathlib import Path

import numpy as np
import torch
from PIL import Image, ImageOps
from scipy.ndimage import label

from formal.analysis.plot_feature_activation_maps import load_checkpoint


def digest(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def write_json(path: Path, data):
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(data, ensure_ascii=False, indent=2, allow_nan=False))
    os.replace(temporary, path)


def write_jsonl(path: Path, rows):
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False, allow_nan=False) + "\n")
    os.replace(temporary, path)


def read_manifest(path: Path, grid: int):
    rows = [json.loads(line) for line in path.read_text().splitlines() if line.strip()]
    if not rows:
        raise ValueError("Empty image manifest")
    identities = set()
    for i, row in enumerate(rows):
        for key in ("city", "pano", "heading", "image"):
            if key not in row:
                raise ValueError(f"Image {i} missing {key}")
        row["heading"] = int(row["heading"])
        identity = (row["city"], row["pano"], row["heading"])
        if identity in identities:
            raise ValueError(f"Duplicate image identity {identity}")
        identities.add(identity)
        for key in ("image", "feature_path"):
            if row.get(key):
                p = Path(row[key])
                row[key] = str((path.parent / p).resolve() if not p.is_absolute() else p)
        if "row_start" in row:
            row["row_start"] = int(row["row_start"])
            if row["row_start"] < 0:
                raise ValueError("Negative feature offset")
        row["grid"] = grid
    return rows


class TopImages:
    """Bounded, deterministic top-k with one observation per image/dimension."""

    def __init__(self, width: int, top: int):
        self.heaps = [[] for _ in range(width)]
        self.support = np.zeros(width, np.int64)
        self.top = top

    def add(self, image_id: int, scores: np.ndarray):
        for j in np.flatnonzero(scores > 0):
            self.support[j] += 1
            item = (float(scores[j]), -int(image_id))
            heap = self.heaps[j]
            if len(heap) < self.top:
                heapq.heappush(heap, item)
            elif item > heap[0]:
                heapq.heapreplace(heap, item)

    def rows(self):
        for j, heap in enumerate(self.heaps):
            for rank, (score, neg_id) in enumerate(sorted(heap, reverse=True), 1):
                yield {"feature": j, "rank": rank, "image_id": -neg_id, "activation": score}


def dense_sparse_image(indices: np.ndarray, values: np.ndarray, width: int):
    if indices.shape != values.shape or indices.ndim != 2:
        raise ValueError("Sparse indices and values must have shape [patch, slot]")
    if not np.isfinite(values).all() or np.any(values < 0):
        raise ValueError("Sparse activations must be finite and nonnegative")
    valid = values > 0
    if np.any((indices[valid] < 0) | (indices[valid] >= width)):
        raise ValueError("Positive activation has an invalid feature index")
    dense = np.zeros((len(indices), width), np.float32)
    patch = np.broadcast_to(np.arange(len(indices))[:, None], indices.shape)
    np.maximum.at(dense, (patch[valid], indices[valid].astype(int)), values[valid])
    return dense


def encode_panorama_batches(model, blocks: np.ndarray, device: torch.device):
    """Equivalent to model.encode on each complete panorama independently.

    Processing several panoramas simultaneously is only a throughput choice:
    each panorama has its own joint BatchTopK budget. This avoids changing the
    analysis when GPU chunk size changes. It is explicitly not a global budget
    over the entire collection and can differ from earlier inference archives.
    """
    with torch.inference_mode():
        x = torch.from_numpy(np.asarray(blocks, np.float32)).to(device)
        positive = torch.relu(model._preactivate(x.flatten(0, 1))).reshape(
            x.shape[0], x.shape[1], model.latent_dim)
        flat = positive.flatten(1)
        n = min(model.k * x.shape[1], flat.shape[1])
        values, indices = torch.topk(flat, n, dim=1, sorted=False)
        sparse = torch.zeros_like(flat).scatter_(1, indices, values)
        return sparse.reshape_as(positive).float().cpu().numpy()


class ActivationSource:
    def __init__(self, rows, args):
        self.rows, self.args = rows, args
        self.grid = args.grid
        self.device = torch.device(args.device)
        self.model = load_checkpoint(Path(args.checkpoint), self.device)
        if not hasattr(self.model, "k"):
            raise ValueError("Checkpoint does not define a TopK budget")
        self.width = self.model.latent_dim
        if args.expected_width and self.width != args.expected_width:
            raise ValueError(f"Expected W={args.expected_width}, found W={self.width}")
        if args.expected_k and self.model.k != args.expected_k:
            raise ValueError(f"Expected k={args.expected_k}, found k={self.model.k}")
        self.indices = self.values = None
        self.shards = {}
        if args.sparse_activations:
            # Extract compressed npz arrays once to a disk-backed local cache.
            # Avoid repeatedly decompressing the complete archive for each image.
            cache = Path(args.outdir) / "sparse_cache"
            cache.mkdir(exist_ok=True)
            with np.load(args.sparse_activations, allow_pickle=True) as z:
                for name in ("idx", "val"):
                    p = cache / f"{name}.npy"
                    if not p.exists():
                        tmp = p.with_suffix(".tmp.npy")
                        np.save(tmp, z[name])
                        os.replace(tmp, p)
                for key in ("city", "pano", "heading"):
                    if key in z:
                        actual = list(map(str, z[key]))
                        expected = [str(r[key]) for r in rows]
                        if actual != expected:
                            raise ValueError(f"Sparse archive {key} order differs from image manifest")
            self.indices = np.load(cache / "idx.npy", mmap_mode="r")
            self.values = np.load(cache / "val.npy", mmap_mode="r")
            if self.indices.shape != self.values.shape or self.indices.ndim != 3:
                raise ValueError("Archive idx/val must have shape [image, patch, slot]")
            if self.indices.shape[:2] != (len(rows), self.grid ** 2):
                raise ValueError("Archive image/patch counts differ from manifest/grid")
        else:
            grouped = defaultdict(dict)
            for i, row in enumerate(rows):
                for key in ("feature_path", "row_start"):
                    if key not in row:
                        raise ValueError(f"Image {i}: require {key}, or supply --sparse-activations")
                grouped[(row["city"], row["pano"])][row["heading"]] = i
            self.groups = []
            self.group_for_image = {}
            for key, views in grouped.items():
                if set(views) != {0, 90, 180, 270}:
                    raise ValueError(f"Incomplete four-view panorama {key}")
                ids = [views[h] for h in (0, 90, 180, 270)]
                self.groups.append(ids)
                for i in ids:
                    self.group_for_image[i] = ids
            self.last_ids = None
            self.last_acts = None

    def feature_block(self, image_id):
        row = self.rows[image_id]
        path = row["feature_path"]
        if path not in self.shards:
            self.shards[path] = np.load(path, mmap_mode="r")
        start = row["row_start"]
        data = self.shards[path][start:start + self.grid ** 2]
        if data.shape != (self.grid ** 2, self.model.input_dim):
            raise ValueError(f"Incomplete or incompatible feature block at image {image_id}")
        if not np.isfinite(data).all():
            raise ValueError(f"Nonfinite features at image {image_id}")
        return data

    def scan(self):
        if self.indices is not None:
            for i in range(len(self.rows)):
                yield i, self.image(i)
        else:
            for start in range(0, len(self.groups), self.args.pano_batch):
                ids_batch = self.groups[start:start + self.args.pano_batch]
                blocks = np.stack([np.concatenate([self.feature_block(i) for i in ids]) for ids in ids_batch])
                acts = encode_panorama_batches(self.model, blocks, self.device)
                for ids, values in zip(ids_batch, acts):
                    for pos, i in enumerate(ids):
                        yield i, values[pos * self.grid ** 2:(pos + 1) * self.grid ** 2]

    def image(self, image_id):
        if self.indices is not None:
            return dense_sparse_image(self.indices[image_id], self.values[image_id], self.width)
        ids = self.group_for_image[image_id]
        if self.last_ids != ids:
            blocks = np.concatenate([self.feature_block(i) for i in ids])[None]
            self.last_acts = encode_panorama_batches(self.model, blocks, self.device)[0]
            self.last_ids = ids
        pos = ids.index(image_id)
        return self.last_acts[pos * self.grid ** 2:(pos + 1) * self.grid ** 2]


def crop_region(heatmap: np.ndarray, image_size, fraction=.5, margin=1, min_side=3):
    """Peak's 8-connected component, bounded context, native pixel coordinates."""
    heatmap = np.asarray(heatmap)
    if heatmap.ndim != 2 or not np.isfinite(heatmap).all() or heatmap.max() <= 0:
        raise ValueError("Crop requires a finite positive two-dimensional activation map")
    gh, gw = heatmap.shape
    peak_index = np.unravel_index(int(heatmap.argmax()), heatmap.shape)
    components, count = label(heatmap >= float(heatmap.max()) * fraction, np.ones((3, 3)))
    peak_component = components[peak_index]
    yy, xx = np.where(components == peak_component)
    x0, x1 = max(0, int(xx.min()) - margin), min(gw, int(xx.max()) + 1 + margin)
    y0, y1 = max(0, int(yy.min()) - margin), min(gh, int(yy.max()) + 1 + margin)
    def expand(lo, hi, bound):
        length = min(bound, max(hi - lo, min_side))
        lo = min(max(0, (lo + hi - length) // 2), bound - length)
        return lo, lo + length
    x0, x1 = expand(x0, x1, gw)
    y0, y1 = expand(y0, y1, gh)
    width, height = image_size
    bbox = [math.floor(x0 * width / gw), math.floor(y0 * height / gh),
            math.ceil(x1 * width / gw), math.ceil(y1 * height / gh)]
    return bbox, {"component_patches": int(len(xx)), "threshold_components": int(count),
                  "peak_patch": [int(peak_index[0]), int(peak_index[1])], "patch_bbox": [x0, y0, x1, y1]}


def source_image(row):
    with Image.open(row["image"]) as src:
        image = ImageOps.exif_transpose(src).convert("RGB")
    # Baidu panorama views may be explicit native horizontal slices.
    if row.get("view_bbox"):
        image = image.crop(tuple(row["view_bbox"]))
    return image


def rank_images(rows, args, source):
    out = Path(args.outdir)
    if (out / "top_images.jsonl").exists():
        print("Reuse completed top-image ranking", flush=True)
        return
    ranking = TopImages(source.width, args.top)
    for done, (i, activations) in enumerate(source.scan(), 1):
        ranking.add(i, activations.max(axis=0))
        if done % 100 == 0 or done == len(rows):
            print(f"Ranked {done:,}/{len(rows):,} images", flush=True)
    np.save(out / "positive_image_support.npy", ranking.support)
    write_jsonl(out / "top_images.jsonl", ranking.rows())


def save_image_crops(task):
    """CPU worker: save native crops from already-selected sparse heatmaps."""
    i, row, items, heatmaps, settings, output = task
    out = Path(output)
    try:
        image = source_image(row)
    except OSError as exc:
        return [], [{"image_id": i, "image": row["image"], "error": str(exc)}]
    crops = []
    for item, heatmap in zip(items, heatmaps):
        j, rank = item["feature"], item["rank"]
        bbox, details = crop_region(heatmap, image.size, *settings)
        directory = out / "crops" / f"f{j:04d}"
        directory.mkdir(parents=True, exist_ok=True)
        target = directory / f"rank{rank:03d}_image{i:07d}.jpg"
        if not target.exists():
            image.crop(tuple(bbox)).save(target, quality=92)
        heat_path = target.with_suffix(".heat.png")
        Image.fromarray(np.round(heatmap / heatmap.max() * 255).astype(np.uint8)).save(heat_path)
        crops.append({**item, "city": row["city"], "pano": row["pano"],
                      "heading": row["heading"], "image": row["image"],
                      "crop": str(target.relative_to(out)), "heatmap": str(heat_path.relative_to(out)),
                      "bbox_native_view": bbox, "native_view_size": list(image.size), **details})
    return crops, []


def make_crops(rows, args, source):
    out = Path(args.outdir)
    if (out / "crops.jsonl").exists():
        print("Reuse completed crop manifest", flush=True)
        return
    selected = [json.loads(line) for line in (out / "top_images.jsonl").read_text().splitlines()]
    by_image = defaultdict(list)
    for item in selected:
        by_image[item["image_id"]].append(item)
    crops, errors = [], []
    settings = (args.crop_fraction, args.margin, args.min_side)
    workers = getattr(args, "crop_workers", 1)
    executor = ProcessPoolExecutor(workers, mp_context=multiprocessing.get_context("spawn")) if workers > 1 else None
    pending = []
    done = 0
    def collect(future):
        nonlocal done
        saved, failed = future.result() if executor else future
        crops.extend(saved)
        errors.extend(failed)
        done += 1
        if done % 100 == 0 or done == len(by_image):
            print(f"Cropped {done:,}/{len(by_image):,} source images; {len(crops):,} regions", flush=True)
    try:
        for i, items in sorted(by_image.items()):
            acts = source.image(i)
            heatmaps = np.stack([acts[:, item["feature"]].reshape(args.grid, args.grid) for item in items])
            for item, heatmap in zip(items, heatmaps):
                if not np.isclose(heatmap.max(), item["activation"], rtol=2e-3, atol=1e-6):
                    raise ValueError(f"Ranking/crop activation mismatch for image {i}, feature {item['feature']}")
            task = (i, rows[i], items, heatmaps, settings, str(out))
            if executor:
                pending.append(executor.submit(save_image_crops, task))
                if len(pending) >= workers * 2:
                    collect(pending.pop(0))
            else:
                collect(save_image_crops(task))
        for future in pending:
            collect(future)
    finally:
        if executor:
            executor.shutdown(wait=True, cancel_futures=True)
    write_json(out / "crop_errors.json", errors)
    write_jsonl(out / "crops.jsonl", sorted(crops, key=lambda r: (r["feature"], r["rank"])))


def embed_crops(args):
    from transformers import AutoImageProcessor, AutoModel

    out = Path(args.outdir)
    rows = [json.loads(line) for line in (out / "crops.jsonl").read_text().splitlines()]
    if not rows:
        raise ValueError("No crops available to embed")
    if (out / "embedding_run.json").exists():
        print("Reuse completed image embeddings", flush=True)
        return
    processor = AutoImageProcessor.from_pretrained(args.vlm_model)
    model = AutoModel.from_pretrained(args.vlm_model).to(args.device).eval()
    config_hash = hashlib.sha256(model.config.to_json_string().encode()).hexdigest()
    state = {"model": args.vlm_model, "config_sha256": config_hash,
             "processor": processor.to_dict(), "crop_manifest_sha256": digest(out / "crops.jsonl"),
             "precision": args.embedding_precision}
    state_path = out / "embedding_state.json"
    if state_path.exists() and json.loads(state_path.read_text()) != state:
        raise ValueError("VLM configuration or crop manifest changed; use a new output directory")
    write_json(state_path, state)
    completed_path = out / "embedding_completed.npy"
    if completed_path.exists():
        completed = np.load(completed_path, mmap_mode="r+")
        embeddings = np.load(out / "crop_embeddings.f32.npy", mmap_mode="r+")
        if len(completed) != len(rows):
            raise ValueError("Embedding checkpoint does not match crop manifest")
    else:
        completed = embeddings = None
    def prepare_batch(start, end):
        images = []
        for r in rows[start:end]:
            with Image.open(out / r["crop"]) as im:
                images.append(im.convert("RGB"))
        return processor(images=images, return_tensors="pt")

    if args.embedding_precision != "fp32" and not str(args.device).startswith("cuda"):
        raise ValueError("Mixed precision requires a CUDA device")
    dtype = {"fp16": torch.float16, "bf16": torch.bfloat16}.get(args.embedding_precision)
    workers = args.embedding_workers
    def incomplete_batches():
        for start in range(0, len(rows), args.embedding_batch):
            end = min(start + args.embedding_batch, len(rows))
            if completed is None or not completed[start:end].all():
                yield start, end
    with ThreadPoolExecutor(max_workers=workers) as executor:
        jobs = iter(incomplete_batches())
        pending = []
        def enqueue():
            try:
                start, end = next(jobs)
            except StopIteration:
                return False
            pending.append((start, end, executor.submit(prepare_batch, start, end)))
            return True
        for _ in range(workers):
            enqueue()
        while pending:
            start, end, future = pending.pop(0)
            inputs = future.result().to(args.device)
            with torch.inference_mode(), (torch.autocast("cuda", dtype=dtype) if dtype else nullcontext()):
                values = model.get_image_features(**inputs)
                if not isinstance(values, torch.Tensor):
                    values = values.pooler_output
                values = torch.nn.functional.normalize(values.float(), dim=-1).cpu().numpy()
            if not np.isfinite(values).all():
                raise ValueError("VLM produced nonfinite embeddings")
            if embeddings is None:
                embeddings = np.lib.format.open_memmap(out / "crop_embeddings.f32.npy", mode="w+", dtype=np.float32,
                                                       shape=(len(rows), values.shape[1]))
                completed = np.lib.format.open_memmap(completed_path, mode="w+", dtype=np.bool_, shape=(len(rows),))
                completed[:] = False
            embeddings[start:end] = values
            embeddings.flush()
            completed[start:end] = True
            completed.flush()
            enqueue()
            if start % (args.embedding_batch * 10) == 0 or end == len(rows):
                print(f"VLM embedded {end:,}/{len(rows):,} crops", flush=True)
    write_json(out / "embedding_run.json", {"model": args.vlm_model, "config_sha256": config_hash,
               "resolved_commit": getattr(model.config, "_commit_hash", None),
               "processor": processor.to_dict(), "crop_manifest_sha256": digest(out / "crops.jsonl"),
               "rows": len(rows), "precision": args.embedding_precision, "embedding_dim": embeddings.shape[1], "normalization": "unit L2 per crop"})


def semantic_prototypes(embeddings, feature_ids, width):
    """Equal-weight unit crop vectors; mean norm measures within-feature agreement."""
    embeddings = np.asarray(embeddings, np.float32)
    if not np.isfinite(embeddings).all():
        raise ValueError("Nonfinite crop embeddings")
    embeddings = embeddings / np.maximum(np.linalg.norm(embeddings, axis=1, keepdims=True), 1e-12)
    feature_ids = np.asarray(feature_ids, int)
    count = np.bincount(feature_ids, minlength=width)
    sums = np.zeros((width, embeddings.shape[1]), np.float32)
    np.add.at(sums, feature_ids, embeddings)
    mean = sums / np.maximum(count[:, None], 1)
    coherence = np.linalg.norm(mean, axis=1)
    prototypes = mean / np.maximum(coherence[:, None], 1e-12)
    return prototypes.astype(np.float32), count, coherence.astype(np.float32)


def reduce_features(args):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import umap

    out = Path(args.outdir)
    crops = [json.loads(line) for line in (out / "crops.jsonl").read_text().splitlines()]
    embeddings = np.load(out / "crop_embeddings.f32.npy", mmap_mode="r")
    completed = np.load(out / "embedding_completed.npy")
    if not completed.all() or len(embeddings) != len(crops):
        raise ValueError("Embedding stage is incomplete")
    run = json.loads((out / "run.json").read_text())
    width = run["width"]
    prototypes, count, coherence = semantic_prototypes(embeddings, [r["feature"] for r in crops], width)
    eligible = (count >= args.min_examples) & (coherence > 1e-8)
    if eligible.sum() < 4:
        raise ValueError("Need at least four supported dimensions for UMAP; do not fabricate missing points")
    coords = np.full((width, 2), np.nan, np.float32)
    reducer = umap.UMAP(n_neighbors=min(args.neighbors, int(eligible.sum()) - 1), min_dist=args.min_dist,
                        metric="cosine", random_state=args.seed, n_jobs=1, init="random")
    coords[eligible] = reducer.fit_transform(prototypes[eligible])
    np.save(out / "feature_semantic_embeddings.f32.npy", prototypes)
    np.savez_compressed(out / "feature_semantic_umap.npz", xy=coords, count=count, coherence=coherence, eligible=eligible)
    with (out / "feature_semantic_umap.csv").open("w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["feature", "umap_1", "umap_2", "crop_count", "coherence", "status"])
        for j in range(width):
            status = "eligible" if eligible[j] else "insufficient_examples" if count[j] < args.min_examples else "degenerate_prototype"
            writer.writerow([j, *([float(x) for x in coords[j]] if eligible[j] else ["", ""]), int(count[j]), float(coherence[j]), status])
    fig, ax = plt.subplots(figsize=(10, 8))
    plot = ax.scatter(*coords[eligible].T, c=coherence[eligible], cmap="viridis", s=18, alpha=.85)
    fig.colorbar(plot, ax=ax, label="Agreement among top-image crop embeddings")
    title = f"{args.dataset}: semantic evidence for {int(eligible.sum())} SAE dimensions"
    ax.set(xlabel="UMAP 1", ylabel="UMAP 2", title=textwrap.fill(title, width=85))
    fig.tight_layout()
    fig.savefig(out / "feature_semantic_umap.png", dpi=180)
    plt.close(fig)
    gallery = defaultdict(list)
    for r in crops:
        gallery[r["feature"]].append({k: r[k] for k in ("crop", "rank", "city", "pano", "heading", "activation", "bbox_native_view")})
    points = [{"feature": j, "x": float(coords[j, 0]) if eligible[j] else None,
               "y": float(coords[j, 1]) if eligible[j] else None, "count": int(count[j]),
               "coherence": float(coherence[j]), "examples": gallery[j]} for j in range(width)]
    write_json(out / "semantic_atlas.json", {"dataset": args.dataset, "model": args.vlm_model,
                                            "run": run, "points": points})
    template = Path(__file__).with_name("semantic_feature_umap.html").read_text()
    # Embedded data allows opening the gallery directly as a local file.
    data = json.dumps({"dataset": args.dataset, "model": args.vlm_model, "points": points}, ensure_ascii=False).replace("<", "\\u003c")
    (out / "index.html").write_text(template.replace("__ATLAS_DATA__", data))
    write_json(out / "reduction_run.json", {"included": int(eligible.sum()), "excluded": int((~eligible).sum()),
               "aggregation": "equal-weight mean of unit crop embeddings, then L2 normalization",
               "top": args.top, "min_examples": args.min_examples, "metric": "cosine", "seed": args.seed,
               "neighbors": reducer.n_neighbors, "min_dist": args.min_dist,
               "limitations": ["UMAP distances do not validate semantic labels", "Mean prototypes can conceal multiple concepts",
                               "Coherence is embedding agreement, not calibrated semantic purity", "Ranking is limited to the supplied manifest"]})
    print(f"Saved UMAP and interactive crop gallery to {out}", flush=True)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--manifest", required=True, help="Explicit JSONL image/feature mapping")
    p.add_argument("--checkpoint", required=True)
    p.add_argument("--dataset", required=True, help="Honest experiment identity; never infer from city count")
    p.add_argument("--outdir", required=True)
    p.add_argument("--sparse-activations", help="Optional existing NPZ with idx, val [image, patch, slot]")
    p.add_argument("--stage", choices=["all", "rank", "crop", "embed", "reduce"], default="all")
    p.add_argument("--device", default="cuda:1")
    p.add_argument("--grid", type=int, default=28)
    p.add_argument("--top", type=int, default=100)
    p.add_argument("--expected-cities", type=int, default=30)
    p.add_argument("--expected-width", type=int, default=1024)
    p.add_argument("--expected-k", type=int, default=8)
    p.add_argument("--pano-batch", type=int, default=4)
    p.add_argument("--crop-fraction", type=float, default=.5)
    p.add_argument("--margin", type=int, default=1)
    p.add_argument("--min-side", type=int, default=3)
    p.add_argument("--vlm-model", default="google/siglip-base-patch16-224")
    p.add_argument("--embedding-batch", type=int, default=64)
    p.add_argument("--embedding-precision", choices=["fp32", "fp16", "bf16"], default="fp32")
    p.add_argument("--embedding-workers", type=int, default=2)
    p.add_argument("--crop-workers", type=int, default=1)
    p.add_argument("--min-examples", type=int, default=20)
    p.add_argument("--neighbors", type=int, default=30)
    p.add_argument("--min-dist", type=float, default=.1)
    p.add_argument("--seed", type=int, default=42)
    args = p.parse_args()
    if not (0 < args.crop_fraction <= 1) or args.margin < 0 or args.min_side < 1:
        p.error("Invalid crop settings")
    if min(args.top, args.grid, args.pano_batch, args.embedding_batch, args.min_examples, args.embedding_workers, args.crop_workers) < 1 or args.neighbors < 2:
        p.error("Counts must be positive and neighbors at least two")
    out = Path(args.outdir).resolve()
    args.outdir = str(out)
    out.mkdir(parents=True, exist_ok=True)
    rows = read_manifest(Path(args.manifest).resolve(), args.grid)
    cities = sorted(set(r["city"] for r in rows))
    if args.expected_cities and len(cities) != args.expected_cities:
        p.error(f"Expected {args.expected_cities} cities, found {len(cities)}; use an explicitly labeled experiment")
    meta = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    model_type = str(meta.get("model_type", "batch_topk")).lower().replace("-", "_")
    if model_type not in {"batch_topk", "batchtopk"}:
        p.error(f"This experiment requires BatchTopK; checkpoint is {model_type}")
    spec = {"dataset": args.dataset, "manifest_sha256": digest(Path(args.manifest)),
            "checkpoint_sha256": digest(Path(args.checkpoint)), "checkpoint": str(Path(args.checkpoint).resolve()),
            "width": int(meta["K"]), "k": int(meta.get("k", meta.get("topk", 8))), "cities": cities, "images": len(rows),
            "grid": args.grid, "top": args.top, "crop_fraction": args.crop_fraction, "margin": args.margin,
            "min_side": args.min_side, "vlm_model": args.vlm_model, "embedding_precision": args.embedding_precision,
            "sparse_source_sha256": digest(Path(args.sparse_activations)) if args.sparse_activations else None,
            "activation": "existing sparse archive (original batching)" if args.sparse_activations else "joint BatchTopK per complete panorama",
            "ranking": "maximum sparse activation per distinct directional image; ties by manifest order"}
    if ((args.expected_width and spec["width"] != args.expected_width)
            or (args.expected_k and spec["k"] != args.expected_k)):
        p.error("Checkpoint width or sparsity differs from expected settings")
    if (out / "run.json").exists() and json.loads((out / "run.json").read_text()) != spec:
        p.error("Output directory belongs to a different source/configuration; use a new directory")
    write_json(out / "run.json", spec)
    # A user may put the input manifest in the result directory. Do not mutate
    # the input during normalization: that would invalidate its source hash.
    if Path(args.manifest).resolve() != out / "images.jsonl":
        write_jsonl(out / "images.jsonl", rows)
    source = None
    if args.stage in {"all", "rank", "crop"}:
        source = ActivationSource(rows, args)
    if args.stage in {"all", "rank"}:
        rank_images(rows, args, source)
    if args.stage in {"all", "crop"}:
        make_crops(rows, args, source)
    if source is not None:
        del source
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    if args.stage in {"all", "embed"}:
        embed_crops(args)
    if args.stage in {"all", "reduce"}:
        reduce_features(args)


if __name__ == "__main__":
    main()
