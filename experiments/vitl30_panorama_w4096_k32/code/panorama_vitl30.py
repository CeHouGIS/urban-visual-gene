"""Resumable 30-city ViT-L/16 panorama-context BatchTopK experiment.

Extraction stores contextualized [panorama, heading, patch, channel] arrays.
Training visits all 784 patch positions over 16 epochs, with one 49-patch
contiguous window per view and epoch. The 128-panorama/city validation split
is fixed before training and never used to update the SAE.
"""
from __future__ import annotations

import argparse
import csv
import fcntl
import hashlib
import json
import os
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from PIL import Image
from torchvision.transforms import v2
from transformers import AutoModel

from sae_experiments.io_utils import digest, save_torch, write_json
from sae_experiments.paths import REPO_ROOT
from sae_experiments.models.base_sae import BatchTopKSAE
from sae_experiments.models.panorama_context import mix_panorama_context

ROOT = REPO_ROOT
OUTPUT = ROOT / "outputs/experiments/nas_30city_vitl_panorama_w4096_k32"
SOURCE = OUTPUT / "source_cohort"
TOKEN_ROOT = Path("/host/root/mnt/nas/huangyj/GoogleSV/urban_visual_gene/nas_30city_vitl_panorama_w4096_k32")
MODEL_DIR = Path("/root/.cache/modelscope/models/facebook--dinov3-vitl16-pretrain-lvd1689m/snapshots/master")
HEADINGS = (0, 90, 180, 270)
PATCHES = 784
DIM = 1024
WIDTH = 4096
K = 32
PATCHES_PER_EPOCH = 49
EPOCHS = 16


def cohort(source: Path) -> list[str]:
    report = json.loads((source / "manifests/batch.json").read_text())
    names = report["cities"]
    if len(names) != 30 or len(set(names)) != 30:
        raise ValueError("Expected the fixed 30-city source cohort")
    return names


def filtered_city(source: Path, output: Path, city: str) -> tuple[pd.DataFrame, str]:
    slug = city.replace("/", "__")
    manifest = source / "manifests" / f"{slug}.parquet"
    quality = source / "quality" / f"{slug}.parquet"
    quality_report = source / "quality" / f"{slug}.json"
    if json.loads(quality_report.read_text())["manifest_sha256"] != digest(manifest):
        raise ValueError(f"QC identity mismatch: {city}")
    frame = pd.read_parquet(manifest)
    qc = pd.read_parquet(quality, columns=["panoid", "is_bad"])
    if len(frame) % 4 or frame.heading.to_numpy().reshape(-1, 4).tolist() != [list(HEADINGS)] * (len(frame) // 4):
        raise ValueError(f"Four-heading order mismatch: {city}")
    if not np.all(frame.panoid.to_numpy().reshape(-1, 4) == frame.panoid.to_numpy()[::4, None]):
        raise ValueError(f"Panorama identity mismatch: {city}")
    if qc.panoid.duplicated().any() or len(qc) != len(frame) // 4:
        raise ValueError(f"QC panorama count mismatch: {city}")
    good = set(qc.loc[~qc.is_bad, "panoid"].astype(str))
    frame = frame[frame.panoid.astype(str).isin(good)].copy().reset_index(drop=True)
    frame["context_pano_index"] = np.repeat(np.arange(len(frame) // 4), 4)
    target = output / "manifests" / f"{slug}.parquet"
    target.parent.mkdir(parents=True, exist_ok=True)
    if target.exists():
        old = pd.read_parquet(target)
        if not old.equals(frame):
            raise ValueError(f"Existing filtered manifest differs: {city}")
    else:
        part = target.with_suffix(".partial.parquet")
        frame.to_parquet(part, index=False)
        os.replace(part, target)
    identity = hashlib.sha256((digest(manifest) + digest(quality) + digest(target)).encode()).hexdigest()
    return frame, identity


def preprocess():
    return v2.Compose([
        v2.Resize((448, 448), interpolation=v2.InterpolationMode.BICUBIC, antialias=True),
        v2.ToImage(), v2.ToDtype(torch.float32, scale=True),
        v2.Normalize(mean=(.485, .456, .406), std=(.229, .224, .225)),
    ])


def extract_city(args, city: str, model, transform, pool) -> None:
    frame, identity = filtered_city(args.source, args.output, city)
    slug = city.replace("/", "__")
    folder = args.token_root / slug
    folder.mkdir(parents=True, exist_ok=True)
    final = folder / "context.f16.npy"
    partial = folder / "context.partial.npy"
    progress_path = folder / "progress.json"
    report_path = folder / "report.json"
    n_panos = len(frame) // 4
    shape = (n_panos, 4, PATCHES, DIM)
    if final.exists():
        report = json.loads(report_path.read_text())
        arr = np.load(final, mmap_mode="r")
        if arr.shape != shape or arr.dtype != np.float16 or report["identity"] != identity:
            raise ValueError(f"Existing tokens do not match inputs: {city}")
        print("complete", city, n_panos, flush=True)
        return
    if partial.exists() and progress_path.exists():
        progress = json.loads(progress_path.read_text())
        arr = np.load(partial, mmap_mode="r+")
        if arr.shape != shape or progress["identity"] != identity:
            raise ValueError(f"Unsafe extraction resume: {city}")
        start = int(progress["completed_panos"])
    elif not partial.exists() and not progress_path.exists():
        arr = np.lib.format.open_memmap(partial, mode="w+", dtype=np.float16, shape=shape)
        start = 0
    else:
        raise ValueError(f"Partial extraction has no matching progress: {city}")
    prefix = 1 + int(getattr(model.config, "num_register_tokens", 0))
    paths = frame.image_path.tolist()

    def read(path):
        with Image.open(path) as image:
            return transform(image.convert("RGB"))

    with torch.inference_mode():
        for at in range(start, n_panos, args.panos_per_batch):
            stop = min(at + args.panos_per_batch, n_panos)
            pixels = torch.stack(list(pool.map(read, paths[at * 4:stop * 4]))).to(args.device)
            if args.device.startswith("cuda"):
                pixels = pixels.to(torch.bfloat16)
            hidden = model(pixel_values=pixels).last_hidden_state[:, prefix:].float()
            if hidden.shape[1:] != (PATCHES, DIM):
                raise ValueError(f"Unexpected DINOv3-L tokens: {tuple(hidden.shape)}")
            hidden = torch.nn.functional.normalize(hidden, dim=-1)
            context = mix_panorama_context(hidden.reshape(-1, 4, PATCHES, DIM), grid_size=28)
            if not torch.isfinite(context).all():
                raise ValueError(f"Non-finite contextualized tokens: {city}, {at}")
            arr[at:stop] = context.cpu().numpy().astype(np.float16)
            del pixels, hidden, context
            if ((at - start) // args.panos_per_batch + 1) % args.flush_every == 0 or stop == n_panos:
                arr.flush()
                write_json(progress_path, {"city": city, "identity": identity,
                                         "completed_panos": stop, "total_panos": n_panos})
                print("extract", city, stop, "/", n_panos, flush=True)
    arr.flush()
    os.replace(partial, final)
    progress_path.unlink()
    write_json(report_path, {"city": city, "identity": identity, "panoramas": n_panos,
                            "images": len(frame), "shape": list(shape), "model": str(args.model_dir),
                            "context_weight": .25, "image_size": 448,
                            "model_weights_sha256": args.model_hash})


def extract(args):
    if args.device.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError("CUDA unavailable")
    names = cohort(args.source)
    weights = args.model_dir / "model.safetensors"
    if not weights.is_file():
        raise FileNotFoundError(weights)
    args.model_hash = digest(weights)
    model = AutoModel.from_pretrained(args.model_dir, local_files_only=True).eval().requires_grad_(False).to(args.device)
    if model.config.hidden_size != DIM or model.config.patch_size != 16:
        raise ValueError("Expected DINOv3 ViT-L/16")
    if args.device.startswith("cuda"):
        model = model.to(torch.bfloat16)
    transform = preprocess()
    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        for city in names[args.city_start:args.city_stop]:
            extract_city(args, city, model, transform, pool)


def training_data(args):
    if args.cached_feature_map:
        return cached_training_data(args)
    cities = []
    for city in cohort(args.source):
        frame, identity = filtered_city(args.source, args.output, city)
        slug = city.replace("/", "__")
        path = args.token_root / slug / "context.f16.npy"
        report_path = path.parent / "report.json"
        if not path.exists() or not report_path.exists():
            raise FileNotFoundError(f"Context features incomplete: {city}")
        report = json.loads(report_path.read_text())
        arr = np.load(path, mmap_mode="r")
        if (report["identity"] != identity or report["model_weights_sha256"] != args.model_hash
                or arr.shape != (len(frame) // 4, 4, PATCHES, DIM) or arr.dtype != np.float16):
            raise ValueError(f"Feature provenance mismatch: {city}")
        ids = np.arange(arr.shape[0])
        seed = int.from_bytes(hashlib.sha256(f"{args.seed}/{city}".encode()).digest()[:8], "little")
        val = np.random.default_rng(seed).choice(ids, args.val_panos, replace=False)
        train = np.setdiff1d(ids, val)
        cities.append((city, arr, train, val, identity))
    return cities


def cached_training_data(args):
    """Use precomputed contextualized features without requiring new DINO work."""
    manifest = json.loads(args.cached_feature_map.read_text())
    if (manifest.get("input_dim"), manifest.get("grid"), manifest.get("context_weight")) != (DIM, 28, .25):
        raise ValueError("Existing cache contract differs from ViT-L panorama inputs")
    if len(manifest["cities"]) != 30:
        raise ValueError("The cache-backed main experiment requires thirty cities")
    cities = []
    audit_rows = []
    for item in manifest["cities"]:
        city = item["city"]
        path = Path(item["feature_path"])
        arr = np.load(path, mmap_mode="r")
        if arr.dtype != np.float16:
            raise ValueError(f"Cache dtype mismatch: {city}")
        if arr.ndim == 4 and arr.shape[1:] == (4, PATCHES, DIM):
            arr = arr.reshape(-1, DIM)
        if arr.ndim != 2 or arr.shape[1] != DIM or arr.shape[0] % (4 * PATCHES):
            raise ValueError(f"Cache shape or panorama alignment mismatch: {city}")
        if arr.shape[0] != item["feature_rows"]:
            raise ValueError(f"Cached row count changed: {city}")
        n = arr.shape[0] // (4 * PATCHES)
        if n <= args.val_panos:
            raise ValueError(f"Insufficient cached panoramas: {city}")
        probe = np.asarray(arr[np.linspace(0, len(arr) - 1, 16, dtype=np.int64)], dtype=np.float32)
        if not np.isfinite(probe).all() or not np.allclose(np.linalg.norm(probe, axis=1), 1, atol=.02):
            raise ValueError(f"Cache tokens are not finite normalized vectors: {city}")
        stat = path.stat()
        identity = hashlib.sha256(json.dumps([str(path), stat.st_size, stat.st_mtime_ns,
                                               arr.shape, manifest["context_weight"]]).encode()).hexdigest()
        arr = arr.reshape(n, 4, PATCHES, DIM)
        ids = np.arange(n)
        seed = int.from_bytes(hashlib.sha256(f"{args.seed}/{city}".encode()).digest()[:8], "little")
        val = np.random.default_rng(seed).choice(ids, args.val_panos, replace=False)
        train = np.setdiff1d(ids, val)
        cities.append((city, arr, train, val, identity))
        audit_rows.append({"city": city, "feature_path": str(path), "panoramas": n,
                           "train_panoramas": len(train), "validation_panoramas": len(val),
                           "identity": identity, "pano_id_mapping_available": item.get("pano_id_mapping_available", False)})
    write_json(args.output / "cached_feature_audit.json", {"source": str(args.cached_feature_map),
               "cities": audit_rows, "panoramas": sum(len(c[2]) + len(c[3]) for c in cities),
               "extraction_required": False})
    return cities


def orders_for_epoch(cities, seed, epoch, steps):
    orders = []
    for city, _, train, _, _ in cities:
        seed_city = int.from_bytes(hashlib.sha256(f"{seed}/{epoch}/{city}".encode()).digest()[:8], "little")
        rng = np.random.default_rng(seed_city)
        chunks = []
        while sum(len(x) for x in chunks) < steps:
            chunks.append(rng.permutation(train))
        orders.append(np.concatenate(chunks)[:steps])
    return orders


def sample_step(cities, orders, step, patch_start):
    stop = patch_start + PATCHES_PER_EPOCH
    blocks = [np.asarray(arr[ids[step], :, patch_start:stop, :], dtype=np.float32).reshape(-1, DIM)
              for (_, arr, _, _, _), ids in zip(cities, orders)]
    return torch.from_numpy(np.concatenate(blocks, axis=0))


def prefetched_steps(cities, orders, start, stop, patch_start, workers):
    """Read independent city windows concurrently and overlap reads with compute."""
    def read_city(pair, step):
        (_, arr, _, _, _), ids = pair
        return np.array(arr[ids[step], :, patch_start:patch_start + PATCHES_PER_EPOCH],
                        dtype=np.float32).reshape(-1, DIM)

    with ThreadPoolExecutor(max_workers=workers) as readers:
        def submit(step):
            return [readers.submit(read_city, pair, step) for pair in zip(cities, orders)]
        pending = submit(start) if start < stop else []
        for step in range(start, stop):
            blocks = [future.result() for future in pending]
            pending = submit(step + 1) if step + 1 < stop else []
            yield step, torch.from_numpy(np.concatenate(blocks, axis=0))


@torch.inference_mode()
def validate(model, cities, device):
    model.eval()
    losses, cosines = [], []
    for _, arr, _, val, _ in cities:
        for pid in val[:2]:
            for start in range(0, PATCHES, 49):
                x = torch.from_numpy(np.asarray(arr[pid, :, start:start + 49], dtype=np.float32).reshape(-1, DIM)).to(device)
                with torch.autocast("cuda", dtype=torch.bfloat16, enabled=device.startswith("cuda")):
                    _, rec = model(x)
                losses.append((rec.float() - x).square().sum(-1).mean().item())
                cosines.append(torch.nn.functional.cosine_similarity(rec.float(), x).mean().item())
    return {"val_squared_l2": float(np.mean(losses)), "val_cosine": float(np.mean(cosines)),
            "val_panoramas_per_city": 2, "val_patch_positions": PATCHES}


def train(args):
    args.output.mkdir(parents=True, exist_ok=True)
    lock = (args.output / "train.lock").open("w")
    try:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    except BlockingIOError:
        raise RuntimeError("Another trainer already owns this output directory")
    torch.set_num_threads(args.cpu_threads)
    args.model_hash = digest(args.model_dir / "model.safetensors")
    cities = training_data(args)
    steps = max(len(item[2]) for item in cities)
    identity = {"cities": [(c, i) for c, _, _, _, i in cities], "weights": args.model_hash,
                "seed": args.seed, "width": WIDTH, "k": K, "epochs": EPOCHS,
                "val_panos": args.val_panos, "patches_per_epoch": PATCHES_PER_EPOCH,
                "steps_per_epoch": steps}
    args.output.mkdir(parents=True, exist_ok=True)
    write_json(args.output / "experiment.json", {"identity": identity, "train_panoramas": sum(len(c[2]) for c in cities),
               "val_panoramas": sum(len(c[3]) for c in cities), "cities": len(cities),
               "train_patch_coverage": "all 784 positions across 16 epochs"})
    torch.manual_seed(args.seed)
    torch.backends.cuda.matmul.allow_tf32 = True
    model = BatchTopKSAE(DIM, WIDTH, K).to(args.device)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-5)
    last = args.output / "checkpoint_last.pt"
    best = args.output / "model_best.pt"
    epoch0, step0, best_loss, history = 0, 0, float("inf"), []
    if last.exists():
        state = torch.load(last, map_location=args.device, weights_only=False)
        if state["identity"] != identity:
            raise ValueError("Checkpoint identity differs from current experiment")
        model.load_state_dict(state["model"])
        opt.load_state_dict(state["optimizer"])
        epoch0, step0, best_loss, history = (state[k] for k in ("epoch", "step", "best_loss", "history"))
    def checkpoint(epoch, step):
        save_torch(last, {"model": model.state_dict(), "optimizer": opt.state_dict(),
                          "identity": identity, "epoch": epoch, "step": step,
                          "best_loss": best_loss, "history": history})
    for epoch in range(epoch0, EPOCHS):
        model.train()
        orders = orders_for_epoch(cities, args.seed, epoch, steps)
        patch_start = epoch * PATCHES_PER_EPOCH
        started = time.monotonic()
        first_step = step0 if epoch == epoch0 else 0
        for step, batch in prefetched_steps(cities, orders, first_step, steps, patch_start, args.workers):
            x = batch.to(args.device)
            opt.zero_grad(set_to_none=True)
            with torch.autocast("cuda", dtype=torch.bfloat16, enabled=args.device.startswith("cuda")):
                acts, rec = model(x)
                loss = (rec.float() - x).square().sum(-1).mean()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.)
            opt.step()
            model.normalize_decoder_()
            if (step + 1) % args.log_every == 0 or step == 0:
                row = {"epoch": epoch + 1, "step": step + 1, "train_squared_l2": float(loss.detach()),
                       "train_active_features": float((acts > 0).sum(-1).float().mean())}
                history.append(row)
                print("train", row, flush=True)
                completed = step - first_step + 1
                seconds_per_step = (time.monotonic() - started) / completed
                write_json(args.output / "training_progress.json", {
                    **row, "state": "running", "pid": os.getpid(),
                    "steps_per_epoch": steps, "epochs": EPOCHS,
                    "completed_steps": epoch * steps + step + 1,
                    "total_steps": EPOCHS * steps, "seconds_per_step": seconds_per_step,
                    "estimated_seconds_remaining": (EPOCHS * steps - epoch * steps - step - 1) * seconds_per_step,
                    "features_reused": bool(args.cached_feature_map), "device": args.device})
            if (step + 1) % args.save_every == 0:
                checkpoint(epoch, step + 1)
        metrics = validate(model, cities, args.device)
        print("validation", epoch + 1, metrics, flush=True)
        if metrics["val_squared_l2"] < best_loss:
            best_loss = metrics["val_squared_l2"]
            save_torch(best, {"state": {k: v.detach().cpu() for k, v in model.state_dict().items()},
                              "D": DIM, "K": WIDTH, "k": K, "model_type": "batch_topk",
                              "identity": identity, "metrics": metrics, "epoch": epoch + 1})
        history[-1].update(metrics)
        checkpoint(epoch + 1, 0)
        with (args.output / "history.csv").open("w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=["epoch", "step", "train_squared_l2",
                                                   "train_active_features", "val_squared_l2", "val_cosine",
                                                   "val_panoramas_per_city", "val_patch_positions"])
            writer.writeheader()
            writer.writerows(history)
        write_json(args.output / "training_report.json", {"status": "complete" if epoch + 1 == EPOCHS else "running",
                    "epoch": epoch + 1, "epochs": EPOCHS, "best_val_squared_l2": best_loss,
                    "model": str(best), "identity": identity})
        step0 = 0


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("command", choices=["extract", "train", "status"])
    p.add_argument("--source", type=Path, default=SOURCE)
    p.add_argument("--output", type=Path, default=OUTPUT)
    p.add_argument("--token-root", type=Path, default=TOKEN_ROOT)
    p.add_argument("--model-dir", type=Path, default=MODEL_DIR)
    p.add_argument("--cached-feature-map", type=Path,
                   help="verified precomputed contextualized cache map; train without DINO extraction")
    p.add_argument("--device", default="cuda:2")
    p.add_argument("--city-start", type=int, default=0)
    p.add_argument("--city-stop", type=int, default=30)
    p.add_argument("--panos-per-batch", type=int, default=2)
    p.add_argument("--workers", type=int, default=8)
    p.add_argument("--cpu-threads", type=int, default=8)
    p.add_argument("--flush-every", type=int, default=50)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--val-panos", type=int, default=128)
    p.add_argument("--lr", type=float, default=3e-4)
    p.add_argument("--save-every", type=int, default=100)
    p.add_argument("--log-every", type=int, default=25)
    args = p.parse_args()
    if not 0 <= args.city_start < args.city_stop <= 30 or min(args.panos_per_batch, args.workers,
            args.flush_every, args.val_panos, args.save_every, args.log_every) <= 0:
        p.error("Invalid city slice or positive count")
    if args.command == "extract":
        extract(args)
    elif args.command == "train":
        train(args)
    else:
        for city in cohort(args.source):
            slug = city.replace("/", "__")
            path = args.token_root / slug / "report.json"
            partial = args.token_root / slug / "progress.json"
            state = json.loads(path.read_text()) if path.exists() else json.loads(partial.read_text()) if partial.exists() else None
            print(city, "complete" if path.exists() else "partial" if partial.exists() else "pending",
                  state.get("completed_panos", state.get("panoramas")) if state else "", flush=True)


if __name__ == "__main__":
    main()
