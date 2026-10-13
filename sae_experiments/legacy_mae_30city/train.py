"""W4096/k32 BatchTopK on thirty-city DINOv3-B/16 patch tokens.

The manifest directory defines the cohort. A NAS batch.json marks replacement
images explicitly; legacy manifests retain their original strict QC count.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import runpy
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch.nn import functional as F

from sae_experiments.models.base_sae import BatchTopKSAE

ROOT = Path(__file__).resolve().parents[2]
CITIES = tuple(runpy.run_path(str(ROOT / "archive/non_batchtopk/scripts/multicity/config.py"))["CITIES"])
OLD_ROOT = Path("/workplace/urban_visual_gene/outputs/experiments/dinov3_multicity")
DEFAULT_DATA = OLD_ROOT / "feature_mae_n30x12800_qc"
DEFAULT_TOKENS = OLD_ROOT / "patch_sae_topk32_w512_n30x12800"
DEFAULT_OUTPUT = ROOT / "outputs/experiments/feature_mae_30city_batchtopk_w4096_k32"
EXPECTED_IMAGES = 378818


def digest(path):
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def write_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix(path.suffix + ".tmp")
    temp.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n")
    os.replace(temp, path)


def save_torch(path, value):
    temp = path.with_suffix(path.suffix + ".tmp")
    torch.save(value, temp)
    os.replace(temp, path)


def audit(data_root, token_root, seed, val_panos, manifest_root=None):
    manifest_root = manifest_root or data_root / "filtered_manifests"
    batch_path = manifest_root / "batch.json"
    nas_batch = json.loads(batch_path.read_text()) if batch_path.is_file() else None
    city_names = tuple(nas_batch["cities"]) if nas_batch else CITIES
    expected_images = int(nas_batch["images"]) if nas_batch else EXPECTED_IMAGES
    if len(city_names) != 30 or len(set(city_names)) != 30:
        raise ValueError("Legacy config must contain exactly thirty cities")
    cities, details, missing = [], [], []
    for name in city_names:
        slug = name.replace("/", "__")
        manifest = manifest_root / (slug + ".parquet")
        tokens = token_root / "patch_tokens" / slug / "tokens.f16.npy"
        token_report_path = tokens.parent / "token_report.json"
        if not manifest.is_file() or not tokens.is_file():
            missing.append({"city": name, "manifest_missing": not manifest.is_file(),
                            "tokens_missing": not tokens.is_file(),
                            "manifest_path": str(manifest), "tokens_path": str(tokens)})
            continue
        frame = pd.read_parquet(manifest, columns=["source_image_index", "panoid"])
        ids = frame.source_image_index.to_numpy(np.int64)
        if not len(ids) or frame.panoid.isna().any() or len(np.unique(ids)) != len(ids):
            raise ValueError(name + ": invalid QC manifest")
        array = np.load(tokens, mmap_mode="r")
        if array.ndim != 3 or array.shape[1:] != (196, 768) or array.dtype != np.float16:
            raise ValueError(name + ": expected float16 [N,196,768], got " + str(array.shape))
        if ids.min() < 0 or ids.max() >= len(array):
            raise ValueError(name + ": source_image_index outside token array")
        sample = np.asarray(array[ids[:min(4, len(ids))]], dtype=np.float32)
        if not np.isfinite(sample).all() or not np.allclose(np.linalg.norm(sample, axis=-1), 1, atol=.03):
            raise ValueError(name + ": expected L2-normalized DINOv3 patch tokens")
        mh = digest(manifest)
        if nas_batch:
            if not token_report_path.is_file():
                raise ValueError(f"{name}: token provenance report missing")
            token_report = json.loads(token_report_path.read_text())
            if token_report.get("manifest_sha256") != mh:
                raise ValueError(f"{name}: token cache and image manifest mismatch")
        panos = frame.panoid.astype(str).to_numpy()
        quality_path = (manifest_root.parent / "quality" / (slug + ".parquet")) if nas_batch else None
        quality_sha256 = None
        if nas_batch:
            quality_report_path = quality_path.with_suffix(".json")
            if not quality_path.is_file() or not quality_report_path.is_file():
                missing.append({"city": name, "quality_missing": True, "quality_path": str(quality_path)})
                continue
            quality_report = json.loads(quality_report_path.read_text())
            if quality_report.get("manifest_sha256") != digest(manifest):
                raise ValueError(f"{name}: quality screening and manifest mismatch")
            quality_sha256 = digest(quality_path)
            quality = pd.read_parquet(quality_path, columns=["panoid", "is_bad"])
            if len(quality) != len(np.unique(panos)) or quality.panoid.duplicated().any():
                raise ValueError(f"{name}: quality screening has wrong panorama identities")
            keep_panos = set(quality.loc[~quality.is_bad, "panoid"].astype(str))
            selected = np.isin(panos, list(keep_panos))
            ids = ids[selected]
            panos = panos[selected]
        unique = np.unique(panos)
        if len(unique) <= val_panos:
            raise ValueError(name + ": insufficient panoramas for validation")
        local_seed = int.from_bytes(hashlib.sha256((str(seed) + name).encode()).digest()[:8], "little")
        selected = np.random.default_rng(local_seed).choice(unique, val_panos, replace=False)
        is_val = np.isin(panos, selected)
        train_ids, val_ids = ids[~is_val], ids[is_val]
        cities.append((name, array, train_ids, val_ids))
        details.append({"city": name, "images": len(ids), "panoramas": len(unique),
                        "manifest_images": len(frame), "excluded_images": len(frame) - len(ids),
                        "train_images": len(train_ids), "val_images": len(val_ids),
                        "manifest_sha256": mh, "quality_sha256": quality_sha256,
                        "token_path": str(tokens.resolve()),
                        "token_shape": list(array.shape)})
    total = sum(row["images"] for row in details)
    identity = hashlib.sha256(json.dumps([(x["city"], x["manifest_sha256"], x["quality_sha256"], x["token_shape"])
                                          for x in details]).encode()).hexdigest()
    manifest_total = sum(row["manifest_images"] for row in details)
    status = "missing_inputs" if missing else "verified" if (manifest_total == expected_images and total > 0) else "wrong_image_count"
    return cities, {"status": status, "expected_cities": 30, "expected_images": expected_images,
                    "actual_images": total, "manifest_images": manifest_total,
                    "excluded_images": manifest_total - total, "missing": missing, "cities": details,
                    "dataset_identity": identity, "seed": seed, "val_panos_per_city": val_panos,
                    "batch_id": nas_batch.get("batch_id") if nas_batch else "feature_mae_original_qc",
                    "images_are_original_feature_mae_images": False if nas_batch else True,
                    "feature_contract": "DINOv3 ViT-B/16, 224px, 14x14x768, L2-normalized patches"}


def batch(cities, rng, images_per_city, patches_per_image, validation=False):
    blocks = []
    for _, array, train_ids, val_ids in cities:
        ids = val_ids if validation else train_ids
        selected = rng.choice(ids, images_per_city, replace=len(ids) < images_per_city)
        patch_ids = rng.integers(0, 196, (images_per_city, patches_per_image))
        blocks.append(np.asarray(array[selected[:, None], patch_ids], dtype=np.float32).reshape(-1, 768))
    return np.concatenate(blocks, axis=0)


def epoch_orders(cities, seed, epoch, target_images):
    """Visit each training image once before repeating short cities."""
    orders = []
    for name, _, train_ids, _ in cities:
        local = int.from_bytes(hashlib.sha256((str(seed) + name + str(epoch)).encode()).digest()[:8], "little")
        source = np.random.default_rng(local)
        pieces, remaining = [], target_images
        while remaining:
            permuted = source.permutation(train_ids)
            take = min(remaining, len(permuted))
            pieces.append(permuted[:take])
            remaining -= take
        orders.append(np.concatenate(pieces))
    return orders


def training_batch(cities, orders, rng, step, images_per_city, patches_per_image):
    blocks = []
    start = step * images_per_city
    for (_, array, _, _), ids in zip(cities, orders):
        selected = ids[start:start + images_per_city]
        patch_ids = rng.integers(0, 196, (images_per_city, patches_per_image))
        blocks.append(np.asarray(array[selected[:, None], patch_ids], dtype=np.float32).reshape(-1, 768))
    return np.concatenate(blocks, axis=0)


@torch.inference_mode()
def validate(model, arrays, device):
    model.eval()
    stats = []
    for values in arrays:
        x = torch.from_numpy(values).to(device)
        with torch.autocast("cuda", dtype=torch.bfloat16, enabled=device.startswith("cuda")):
            a, y = model(x)
        stats.append([(y.float() - x).square().sum(-1).mean().item(),
                      F.cosine_similarity(y.float(), x, dim=-1).mean().item(),
                      (a > 0).sum(-1).float().mean().item()])
    loss, cosine, active = np.mean(stats, axis=0)
    return {"val_squared_l2": float(loss), "val_cosine": float(cosine),
            "val_active_features": float(active)}


def train(args, cities, report):
    if args.device.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError("CUDA unavailable")
    torch.manual_seed(args.seed)
    torch.backends.cuda.matmul.allow_tf32 = True
    model = BatchTopKSAE(768, 4096, 32).to(args.device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-5)
    identity = {"data": report["dataset_identity"], "width": 4096, "k": 32,
                "seed": args.seed, "images_per_city": args.images_per_city,
                "patches_per_image": args.patches_per_image, "steps_per_epoch": args.steps_per_epoch,
                "val_panos": args.val_panos}
    last, best = args.output / "checkpoint_last.pt", args.output / "model_best.pt"
    rng = np.random.default_rng(args.seed)
    val_rng = np.random.default_rng(args.seed + 10000)
    val_arrays = [batch(cities, val_rng, args.val_images_per_city, args.val_patches_per_image, True)
                  for _ in range(args.val_batches)]
    if args.steps_per_epoch * args.images_per_city < max(len(row[2]) for row in cities):
        raise ValueError("steps_per_epoch * images_per_city must cover every training image")
    epoch0, step0, best_loss, history = 0, 0, float("inf"), []
    if last.exists():
        state = torch.load(last, map_location=args.device, weights_only=False)
        if state["identity"] != identity:
            raise ValueError("checkpoint data/config mismatch; refusing unsafe resume")
        model.load_state_dict(state["model"])
        optimizer.load_state_dict(state["optimizer"])
        rng.bit_generator.state = state["rng_state"]
        epoch0, step0, best_loss, history = (state[k] for k in ("epoch", "step", "best_loss", "history"))
        if epoch0 == args.epochs:
            print("training already complete", flush=True)
            return
    def checkpoint(epoch, step):
        save_torch(last, {"model": model.state_dict(), "optimizer": optimizer.state_dict(),
                          "rng_state": rng.bit_generator.state, "epoch": epoch, "step": step,
                          "best_loss": best_loss, "history": history, "identity": identity})
    for epoch in range(epoch0, args.epochs):
        model.train()
        orders = epoch_orders(cities, args.seed, epoch, args.steps_per_epoch * args.images_per_city)
        for step in range(step0 if epoch == epoch0 else 0, args.steps_per_epoch):
            x = torch.from_numpy(training_batch(cities, orders, rng, step, args.images_per_city,
                                                args.patches_per_image)).to(args.device)
            optimizer.zero_grad(set_to_none=True)
            with torch.autocast("cuda", dtype=torch.bfloat16, enabled=args.device.startswith("cuda")):
                acts, reconstruction = model(x)
                loss = (reconstruction.float() - x).square().sum(-1).mean()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            model.normalize_decoder_()
            history.append({"epoch": epoch + 1, "step": step + 1,
                            "train_squared_l2": float(loss.detach()),
                            "train_active_features": float((acts > 0).sum(-1).float().mean())})
            if (step + 1) % args.log_every == 0:
                print("epoch", epoch + 1, "step", step + 1, "loss", history[-1]["train_squared_l2"], flush=True)
            if (step + 1) % args.save_every == 0:
                checkpoint(epoch, step + 1)
        metrics = validate(model, val_arrays, args.device)
        history[-1].update(metrics)
        print("validation", epoch + 1, metrics, flush=True)
        if metrics["val_squared_l2"] < best_loss:
            best_loss = metrics["val_squared_l2"]
            save_torch(best, {"state": {k: v.detach().cpu() for k, v in model.state_dict().items()},
                              "D": 768, "K": 4096, "k": 32, "model_type": "batch_topk",
                              "identity": identity, "metrics": metrics, "epoch": epoch + 1})
        checkpoint(epoch + 1, 0)
        step0 = 0
        with (args.output / "history.csv").open("w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=["epoch", "step", "train_squared_l2",
                                                   "train_active_features", "val_squared_l2",
                                                   "val_cosine", "val_active_features"])
            writer.writeheader(); writer.writerows(history)
        write_json(args.output / "training_report.json", {"status": "complete" if epoch + 1 == args.epochs else "running",
                    "epoch": epoch + 1, "epochs": args.epochs, "best_val_squared_l2": best_loss,
                    "model": str(best), "data_identity": identity["data"]})


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--data-root", type=Path, default=DEFAULT_DATA)
    p.add_argument("--token-root", type=Path, default=DEFAULT_TOKENS)
    p.add_argument("--manifest-root", type=Path, help="recovered QC manifest directory, if original data root is unavailable")
    p.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    p.add_argument("--device", default="cuda:1")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--val-panos", type=int, default=128)
    p.add_argument("--images-per-city", type=int, default=4)
    p.add_argument("--patches-per-image", type=int, default=32)
    p.add_argument("--steps-per-epoch", type=int, default=3200)
    p.add_argument("--epochs", type=int, default=10)
    p.add_argument("--val-images-per-city", type=int, default=2)
    p.add_argument("--val-patches-per-image", type=int, default=64)
    p.add_argument("--val-batches", type=int, default=8)
    p.add_argument("--lr", type=float, default=3e-4)
    p.add_argument("--log-every", type=int, default=25)
    p.add_argument("--save-every", type=int, default=100)
    p.add_argument("--audit-only", action="store_true")
    args = p.parse_args()
    if min(args.val_panos, args.images_per_city, args.patches_per_image, args.steps_per_epoch,
           args.epochs, args.val_images_per_city, args.val_patches_per_image,
           args.val_batches, args.log_every, args.save_every) <= 0:
        p.error("all sampling/training counts must be positive")
    if max(args.patches_per_image, args.val_patches_per_image) > 196:
        p.error("patches per image cannot exceed 196")
    args.output.mkdir(parents=True, exist_ok=True)
    cities, report = audit(args.data_root, args.token_root, args.seed, args.val_panos, args.manifest_root)
    write_json(args.output / "data_audit.json", report)
    print("data audit:", report["status"], report["actual_images"], "/", report["expected_images"], flush=True)
    if report["status"] != "verified":
        raise SystemExit(2)
    if not args.audit_only:
        train(args, cities, report)


if __name__ == "__main__":
    main()
