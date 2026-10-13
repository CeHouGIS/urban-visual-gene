"""Encode all cohort images with a trained W4096/k32 BatchTopK model.

Writes an image-by-feature array. The image score is the mean of the twenty
strongest of 196 patch activations, matching the old paper's pooling rule.
Unlike the old MAE representation, sparse SAE activations may be zero.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from sae_experiments.models.base_sae import BatchTopKSAE
from .train import DEFAULT_DATA, DEFAULT_OUTPUT, DEFAULT_TOKENS, audit, digest, write_json


def infer(args):
    ckpt = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    if (ckpt.get("D"), ckpt.get("K"), ckpt.get("k"), ckpt.get("model_type")) != (768, 4096, 32, "batch_topk"):
        raise ValueError("checkpoint is not a W4096/k32 BatchTopK model")
    cities, report = audit(args.data_root, args.token_root, args.seed, args.val_panos, args.manifest_root)
    if report["status"] != "verified" or ckpt["identity"]["data"] != report["dataset_identity"]:
        raise ValueError("checkpoint and thirty-city data identities do not match")
    model = BatchTopKSAE(768, 4096, 32).to(args.device).eval()
    model.load_state_dict(ckpt["state"])
    args.output.mkdir(parents=True, exist_ok=True)
    path = args.output / "image_activations.f16.npy"
    count_path = args.output / "image_cluster64_patch_winners.u16.npy"
    cluster_path = args.output / "feature_to_cluster64.npy"
    if not cluster_path.exists():
        raise FileNotFoundError("run python -m sae_experiments.legacy_mae_30city.cluster first")
    cluster_report = json.loads((args.output / "cluster_report.json").read_text())
    checkpoint_hash = digest(args.checkpoint)
    if cluster_report["checkpoint_sha256"] != checkpoint_hash:
        raise ValueError("cluster labels were derived from a different checkpoint")
    cluster_labels = np.load(cluster_path)
    if cluster_labels.shape != (4096,) or set(np.unique(cluster_labels)) != set(range(64)):
        raise ValueError("invalid 64-cluster feature labels")
    progress_path = args.output / "inference_progress.json"
    count = sum(int(city["images"]) for city in report["cities"])
    if path.exists():
        if not progress_path.exists():
            raise ValueError("activation array exists without progress file")
        scores = np.load(path, mmap_mode="r+")
        if not count_path.exists():
            raise ValueError("activation cache exists but patch-winner cache is absent")
        cluster_counts = np.load(count_path, mmap_mode="r+")
        progress = json.loads(progress_path.read_text())
        if (scores.shape != (count, 4096) or cluster_counts.shape != (count, 64)
                or progress["dataset_identity"] != report["dataset_identity"]
                or progress.get("checkpoint_hash") != checkpoint_hash):
            raise ValueError("activation cache shape/identity mismatch")
        done = int(progress["images_done"])
    else:
        scores = np.lib.format.open_memmap(path, mode="w+", dtype=np.float16, shape=(count, 4096))
        cluster_counts = np.lib.format.open_memmap(count_path, mode="w+", dtype=np.uint16, shape=(count, 64))
        done = 0
        write_json(progress_path, {"images_done": 0, "dataset_identity": report["dataset_identity"],
                                   "checkpoint_hash": checkpoint_hash})
    cursor = 0
    selected_rows = []
    with torch.inference_mode():
        for name, array, _, _ in cities:
            manifest = (args.manifest_root or args.data_root / "filtered_manifests") / (name.replace("/", "__") + ".parquet")
            frame = pd.read_parquet(manifest, columns=["source_image_index", "panoid"])
            if not report["images_are_original_feature_mae_images"]:
                quality_path = args.manifest_root.parent / "quality" / (name.replace("/", "__") + ".parquet")
                quality = pd.read_parquet(quality_path, columns=["panoid", "is_bad"])
                good = set(quality.loc[~quality.is_bad, "panoid"].astype(str))
                frame = frame[frame.panoid.astype(str).isin(good)]
            ids = frame.source_image_index.to_numpy(np.int64)
            selected_rows.append(pd.DataFrame({"city": name, "source_image_index": ids}))
            for start in range(0, len(ids), args.batch_images):
                global_start = cursor + start
                if global_start + min(args.batch_images, len(ids) - start) <= done:
                    continue
                if global_start < done:
                    raise ValueError("progress ended mid-batch; refusing to skip uncomputed images")
                block = np.asarray(array[ids[start:start + args.batch_images]], dtype=np.float32)
                x = torch.from_numpy(block.reshape(-1, 768)).to(args.device)
                with torch.autocast("cuda", dtype=torch.bfloat16, enabled=args.device.startswith("cuda")):
                    acts = model.encode(x)
                pooled = acts.float().reshape(len(block), 196, 4096).topk(20, dim=1).values.mean(1)
                scores[global_start:global_start + len(block)] = pooled.cpu().numpy().astype(np.float16)
                patch_acts = acts.float().reshape(len(block), 196, 4096)
                winners = patch_acts.argmax(-1).cpu().numpy()
                assigned = patch_acts.amax(-1).cpu().numpy() > 0
                winners = cluster_labels[winners]
                for local, labels in enumerate(winners):
                    cluster_counts[global_start + local] = np.bincount(labels[assigned[local]], minlength=64)
                if ((start // args.batch_images + 1) % args.flush_batches == 0 or start + len(block) == len(ids)):
                    scores.flush()
                    cluster_counts.flush()
                    write_json(progress_path, {"images_done": global_start + len(block),
                                               "dataset_identity": report["dataset_identity"],
                                               "checkpoint_hash": checkpoint_hash,
                                               "city": name})
            cursor += len(ids)
            print("encoded", name, len(ids), "images; total", cursor, flush=True)
    pd.concat(selected_rows, ignore_index=True).to_parquet(args.output / "inference_image_index.parquet", index=False)
    write_json(args.output / "inference_report.json", {"status": "complete", "images": count,
                "features": 4096, "pooling": "mean of strongest 20 of 196 patch activations",
                "checkpoint": str(args.checkpoint.resolve()), "data_identity": report["dataset_identity"],
                "checkpoint_sha256": checkpoint_hash,
                "activations": str(path.resolve()), "patch_winner_counts": str(count_path.resolve())})


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--data-root", type=Path, default=DEFAULT_DATA)
    p.add_argument("--token-root", type=Path, default=DEFAULT_TOKENS)
    p.add_argument("--manifest-root", type=Path)
    p.add_argument("--checkpoint", type=Path, default=DEFAULT_OUTPUT / "model_best.pt")
    p.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    p.add_argument("--device", default="cuda:2")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--val-panos", type=int, default=128)
    p.add_argument("--batch-images", type=int, default=4)
    p.add_argument("--flush-batches", type=int, default=100)
    args = p.parse_args()
    if args.batch_images < 1 or args.flush_batches < 1:
        p.error("batch-images and flush-batches must be positive")
    infer(args)


if __name__ == "__main__":
    main()
