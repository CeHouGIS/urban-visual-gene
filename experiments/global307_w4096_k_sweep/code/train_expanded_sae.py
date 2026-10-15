"""Train a W4096 BatchTopK SAE on the formal 30M-patch cache.

Use CUDA_VISIBLE_DEVICES to choose the physical GPU. The formal trainer keeps
an optimizer checkpoint after every epoch and resumes from it automatically.
"""
from __future__ import annotations

import argparse
import json
import random
from pathlib import Path

import numpy as np
import torch

from formal import gpu_run


DEFAULT_CACHE = Path("/tmp/uvg_sae_cache/batchtopk_30c1fc1025090490_30000000.f16.npy")
DEFAULT_OUT = Path("/host/root/mnt/nas/huangyj/GoogleSV/urban_visual_gene/formal_batchtopk_307city_w4096_k8")
DEFAULT_QC = Path("/host/root/mnt/nas/huangyj/GoogleSV/urban_visual_gene/formal_batchtopk_307city_ablation_k2048/feature_qc_training_manifest.json")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sample-path", type=Path, default=DEFAULT_CACHE)
    parser.add_argument("--outdir", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--qc-manifest", type=Path, default=DEFAULT_QC)
    parser.add_argument("--width", type=int, default=4096)
    parser.add_argument("--k", type=int, default=8)
    parser.add_argument("--epochs", type=int, default=60)
    parser.add_argument("--batch-size", type=int, default=8192)
    parser.add_argument("--train-block-rows", type=int, default=262144)
    args = parser.parse_args()
    if not torch.cuda.is_available():
        parser.error("CUDA is required for the formal run")
    cache = np.load(args.sample_path, mmap_mode="r")
    if cache.shape != (30_000_000, 1024) or cache.dtype != np.float16:
        parser.error(f"expected the formal FP16 30M x 1024 cache, got {cache.shape} {cache.dtype}")
    if args.width <= cache.shape[1]:
        parser.error("dictionary width must exceed the 1024-D DINOv3 input")
    qc = json.loads(args.qc_manifest.read_text())
    if len(qc["cities"]) != 304 or qc["rows_per_pano"] != 3136:
        parser.error("QC manifest does not describe the formal 304-city cache")
    marker = json.loads(args.sample_path.with_suffix(".json").read_text())
    if marker["rows"] != len(cache) or marker["cities"] != len(qc["cities"]):
        parser.error("cache marker and QC manifest disagree")

    random.seed(0)
    np.random.seed(0)
    torch.manual_seed(0)
    torch.cuda.manual_seed_all(0)
    torch.set_num_threads(8)
    args.outdir.mkdir(parents=True, exist_ok=True)
    (args.outdir / "run_manifest.json").write_text(json.dumps({
        "sample_path": str(args.sample_path), "qc_manifest": str(args.qc_manifest),
        "training_cities": len(qc["cities"]),
        "verified_image_mapping_cities": sum(bool(city.get("verified")) for city in qc["cities"].values()),
        "sample_rows": len(cache), "input_dim": cache.shape[1],
        "dictionary_width": args.width, "batch_topk_k": args.k,
        "epochs": args.epochs, "batch_size": args.batch_size,
        "context_weight": 0.25, "image_grid": "four 28x28 views per panorama",
        "reference_training_code": "formal.gpu_run.train_sae",
    }, indent=2) + "\n")
    gpu_run.OUT = args.outdir
    formal_args = argparse.Namespace(
        res=448, topk=args.k, epochs=args.epochs, sae_batch_size=args.batch_size,
        train_block_rows=args.train_block_rows, max_train_patches=len(cache),
        context_weight=0.25, sae_qc_manifest=str(args.qc_manifest),
    )
    gpu_run.train_sae(cache, args.width, formal_args)


if __name__ == "__main__":
    main()
