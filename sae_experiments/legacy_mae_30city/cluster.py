"""Cluster W4096 decoder directions into the paper's 32/64 nested scales.

The old MAE used an E+D+P hierarchy; here SAE dictionary directions provide
the model-specific feature geometry. IDs are deterministic for a checkpoint.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch
from scipy.cluster.hierarchy import cut_tree, linkage
from scipy.spatial.distance import pdist

from .train import DEFAULT_OUTPUT, digest, write_json


def cluster(checkpoint: Path, output: Path) -> dict:
    ckpt = torch.load(checkpoint, map_location="cpu", weights_only=False)
    if (ckpt.get("D"), ckpt.get("K"), ckpt.get("k"), ckpt.get("model_type")) != (768, 4096, 32, "batch_topk"):
        raise ValueError("expected old-data W4096/k32 BatchTopK checkpoint")
    directions = ckpt["state"]["decoder.weight"].float().numpy().T
    directions /= np.linalg.norm(directions, axis=1, keepdims=True).clip(min=1e-8)
    distances = pdist(directions, metric="cosine")
    tree = linkage(distances, method="average")
    labels = cut_tree(tree, n_clusters=[32, 64]).astype(np.int16)
    output.mkdir(parents=True, exist_ok=True)
    np.save(output / "feature_to_cluster32.npy", labels[:, 0])
    np.save(output / "feature_to_cluster64.npy", labels[:, 1])
    np.save(output / "hierarchy_linkage.npy", tree)
    report = {"status": "complete", "features": 4096, "cluster_counts": [32, 64],
              "method": "average linkage of cosine distances between SAE decoder directions",
              "checkpoint": str(checkpoint.resolve()), "checkpoint_sha256": digest(checkpoint),
              "dataset_identity": ckpt["identity"]["data"]}
    write_json(output / "cluster_report.json", report)
    return report


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--checkpoint", type=Path, default=DEFAULT_OUTPUT / "model_best.pt")
    p.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = p.parse_args()
    print(json.dumps(cluster(args.checkpoint, args.output), indent=2))


if __name__ == "__main__":
    main()
