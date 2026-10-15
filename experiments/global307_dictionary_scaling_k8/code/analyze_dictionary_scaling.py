"""Measure semantic feature stability while scaling a BatchTopK dictionary.

The three models are encoded on exactly the same ordered evaluation rows and
with the same batch size.  This matters because BatchTopK selects a budget
jointly over each batch.  The script keeps only sparse summaries (per-feature
top activating rows and moments), so it never materializes an N x K activation
matrix on disk.

Example::

  python -m formal.analysis.analyze_dictionary_scaling \
      --sample-path /tmp/uvg_sae_cache/batchtopk_30c1fc1025090490_30000000.f16.npy \
      --checkpoint 512=/host/root/mnt/nas/.../sae_448_k512.pt \
      --checkpoint 1024=/host/root/mnt/nas/.../sae_448_k1024.pt \
      --checkpoint 2048=/host/root/mnt/nas/.../sae_448_k2048.pt \
      --eval-patches 100000 --outdir /host/root/mnt/nas/.../scaling_analysis
"""
from __future__ import annotations

import argparse
import csv
import json
import math
import time
from pathlib import Path

import numpy as np
import torch

from sae_experiments.models.base_sae import build_sae


DEFAULT_CACHE = "/tmp/uvg_sae_cache/batchtopk_30c1fc1025090490_30000000.f16.npy"
DEFAULT_CHECKPOINTS = {
    512: "/host/root/mnt/nas/huangyj/GoogleSV/urban_visual_gene/formal_batchtopk_307city_ablation_k512/sae_448_k512.pt",
    1024: "/host/root/mnt/nas/huangyj/GoogleSV/urban_visual_gene/formal_batchtopk_307city_qc_filtered/sae_448_k1024.pt",
    2048: "/host/root/mnt/nas/huangyj/GoogleSV/urban_visual_gene/formal_batchtopk_307city_ablation_k2048/sae_448_k2048.pt",
}


def log(*args: object) -> None:
    print(f"[{time.strftime('%H:%M:%S')}]", *args, flush=True)


def load_model(path: Path, device: torch.device):
    meta = torch.load(path, map_location="cpu", weights_only=False)
    d = int(meta["D"])
    width = int(meta["K"])
    topk = int(meta.get("k", meta.get("topk", 8)))
    model = build_sae(
        input_dim=d,
        config={
            "type": meta.get("model_type", "batch_topk"),
            "latent_dim": width,
            "k": topk,
            "decoder_unit_norm": True,
        },
    ).to(device)
    model.load_state_dict(meta["state"])
    model.eval()
    # Decoder columns are the semantic directions.  Normalize defensively so
    # checkpoints made before the decoder-normalization hook remain comparable.
    decoder = model.decoder.weight.detach().to(torch.float32)
    decoder = decoder / decoder.norm(dim=0, keepdim=True).clamp_min(1e-8)
    log(f"loaded W={width} top-k={topk} loss={meta.get('final_train_loss')} from {path}")
    return model, decoder.cpu().numpy().T.astype(np.float32), meta


def choose_eval_indices(n_rows: int, count: int, seed: int) -> np.ndarray:
    count = min(int(count), int(n_rows))
    if count <= 0:
        raise ValueError("--eval-patches must be positive")
    rng = np.random.default_rng(seed)
    # Sorted indices make the mmap access nearly sequential.  The resulting
    # order is still fixed and therefore identical for every model.
    return np.sort(rng.choice(n_rows, size=count, replace=False).astype(np.int64))


def update_top_rows(
    current_values: np.ndarray,
    current_rows: np.ndarray,
    acts: np.ndarray,
    rows: np.ndarray,
) -> None:
    """Merge a batch into per-feature top-N reservoirs in-place."""
    n_keep = current_values.shape[1]
    local_n = min(n_keep, acts.shape[0])
    # topk along rows, for every feature.  Zero activations are retained only
    # when a feature has fewer than n_keep positive samples overall.
    local_order = np.argpartition(acts, -local_n, axis=0)[-local_n:]
    cols = np.arange(acts.shape[1])[None, :]
    local_values = acts[local_order, cols].T
    local_rows = rows[local_order].T
    merged_values = np.concatenate((current_values, local_values), axis=1)
    merged_rows = np.concatenate((current_rows, local_rows), axis=1)
    keep = np.argpartition(merged_values, -n_keep, axis=1)[:, -n_keep:]
    current_values[:] = np.take_along_axis(merged_values, keep, axis=1)
    current_rows[:] = np.take_along_axis(merged_rows, keep, axis=1)


@torch.no_grad()
def collect_stats(
    features: np.ndarray,
    eval_rows: np.ndarray,
    model,
    decoder: np.ndarray,
    batch_size: int,
    top_rows: int,
    device: torch.device,
) -> dict[str, np.ndarray]:
    width = decoder.shape[0]
    top_values = np.zeros((width, top_rows), dtype=np.float32)
    top_indices = np.full((width, top_rows), -1, dtype=np.int64)
    frequency = np.zeros(width, dtype=np.int64)
    sum_activation = np.zeros(width, dtype=np.float64)
    sum_sq_activation = np.zeros(width, dtype=np.float64)
    max_activation = np.zeros(width, dtype=np.float32)
    n = len(eval_rows)
    for start in range(0, n, batch_size):
        stop = min(start + batch_size, n)
        sample_rows = eval_rows[start:stop]
        x = torch.from_numpy(np.asarray(features[sample_rows], dtype=np.float32)).to(
            device, non_blocking=True
        )
        acts = model.encode(x).to(torch.float32).cpu().numpy()
        positive = acts > 0
        frequency += positive.sum(axis=0, dtype=np.int64)
        sum_activation += acts.sum(axis=0, dtype=np.float64)
        sum_sq_activation += np.square(acts, dtype=np.float32).sum(axis=0, dtype=np.float64)
        max_activation = np.maximum(max_activation, acts.max(axis=0))
        update_top_rows(top_values, top_indices, acts, sample_rows)
        if stop == n or (start // batch_size) % 25 == 0:
            log(f"  encoded {stop:,}/{n:,}")
    mean = sum_activation / max(n, 1)
    variance = np.maximum(sum_sq_activation / max(n, 1) - mean * mean, 0.0)
    return {
        "decoder": decoder,
        "frequency": frequency,
        "frequency_fraction": frequency.astype(np.float64) / max(n, 1),
        "mean_activation": mean,
        "std_activation": np.sqrt(variance),
        "max_activation": max_activation,
        "top_rows": top_indices,
        "top_values": top_values,
    }


def row_overlap(a: np.ndarray, b: np.ndarray) -> float:
    a = a[a >= 0]
    b = b[b >= 0]
    if not len(a) or not len(b):
        return 0.0
    # The reservoirs are short (default 32), so this is faster and clearer
    # than building a global sparse activation matrix.
    return float(np.intersect1d(a, b, assume_unique=False).size / min(len(a), len(b)))


def pair_metrics(left: dict[str, np.ndarray], right: dict[str, np.ndarray], candidate_top: int):
    cosine = left["decoder"] @ right["decoder"].T
    # Candidate pairs are restricted by decoder similarity.  This reduces the
    # expensive top-row overlap work while retaining all plausible matches.
    candidate_top = min(candidate_top, cosine.shape[1])
    pairs = []
    overlap = np.zeros_like(cosine, dtype=np.float32)
    for i in range(cosine.shape[0]):
        candidates = np.argpartition(cosine[i], -candidate_top)[-candidate_top:]
        for j in candidates:
            ov = row_overlap(left["top_rows"][i], right["top_rows"][j])
            overlap[i, j] = ov
            # Decoder direction is the primary identity signal.  Activation
            # examples provide the data-dependent semantic check.
            score = 0.7 * float(cosine[i, j]) + 0.3 * ov
            pairs.append((score, float(cosine[i, j]), ov, i, int(j)))
    return cosine, overlap, pairs


def classify_pair(
    left: dict[str, np.ndarray],
    right: dict[str, np.ndarray],
    cosine: np.ndarray,
    overlap: np.ndarray,
    stable_cos: float,
    stable_overlap: float,
    split_cos: float,
    split_overlap: float,
) -> tuple[list[dict], dict]:
    n_left, n_right = cosine.shape
    # A pair can only be a semantic candidate when both its vector and its
    # high-activation examples agree.  Mutual nearest neighbors define the
    # conservative one-to-one stable set.
    score = 0.7 * cosine + 0.3 * overlap
    left_best = score.argmax(axis=1)
    right_best = score.argmax(axis=0)
    stable = []
    for i, j in enumerate(left_best):
        if right_best[j] != i:
            continue
        if cosine[i, j] >= stable_cos and overlap[i, j] >= stable_overlap:
            stable.append((i, int(j)))

    # Split/merge candidates are intentionally reported separately from the
    # strict stable set.  Their combined top examples cover a parent feature.
    split = []
    for i in range(n_left):
        js = np.flatnonzero((cosine[i] >= split_cos) & (overlap[i] >= split_overlap))
        if len(js) >= 2:
            covered = np.unique(np.concatenate([
                right["top_rows"][j][right["top_rows"][j] >= 0] for j in js
            ]))
            parent = right["top_rows"][i] if i < n_right else np.array([], dtype=np.int64)
            # Coverage is evaluated against the left feature's examples.
            parent_rows = left["top_rows"][i]
            coverage = len(np.intersect1d(parent_rows[parent_rows >= 0], covered)) / max(1, len(parent_rows[parent_rows >= 0]))
            split.append({"source": i, "targets": [int(j) for j in js], "coverage": float(coverage)})

    stable_left = {i for i, _ in stable}
    stable_right = {j for _, j in stable}
    summary = {
        "stable_one_to_one": len(stable),
        "stable_left_fraction": len(stable_left) / max(1, n_left),
        "stable_right_fraction": len(stable_right) / max(1, n_right),
        "split_sources": len(split),
        "split_targets": int(sum(len(row["targets"]) for row in split)),
        "left_unmatched": n_left - len(stable_left),
        "right_unmatched": n_right - len(stable_right),
    }
    matches = [
        {
            "left": int(i),
            "right": int(j),
            "decoder_cosine": float(cosine[i, j]),
            "top_row_overlap": float(overlap[i, j]),
            "relation": "stable",
        }
        for i, j in stable
    ]
    for row in split:
        for j in row["targets"]:
            matches.append({
                "left": int(row["source"]), "right": int(j),
                "decoder_cosine": float(cosine[row["source"], j]),
                "top_row_overlap": float(overlap[row["source"], j]),
                "relation": "split_candidate", "parent_coverage": row["coverage"],
            })
    return matches, {"summary": summary, "split": split}


def write_feature_csv(path: Path, stats: dict[str, np.ndarray]) -> None:
    with path.open("w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["feature", "frequency", "frequency_fraction", "mean_activation", "std_activation", "max_activation", "top_eval_rows"])
        for i in range(len(stats["frequency"])):
            writer.writerow([
                i, int(stats["frequency"][i]), float(stats["frequency_fraction"][i]),
                float(stats["mean_activation"][i]), float(stats["std_activation"][i]),
                float(stats["max_activation"][i]), ";".join(map(str, stats["top_rows"][i].tolist())),
            ])


def parse_checkpoint(value: str) -> tuple[int, Path]:
    if "=" not in value:
        raise argparse.ArgumentTypeError("checkpoint must be WIDTH=PATH")
    width, path = value.split("=", 1)
    try:
        return int(width), Path(path)
    except ValueError as exc:
        raise argparse.ArgumentTypeError("checkpoint width must be an integer") from exc


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--sample-path", default=DEFAULT_CACHE)
    p.add_argument("--checkpoint", action="append", type=parse_checkpoint, dest="checkpoints", help="WIDTH=CHECKPOINT; repeat for each dictionary")
    p.add_argument("--outdir", required=True)
    p.add_argument("--eval-patches", type=int, default=100_000)
    p.add_argument("--seed", type=int, default=20261008)
    p.add_argument("--batch-size", type=int, default=4096)
    p.add_argument("--top-rows", type=int, default=32)
    p.add_argument("--candidate-top", type=int, default=32)
    p.add_argument("--stable-cosine", type=float, default=0.70)
    p.add_argument("--stable-overlap", type=float, default=0.10)
    p.add_argument("--split-cosine", type=float, default=0.45)
    p.add_argument("--split-overlap", type=float, default=0.03)
    p.add_argument("--device", default="cuda:0")
    args = p.parse_args()

    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    features = np.load(args.sample_path, mmap_mode="r")
    if features.ndim != 2 or features.shape[1] != 1024:
        raise ValueError(f"expected [N,1024] feature cache, got {features.shape}")
    eval_rows = choose_eval_indices(features.shape[0], args.eval_patches, args.seed)
    np.save(outdir / "eval_patch_indices.npy", eval_rows)
    (outdir / "evaluation_manifest.json").write_text(json.dumps({
        "sample_path": str(args.sample_path), "n_source_rows": int(features.shape[0]),
        "n_eval_rows": int(len(eval_rows)), "seed": args.seed,
        "batch_size": args.batch_size, "top_rows": args.top_rows,
    }, indent=2))

    checkpoints = dict(args.checkpoints or [])
    if not checkpoints:
        checkpoints = DEFAULT_CHECKPOINTS
    if len(checkpoints) < 2:
        raise ValueError("provide at least two --checkpoint WIDTH=PATH entries")
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    all_stats: dict[int, dict[str, np.ndarray]] = {}
    model_meta = {}
    for width in sorted(checkpoints):
        model, decoder, meta = load_model(Path(checkpoints[width]), device)
        stats = collect_stats(features, eval_rows, model, decoder, args.batch_size, args.top_rows, device)
        all_stats[width] = stats
        model_meta[str(width)] = {"checkpoint": str(checkpoints[width]), "D": int(meta["D"]), "K": int(meta["K"]), "topk": int(meta.get("k", meta.get("topk", 8))), "final_train_loss": meta.get("final_train_loss")}
        np.savez_compressed(outdir / f"feature_stats_w{width}.npz", **stats)
        write_feature_csv(outdir / f"feature_stats_w{width}.csv", stats)
        del model
        if device.type == "cuda":
            torch.cuda.empty_cache()

    pair_summary = {}
    pair_files = []
    widths = sorted(all_stats)
    for left_width, right_width in zip(widths, widths[1:]):
        left, right = all_stats[left_width], all_stats[right_width]
        cosine, overlap, _ = pair_metrics(left, right, args.candidate_top)
        matches, classification = classify_pair(left, right, cosine, overlap, args.stable_cosine, args.stable_overlap, args.split_cosine, args.split_overlap)
        pair_name = f"w{left_width}_to_w{right_width}"
        with (outdir / f"matches_{pair_name}.csv").open("w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=["left", "right", "decoder_cosine", "top_row_overlap", "relation", "parent_coverage"])
            writer.writeheader()
            for row in matches:
                writer.writerow({k: row.get(k, "") for k in writer.fieldnames})
        np.savez_compressed(outdir / f"pair_metrics_{pair_name}.npz", decoder_cosine=cosine, top_row_overlap=overlap)
        pair_summary[pair_name] = classification
        pair_files.append(f"matches_{pair_name}.csv")
        log(pair_name, classification["summary"])

    summary = {
        "evaluation": {"sample_path": str(args.sample_path), "n_source_rows": int(features.shape[0]), "n_eval_rows": int(len(eval_rows)), "seed": args.seed, "batch_size": args.batch_size, "top_rows": args.top_rows},
        "models": model_meta,
        "thresholds": {"stable_cosine": args.stable_cosine, "stable_overlap": args.stable_overlap, "split_cosine": args.split_cosine, "split_overlap": args.split_overlap},
        "pairs": pair_summary,
        "files": pair_files,
        "interpretation": {
            "stable": "mutual one-to-one match passing decoder and top-activation thresholds",
            "split_candidate": "one source feature has at least two plausible target features; inspect parent_coverage and examples",
            "left_unmatched": "features in the smaller/left dictionary without a strict stable match",
            "right_unmatched": "features in the larger/right dictionary without a strict stable match; these include new or reorganized features",
        },
    }
    (outdir / "scaling_summary.json").write_text(json.dumps(summary, indent=2))
    log(f"done -> {outdir}")


if __name__ == "__main__":
    main()
