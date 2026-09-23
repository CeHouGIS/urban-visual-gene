"""Train a city-balanced W=512, TopK=32 sparse autoencoder."""
from __future__ import annotations

import scripts._env  # noqa: F401

import argparse
import csv
import json
import math
import os
import random
import time
from pathlib import Path

import numpy as np
import torch
from torch import nn

from scripts.multicity.config import (
    CITIES,
    FEATURE_DIM,
    OUTPUT_ROOT,
    SEED,
    TOPK,
    WIDTH,
    feature_path,
)


class TopKSAE(nn.Module):
    """Non-negative Top-K sparse autoencoder with unit-norm dictionary atoms."""

    def __init__(self, input_dim: int = FEATURE_DIM, width: int = WIDTH, topk: int = TOPK):
        super().__init__()
        if not 0 < topk <= width:
            raise ValueError(f"topk must be in [1, width], got {topk}/{width}")
        self.input_dim = input_dim
        self.width = width
        self.topk = topk
        self.encoder = nn.Linear(input_dim, width)
        self.decoder = nn.Linear(width, input_dim)
        nn.init.kaiming_uniform_(self.encoder.weight, a=math.sqrt(5))
        nn.init.normal_(self.decoder.weight, std=1 / math.sqrt(input_dim))
        self.normalize_dictionary_()

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        # Center on the learned reconstruction bias.  DINO patch embeddings
        # are anisotropic; without centering, common directions monopolize the
        # Top-K competition and many dictionary atoms die.
        dense = torch.relu(self.encoder(x - self.decoder.bias))
        values, indices = torch.topk(dense, self.topk, dim=-1, sorted=False)
        sparse = torch.zeros_like(dense)
        return sparse.scatter(-1, indices, values)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        codes = self.encode(x)
        return self.decoder(codes), codes

    @torch.no_grad()
    def normalize_dictionary_(self) -> None:
        # nn.Linear stores its W dictionary atoms as columns of [D, W].
        self.decoder.weight.div_(self.decoder.weight.norm(dim=0, keepdim=True).clamp_min(1e-8))


class BalancedFeatureStore:
    """Memory-map equal-size city matrices and emit city-balanced batches."""

    def __init__(self, city_keys: list[str], output_root: Path, val_per_city: int):
        self.city_keys = city_keys
        self.arrays = [np.load(feature_path(c, output_root), mmap_mode="r") for c in city_keys]
        shapes = {tuple(a.shape) for a in self.arrays}
        if len(shapes) != 1:
            raise ValueError(f"feature shapes differ across cities: {shapes}")
        self.n_per_city, self.dim = self.arrays[0].shape
        if not 0 < val_per_city < self.n_per_city:
            raise ValueError("val_per_city must leave at least one training row")
        self.val_per_city = val_per_city
        self.train_per_city = self.n_per_city - val_per_city

    def train_batches(self, per_city: int, rng: np.random.Generator):
        permutations = [rng.permutation(self.train_per_city) for _ in self.arrays]
        for start in range(0, self.train_per_city, per_city):
            blocks = [
                np.asarray(a[p[start:start + per_city]], dtype=np.float32)
                for a, p in zip(self.arrays, permutations)
            ]
            yield np.concatenate(blocks, axis=0)

    def validation_batches(self, per_city: int):
        start0 = self.train_per_city
        for offset in range(0, self.val_per_city, per_city):
            blocks = [
                np.asarray(a[start0 + offset:start0 + offset + per_city], dtype=np.float32)
                for a in self.arrays
            ]
            yield np.concatenate(blocks, axis=0)


def reconstruction_loss(x: torch.Tensor, reconstruction: torch.Tensor) -> torch.Tensor:
    """Mean per-sample squared L2 reconstruction error."""
    return (reconstruction - x).square().sum(dim=1).mean()


@torch.no_grad()
def evaluate(model: TopKSAE, batches, device: str = "cuda") -> dict:
    model.eval()
    loss_sum = 0.0
    samples = 0
    usage = torch.zeros(model.width, dtype=torch.int64, device=device)
    active_sum = 0
    for array in batches:
        x = torch.from_numpy(array).to(device, non_blocking=True)
        with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
            reconstruction, codes = model(x)
            loss = reconstruction_loss(x, reconstruction)
        n = len(x)
        loss_sum += float(loss) * n
        samples += n
        active = codes.ne(0)
        active_sum += int(active.sum())
        usage += active.sum(dim=0)
    used = int(usage.gt(0).sum())
    probabilities = usage.float() / usage.sum().clamp_min(1)
    entropy = float(-(probabilities * probabilities.clamp_min(1e-12).log()).sum())
    return {
        "loss": loss_sum / samples,
        "mean_active": active_sum / samples,
        "used_features": used,
        "dead_features": model.width - used,
        "usage_entropy": entropy,
    }


def _atomic_torch_save(value: dict, path: Path) -> None:
    temp = path.with_suffix(path.suffix + ".tmp")
    torch.save(value, temp)
    os.replace(temp, path)


def train(args: argparse.Namespace) -> dict:
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.set_float32_matmul_precision("high")

    store = BalancedFeatureStore(args.cities, args.output_root, args.val_per_city)
    if store.dim != args.input_dim:
        raise ValueError(f"feature dim {store.dim} != requested input dim {args.input_dim}")
    train_out = args.output_root / "topk_sae"
    train_out.mkdir(parents=True, exist_ok=True)
    last_path = train_out / "checkpoint_last.pt"
    best_path = train_out / "model_best.pt"
    model = TopKSAE(args.input_dim, args.width, args.topk).to("cuda")
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    start_epoch, best_loss, stale = 0, float("inf"), 0
    history: list[dict] = []
    if args.resume and last_path.exists():
        checkpoint = torch.load(last_path, map_location="cuda", weights_only=False)
        model.load_state_dict(checkpoint["model"])
        optimizer.load_state_dict(checkpoint["optimizer"])
        start_epoch = int(checkpoint["epoch"]) + 1
        best_loss = float(checkpoint["best_loss"])
        stale = int(checkpoint["stale"])
        history = checkpoint.get("history", [])
        print(f"resuming from epoch {start_epoch}", flush=True)

    rng = np.random.default_rng(args.seed + start_epoch)
    started = time.time()
    torch.cuda.reset_peak_memory_stats()
    for epoch in range(start_epoch, args.epochs):
        model.train()
        train_loss_sum = 0.0
        train_samples = 0
        for array in store.train_batches(args.per_city_batch, rng):
            x = torch.from_numpy(array).to("cuda", non_blocking=True)
            optimizer.zero_grad(set_to_none=True)
            with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                reconstruction, _ = model(x)
                loss = reconstruction_loss(x, reconstruction)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
            optimizer.step()
            model.normalize_dictionary_()
            train_loss_sum += float(loss.detach()) * len(x)
            train_samples += len(x)

        validation = evaluate(
            model, store.validation_batches(args.per_city_batch), device="cuda"
        )
        row = {
            "epoch": epoch + 1,
            "train_loss": train_loss_sum / train_samples,
            "val_loss": validation["loss"],
            "mean_active": validation["mean_active"],
            "used_features": validation["used_features"],
            "dead_features": validation["dead_features"],
            "usage_entropy": validation["usage_entropy"],
            "elapsed_seconds": time.time() - started,
        }
        history.append(row)
        improved = validation["loss"] < best_loss - args.min_delta
        if improved:
            best_loss = validation["loss"]
            stale = 0
            _atomic_torch_save(
                {"model": model.state_dict(), "epoch": epoch, "metrics": row, "args": vars(args)},
                best_path,
            )
        else:
            stale += 1
        _atomic_torch_save(
            {
                "model": model.state_dict(), "optimizer": optimizer.state_dict(),
                "epoch": epoch, "best_loss": best_loss, "stale": stale,
                "history": history, "args": vars(args),
            },
            last_path,
        )
        print(
            f"epoch {epoch+1:03d}: train={row['train_loss']:.6f} "
            f"val={row['val_loss']:.6f} active={row['mean_active']:.2f} "
            f"dead={row['dead_features']} stale={stale}/{args.patience}",
            flush=True,
        )
        if stale >= args.patience:
            print("early stopping", flush=True)
            break

    best = torch.load(best_path, map_location="cuda", weights_only=False)
    model.load_state_dict(best["model"])
    model.normalize_dictionary_()
    basis = model.decoder.weight.detach().T.float().cpu().numpy()
    np.save(train_out / "global_pano_visual_basis.npy", basis)
    with (train_out / "history.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(history[0]))
        writer.writeheader()
        writer.writerows(history)
    report = {
        "model": "DINOv3 ViT-B/16 CLS x four headings",
        "n_cities": len(args.cities),
        "cities": args.cities,
        "panos_per_city": store.n_per_city,
        "training_panos": store.train_per_city * len(args.cities),
        "validation_panos": store.val_per_city * len(args.cities),
        "input_dim": args.input_dim,
        "width": args.width,
        "topk": args.topk,
        "best_epoch": int(best["epoch"]) + 1,
        "best_metrics": best["metrics"],
        "basis_shape": list(basis.shape),
        "basis_norm_min": float(np.linalg.norm(basis, axis=1).min()),
        "basis_norm_max": float(np.linalg.norm(basis, axis=1).max()),
        "peak_gpu_gib": torch.cuda.max_memory_allocated() / 2**30,
        "elapsed_seconds": time.time() - started,
    }
    (train_out / "training_report.json").write_text(json.dumps(report, indent=2) + "\n")
    return report


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--cities", nargs="*", default=list(CITIES))
    ap.add_argument("--output-root", type=Path, default=OUTPUT_ROOT)
    ap.add_argument("--input-dim", type=int, default=FEATURE_DIM)
    ap.add_argument("--width", type=int, default=WIDTH)
    ap.add_argument("--topk", type=int, default=TOPK)
    ap.add_argument("--epochs", type=int, default=30)
    ap.add_argument("--patience", type=int, default=6)
    ap.add_argument("--min-delta", type=float, default=1e-5)
    ap.add_argument("--val-per-city", type=int, default=1000)
    ap.add_argument("--per-city-batch", type=int, default=32)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--weight-decay", type=float, default=1e-5)
    ap.add_argument("--grad-clip", type=float, default=1.0)
    ap.add_argument("--seed", type=int, default=SEED)
    ap.add_argument("--no-resume", dest="resume", action="store_false")
    ap.set_defaults(resume=True)
    args = ap.parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for Top-K SAE training")
    report = train(args)
    print(json.dumps(report, indent=2), flush=True)


if __name__ == "__main__":
    main()
