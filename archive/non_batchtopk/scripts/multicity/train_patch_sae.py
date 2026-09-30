"""Train a city-balanced Top-K SAE on normalized DINOv3 patch tokens."""
from __future__ import annotations

import scripts._env  # noqa: F401

import argparse
import csv
import json
import os
import time
from pathlib import Path

import numpy as np
import torch

from scripts.multicity.config import CITIES
from scripts.multicity.patch_config import INPUT_DIM, OUTPUT_ROOT, SEED, TOPK, WIDTH, token_path
from scripts.multicity.train_topk_sae import TopKSAE, reconstruction_loss


class PatchStore:
    def __init__(
        self, cities: list[str], root: Path, val_images: int,
        patches_per_image: int, seed: int,
    ):
        self.cities = cities
        self.arrays = [
            self._sampled_pool(c, root, patches_per_image, seed)
            for c in cities
        ]
        shapes = {a.shape for a in self.arrays}
        if len(shapes) != 1:
            raise ValueError(f"token shapes differ: {shapes}")
        self.n_images, self.n_patches, self.dim = self.arrays[0].shape
        self.val_images = val_images
        self.train_images = self.n_images - val_images

    @staticmethod
    def _sampled_pool(
        city: str, root: Path, patches_per_image: int, seed: int
    ) -> np.memmap:
        pool = (
            root / "training_pool" / f"patches_{patches_per_image}" /
            f"{city.replace('/', '__')}.f16.npy"
        )
        source = np.load(token_path(city, root), mmap_mode="r")
        expected = (len(source), patches_per_image, source.shape[-1])
        if pool.exists():
            existing = np.load(pool, mmap_mode="r")
            if existing.shape == expected:
                return existing
        pool.parent.mkdir(parents=True, exist_ok=True)
        temp = pool.with_name(pool.name + ".tmp.npy")
        destination = np.lib.format.open_memmap(
            temp, mode="w+", dtype=np.float16, shape=expected
        )
        # Read contiguous image blocks from the large memmap, then sample in
        # RAM.  This turns millions of tiny random disk reads into sequential
        # reads and retains equal patch coverage for every image.
        city_seed = seed + sum((i + 1) * ord(ch) for i, ch in enumerate(city))
        rng = np.random.default_rng(city_seed)
        for start in range(0, len(source), 256):
            end = min(start + 256, len(source))
            block = np.asarray(source[start:end])
            picks = rng.integers(
                0, source.shape[1], size=(end - start, patches_per_image)
            )
            destination[start:end] = block[np.arange(end - start)[:, None], picks]
        destination.flush()
        del destination
        os.replace(temp, pool)
        print(f"built training pool: {city} -> {pool}", flush=True)
        return np.load(pool, mmap_mode="r")

    def train_batches(self, images_per_city: int, rng: np.random.Generator):
        orders = [rng.permutation(self.train_images) for _ in self.arrays]
        for start in range(0, self.train_images, images_per_city):
            blocks = []
            for array, order in zip(self.arrays, orders):
                image_ids = order[start:start + images_per_city]
                block = np.asarray(array[image_ids], dtype=np.float32)
                blocks.append(block.reshape(-1, self.dim))
            yield np.concatenate(blocks, axis=0)

    def validation_batches(self, image_batch: int = 8):
        for offset in range(0, self.val_images, image_batch):
            blocks = []
            for array in self.arrays:
                block = np.asarray(
                    array[
                        self.train_images + offset:
                        self.train_images + offset + image_batch
                    ],
                    dtype=np.float32,
                )
                blocks.append(block.reshape(-1, self.dim))
            yield np.concatenate(blocks, axis=0)

    def mean_token(self) -> np.ndarray:
        total = np.zeros(self.dim, dtype=np.float64)
        count = 0
        for array in self.arrays:
            for start in range(0, len(array), 256):
                block = np.asarray(array[start:start + 256], dtype=np.float32)
                total += block.sum(axis=(0, 1), dtype=np.float64)
                count += block.shape[0] * block.shape[1]
        return (total / count).astype(np.float32)


def _save(value: dict, path: Path) -> None:
    temp = path.with_suffix(path.suffix + ".tmp")
    torch.save(value, temp)
    os.replace(temp, path)


@torch.no_grad()
def revive_rare_features(
    model: TopKSAE,
    optimizer: torch.optim.Optimizer,
    usage: torch.Tensor,
    residual_examples: torch.Tensor,
    min_uses: int,
) -> int:
    """Re-seed unused atoms from high-error tokens so all dimensions learn."""
    rare = torch.where(usage < min_uses)[0]
    if len(rare) == 0:
        return 0
    if len(residual_examples) < len(rare):
        repeats = int(np.ceil(len(rare) / len(residual_examples)))
        residual_examples = residual_examples.repeat(repeats, 1)
    centered = residual_examples[:len(rare)] - model.decoder.bias
    atoms = torch.nn.functional.normalize(centered, dim=1)
    live = torch.where(usage >= min_uses)[0]
    scale = (
        model.encoder.weight[live].norm(dim=1).median()
        if len(live) else torch.tensor(0.25, device=atoms.device)
    )
    model.encoder.weight[rare] = atoms * scale
    model.encoder.bias[rare] = 0.0
    model.decoder.weight[:, rare] = atoms.T

    # Remove stale Adam moments only for re-seeded coordinates.
    for parameter, axis in (
        (model.encoder.weight, 0),
        (model.encoder.bias, 0),
        (model.decoder.weight, 1),
    ):
        state = optimizer.state.get(parameter, {})
        for key in ("exp_avg", "exp_avg_sq", "max_exp_avg_sq"):
            value = state.get(key)
            if value is not None:
                if axis == 0:
                    value[rare] = 0
                else:
                    value[:, rare] = 0
    model.normalize_dictionary_()
    return len(rare)


@torch.no_grad()
def validate(model: TopKSAE, batches) -> dict:
    model.eval()
    total_loss = total_n = total_active = 0
    usage = torch.zeros(model.width, dtype=torch.int64, device="cuda")
    for array in batches:
        x = torch.from_numpy(array).to("cuda", non_blocking=True)
        with torch.autocast("cuda", dtype=torch.bfloat16):
            reconstruction, codes = model(x)
            loss = reconstruction_loss(x, reconstruction)
        total_loss += float(loss) * len(x)
        total_n += len(x)
        active = codes.ne(0)
        total_active += int(active.sum())
        usage += active.sum(0)
    return {
        "loss": total_loss / total_n,
        "mean_active": total_active / total_n,
        "dead_features": int(usage.eq(0).sum()),
        "used_features": int(usage.gt(0).sum()),
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--cities", nargs="*", default=list(CITIES))
    ap.add_argument("--output-root", type=Path, default=OUTPUT_ROOT)
    ap.add_argument("--width", type=int, default=WIDTH)
    ap.add_argument("--topk", type=int, default=TOPK)
    ap.add_argument("--epochs", type=int, default=10)
    ap.add_argument("--patience", type=int, default=3)
    ap.add_argument("--val-images", type=int, default=1000)
    ap.add_argument("--images-per-city", type=int, default=16)
    ap.add_argument("--patches-per-image", type=int, default=32)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--min-feature-uses", type=int, default=100)
    ap.add_argument("--seed", type=int, default=SEED)
    args = ap.parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required")
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    torch.backends.cuda.matmul.allow_tf32 = True
    store = PatchStore(
        args.cities, args.output_root, args.val_images,
        args.patches_per_image, args.seed,
    )
    if store.dim != INPUT_DIM:
        raise ValueError(f"expected token dim {INPUT_DIM}, got {store.dim}")
    out = args.output_root / "patch_sae"
    out.mkdir(parents=True, exist_ok=True)
    last_path, best_path = out / "checkpoint_last.pt", out / "model_best.pt"
    model = TopKSAE(INPUT_DIM, args.width, args.topk).to("cuda")
    with torch.no_grad():
        model.decoder.bias.copy_(torch.from_numpy(store.mean_token()).to("cuda"))
        # Start encoder directions aligned with dictionary atoms.  The 0.5
        # scale keeps newly activated codes in the same range as live codes.
        model.encoder.weight.copy_(model.decoder.weight.T * 0.5)
        model.encoder.bias.zero_()
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-5)
    start_epoch, best_score, stale, history = 0, float("inf"), 0, []
    if last_path.exists():
        state = torch.load(last_path, map_location="cuda", weights_only=False)
        model.load_state_dict(state["model"])
        optimizer.load_state_dict(state["optimizer"])
        start_epoch = state["epoch"] + 1
        best_score = state.get("best_score", state.get("best_loss", float("inf")))
        stale, history = state["stale"], state["history"]
        print(f"resuming at epoch {start_epoch + 1}", flush=True)
    rng = np.random.default_rng(args.seed + start_epoch)
    started = time.time()
    torch.cuda.reset_peak_memory_stats()
    for epoch in range(start_epoch, args.epochs):
        model.train()
        loss_sum = n_sum = 0
        train_usage = torch.zeros(model.width, dtype=torch.int64, device="cuda")
        residual_values = torch.empty(0, device="cuda")
        residual_examples = torch.empty((0, INPUT_DIM), device="cuda")
        for array in store.train_batches(args.images_per_city, rng):
            x = torch.from_numpy(array).to("cuda", non_blocking=True)
            optimizer.zero_grad(set_to_none=True)
            with torch.autocast("cuda", dtype=torch.bfloat16):
                reconstruction, codes = model(x)
                loss = reconstruction_loss(x, reconstruction)
            with torch.no_grad():
                train_usage += codes.ne(0).sum(0)
                errors = (reconstruction - x).square().sum(1)
                local_n = min(model.width, len(errors))
                local_values, local_indices = torch.topk(errors, local_n)
                candidate_values = torch.cat((residual_values, local_values))
                candidate_examples = torch.cat((residual_examples, x[local_indices]))
                keep_n = min(model.width, len(candidate_values))
                residual_values, keep = torch.topk(candidate_values, keep_n)
                residual_examples = candidate_examples[keep]
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            model.normalize_dictionary_()
            loss_sum += float(loss.detach()) * len(x)
            n_sum += len(x)
        revived = revive_rare_features(
            model, optimizer, train_usage, residual_examples, args.min_feature_uses
        )
        metrics = validate(model, store.validation_batches())
        row = {
            "epoch": epoch + 1, "train_loss": loss_sum / n_sum,
            "val_loss": metrics["loss"], "mean_active": metrics["mean_active"],
            "dead_features": metrics["dead_features"],
            "revived_features": revived,
            "elapsed_seconds": time.time() - started,
        }
        history.append(row)
        # A tiny coverage penalty prevents selecting a low-loss checkpoint in
        # which hundreds of dimensions have no heatmap response.
        score = metrics["loss"] + 1e-4 * metrics["dead_features"]
        if score < best_score - 1e-5:
            best_score, stale = score, 0
            _save({"model": model.state_dict(), "epoch": epoch, "metrics": row,
                   "config": vars(args)}, best_path)
        else:
            stale += 1
        _save({"model": model.state_dict(), "optimizer": optimizer.state_dict(),
               "epoch": epoch, "best_score": best_score, "stale": stale,
               "history": history, "config": vars(args)}, last_path)
        print(
            f"epoch {epoch+1:02d}: train={row['train_loss']:.6f} "
            f"val={row['val_loss']:.6f} active={row['mean_active']:.2f} "
            f"dead={row['dead_features']} revived={revived}", flush=True,
        )
        if stale >= args.patience:
            break
    best = torch.load(best_path, map_location="cuda", weights_only=False)
    model.load_state_dict(best["model"])
    basis = model.decoder.weight.detach().T.float().cpu().numpy()
    np.save(out / "urban_visual_gene_basis.npy", basis)
    with (out / "history.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(history[0]))
        writer.writeheader(); writer.writerows(history)
    report = {
        "backbone": "DINOv3 ViT-B/16 patch tokens",
        "cities": args.cities, "images_per_city": store.n_images,
        "patch_grid": [14, 14], "input_dim": INPUT_DIM,
        "width": args.width, "topk": args.topk,
        "best_epoch": best["epoch"] + 1, "best_metrics": best["metrics"],
        "basis_shape": list(basis.shape),
        "peak_gpu_gib": torch.cuda.max_memory_allocated() / 2**30,
    }
    (out / "training_report.json").write_text(json.dumps(report, indent=2) + "\n")


if __name__ == "__main__":
    main()
