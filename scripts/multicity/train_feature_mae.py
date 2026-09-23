#!/usr/bin/env python3
"""Train a masked autoencoder over filtered DINOv3 ViT-B/16 patch tokens.

The encoder maps every 768-D DINO patch token to a 512-D contextual code.
During training 75% of the 14x14 positions are hidden and reconstructed.  At
inference ``encode_full`` returns a 14x14x512 tensor suitable for per-dimension
heatmaps.
"""
from __future__ import annotations

import scripts._env  # noqa: F401

import argparse
import csv
import json
import math
import mmap
import os
import time
from pathlib import Path
from typing import Optional

import numpy as np
import torch
from torch import nn

from scripts.multicity.config import CITIES, SEED, city_slug
from scripts.multicity.patch_config import (
    INPUT_DIM,
    OUTPUT_ROOT as TOKEN_ROOT,
    PATCHES_PER_IMAGE,
    WIDTH,
    token_path,
)

DEFAULT_DATA_ROOT = TOKEN_ROOT.parent / "feature_mae_n30x12800_qc"


def _sincos_1d(dim: int, positions: np.ndarray) -> np.ndarray:
    frequencies = 1.0 / (10000 ** (np.arange(dim // 2, dtype=np.float64) / (dim / 2)))
    angles = positions.reshape(-1, 1) * frequencies.reshape(1, -1)
    return np.concatenate((np.sin(angles), np.cos(angles)), axis=1)


def sincos_2d(dim: int, grid_size: int) -> torch.Tensor:
    if dim % 4:
        raise ValueError("2D sine/cosine dimension must be divisible by four")
    y, x = np.meshgrid(np.arange(grid_size), np.arange(grid_size), indexing="ij")
    embedding = np.concatenate(
        (_sincos_1d(dim // 2, y.reshape(-1)), _sincos_1d(dim // 2, x.reshape(-1))),
        axis=1,
    )
    return torch.from_numpy(embedding.astype(np.float32)).unsqueeze(0)


class FeatureMAE(nn.Module):
    def __init__(
        self,
        input_dim: int = INPUT_DIM,
        width: int = WIDTH,
        patches: int = PATCHES_PER_IMAGE,
        encoder_layers: int = 4,
        encoder_heads: int = 8,
        decoder_width: int = 256,
        decoder_layers: int = 2,
        decoder_heads: int = 8,
        mlp_ratio: float = 4.0,
        mask_ratio: float = 0.75,
    ) -> None:
        super().__init__()
        grid = int(round(math.sqrt(patches)))
        if grid * grid != patches:
            raise ValueError(f"patch count must be square, got {patches}")
        self.input_dim = input_dim
        self.width = width
        self.patches = patches
        self.mask_ratio = mask_ratio
        self.input_projection = nn.Linear(input_dim, width)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=width,
            nhead=encoder_heads,
            dim_feedforward=int(width * mlp_ratio),
            dropout=0.0,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(
            encoder_layer, num_layers=encoder_layers, enable_nested_tensor=False
        )
        self.encoder_norm = nn.LayerNorm(width)
        self.decoder_projection = nn.Linear(width, decoder_width)
        self.mask_token = nn.Parameter(torch.zeros(1, 1, decoder_width))
        decoder_layer = nn.TransformerEncoderLayer(
            d_model=decoder_width,
            nhead=decoder_heads,
            dim_feedforward=int(decoder_width * mlp_ratio),
            dropout=0.0,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.decoder = nn.TransformerEncoder(
            decoder_layer, num_layers=decoder_layers, enable_nested_tensor=False
        )
        self.decoder_norm = nn.LayerNorm(decoder_width)
        self.output_projection = nn.Linear(decoder_width, input_dim)
        self.register_buffer("encoder_position", sincos_2d(width, grid), persistent=True)
        self.register_buffer(
            "decoder_position", sincos_2d(decoder_width, grid), persistent=True
        )
        nn.init.normal_(self.mask_token, std=0.02)
        self.apply(self._initialize)

    @staticmethod
    def _initialize(module: nn.Module) -> None:
        if isinstance(module, nn.Linear):
            nn.init.xavier_uniform_(module.weight)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.LayerNorm):
            nn.init.ones_(module.weight)
            nn.init.zeros_(module.bias)

    def random_mask(self, tokens: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        batch, length, dim = tokens.shape
        keep = max(1, int(length * (1.0 - self.mask_ratio)))
        noise = torch.rand(batch, length, device=tokens.device)
        shuffle = noise.argsort(dim=1)
        restore = shuffle.argsort(dim=1)
        keep_ids = shuffle[:, :keep]
        visible = torch.gather(tokens, 1, keep_ids.unsqueeze(-1).expand(-1, -1, dim))
        mask = torch.ones(batch, length, device=tokens.device)
        mask[:, :keep] = 0
        mask = torch.gather(mask, 1, restore)
        return visible, mask, restore

    def forward(self, target: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        projected = self.input_projection(target) + self.encoder_position
        visible, mask, restore = self.random_mask(projected)
        encoded = self.encoder_norm(self.encoder(visible))
        decoded_visible = self.decoder_projection(encoded)
        missing = self.patches - decoded_visible.shape[1]
        sequence = torch.cat(
            (decoded_visible, self.mask_token.expand(len(target), missing, -1)), dim=1
        )
        sequence = torch.gather(
            sequence, 1, restore.unsqueeze(-1).expand(-1, -1, sequence.shape[-1])
        )
        sequence = sequence + self.decoder_position
        prediction = self.output_projection(self.decoder_norm(self.decoder(sequence)))
        return prediction, mask

    @torch.inference_mode()
    def encode_full(self, target: torch.Tensor) -> torch.Tensor:
        projected = self.input_projection(target) + self.encoder_position
        return self.encoder_norm(self.encoder(projected))


def masked_cosine_loss(target: torch.Tensor, prediction: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    per_patch = 1.0 - nn.functional.cosine_similarity(target, prediction, dim=-1)
    return (per_patch * mask).sum() / mask.sum().clamp_min(1)


class FilteredTokenStore:
    def __init__(
        self, cities: list[str], token_root: Path, data_root: Path,
        validation_images: int, seed: int,
        train_manifest_root: Optional[Path] = None,
        validation_manifest_root: Optional[Path] = None,
        epoch_images_per_city: int = 0,
    ) -> None:
        import pandas as pd

        self.cities = cities
        self.arrays: list[np.memmap] = []
        self.train_indices: list[np.ndarray] = []
        self.validation_indices: list[np.ndarray] = []
        self.counts: dict[str, int] = {}
        self.training_counts: dict[str, int] = {}
        self.validation_counts: dict[str, int] = {}
        for city in cities:
            manifest_root = train_manifest_root or (data_root / "filtered_manifests")
            manifest_path = manifest_root / f"{city_slug(city)}.parquet"
            frame = pd.read_parquet(manifest_path, columns=["source_image_index"])
            indices = frame["source_image_index"].to_numpy(np.int64)
            if validation_manifest_root is None:
                if len(indices) <= validation_images:
                    raise ValueError(f"{city}: only {len(indices)} clean images")
                city_seed = seed + sum((i + 1) * ord(ch) for i, ch in enumerate(city))
                order = np.random.default_rng(city_seed).permutation(indices)
                validation = order[:validation_images]
                training = order[validation_images:]
            else:
                validation_path = validation_manifest_root / f"{city_slug(city)}.parquet"
                validation_frame = pd.read_parquet(
                    validation_path, columns=["source_image_index"]
                )
                validation = validation_frame["source_image_index"].to_numpy(np.int64)
                training = indices
                overlap = np.intersect1d(training, validation, assume_unique=False)
                if len(overlap):
                    raise ValueError(
                        f"{city}: {len(overlap)} images overlap between train and validation"
                    )
            self.validation_indices.append(validation)
            self.train_indices.append(training)
            array = np.load(token_path(city, token_root), mmap_mode="r")
            if array.shape[1:] != (PATCHES_PER_IMAGE, INPUT_DIM):
                raise ValueError(f"{city}: unexpected token shape {array.shape}")
            self.arrays.append(array)
            if hasattr(array._mmap, "madvise"):
                array._mmap.madvise(mmap.MADV_RANDOM)
            self.training_counts[city] = len(training)
            self.validation_counts[city] = len(validation)
            self.counts[city] = len(training) + len(validation)
        natural_epoch_size = max(map(len, self.train_indices))
        self.train_images_per_city = epoch_images_per_city or natural_epoch_size
        if self.train_images_per_city <= 0:
            raise ValueError("epoch_images_per_city must be positive")

    @staticmethod
    def _balanced_order(indices: np.ndarray, target: int, rng: np.random.Generator) -> np.ndarray:
        pieces = []
        remaining = target
        while remaining:
            order = rng.permutation(indices)
            take = min(remaining, len(order))
            pieces.append(order[:take])
            remaining -= take
        return np.concatenate(pieces)

    def train_batches(self, per_city: int, rng: np.random.Generator):
        orders = [
            self._balanced_order(indices, self.train_images_per_city, rng)
            for indices in self.train_indices
        ]
        for batch_index, start in enumerate(
            range(0, self.train_images_per_city, per_city), 1
        ):
            blocks = []
            for array, order in zip(self.arrays, orders):
                # Sorting within a shuffled mini-batch preserves its samples while
                # turning scattered mmap reads into much friendlier ascending I/O.
                image_ids = np.sort(order[start : start + per_city])
                blocks.append(np.asarray(array[image_ids], dtype=np.float32))
            yield np.concatenate(blocks, axis=0)
            # Random reads across 30 x ~3.7 GB memmaps otherwise accumulate as
            # file-backed RSS until the 64 GB host is killed by its cgroup.
            # The current training batch is a regular ndarray copy, so these
            # source pages can be discarded safely and re-read next epoch.
            if batch_index % 16 == 0:
                self.release_file_cache()

    def validation_batches(self, per_city: int):
        count = len(self.validation_indices[0])
        for start in range(0, count, per_city):
            blocks = []
            for array, indices in zip(self.arrays, self.validation_indices):
                blocks.append(
                    np.asarray(array[indices[start : start + per_city]], dtype=np.float32)
                )
            yield np.concatenate(blocks, axis=0)
        self.release_file_cache()

    def release_file_cache(self) -> None:
        if not hasattr(mmap, "MADV_DONTNEED"):
            return
        for array in self.arrays:
            mapping = getattr(array, "_mmap", None)
            if mapping is not None and hasattr(mapping, "madvise"):
                mapping.madvise(mmap.MADV_DONTNEED)


def _atomic_torch_save(value: dict, path: Path) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    torch.save(value, temporary)
    os.replace(temporary, path)


def _config(args: argparse.Namespace) -> dict:
    return {
        key: str(value) if isinstance(value, Path) else value
        for key, value in vars(args).items()
    }


@torch.inference_mode()
def validate(
    model: FeatureMAE, store: FilteredTokenStore, per_city: int, microbatch: int,
) -> float:
    model.eval()
    total_loss = 0.0
    total_images = 0
    for array in store.validation_batches(per_city):
        for start in range(0, len(array), microbatch):
            batch = torch.from_numpy(array[start : start + microbatch]).to(
                "cuda", non_blocking=True
            )
            with torch.autocast("cuda", dtype=torch.bfloat16):
                prediction, mask = model(batch)
                loss = masked_cosine_loss(batch, prediction, mask)
            total_loss += float(loss) * len(batch)
            total_images += len(batch)
    return total_loss / total_images


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--cities", nargs="*", default=list(CITIES))
    parser.add_argument("--token-root", type=Path, default=TOKEN_ROOT)
    parser.add_argument("--data-root", type=Path, default=DEFAULT_DATA_ROOT)
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--width", type=int, default=WIDTH)
    parser.add_argument("--encoder-layers", type=int, default=4)
    parser.add_argument("--decoder-width", type=int, default=256)
    parser.add_argument("--decoder-layers", type=int, default=2)
    parser.add_argument("--mask-ratio", type=float, default=0.75)
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument(
        "--stop-after-epoch", type=int, default=0,
        help="exit cleanly after this 1-based epoch; keeps --epochs for LR scheduling",
    )
    parser.add_argument("--patience", type=int, default=3)
    parser.add_argument("--validation-images", type=int, default=128)
    parser.add_argument(
        "--train-manifest-root", type=Path,
        help="directory containing per-city training parquet manifests",
    )
    parser.add_argument(
        "--validation-manifest-root", type=Path,
        help="directory containing fixed per-city validation parquet manifests",
    )
    parser.add_argument(
        "--epoch-images-per-city", type=int, default=0,
        help="balanced images drawn per city per epoch; zero uses the largest training split",
    )
    parser.add_argument("--images-per-city", type=int, default=2)
    parser.add_argument("--microbatch", type=int, default=15)
    parser.add_argument("--gradient-accumulation", type=int, default=4)
    parser.add_argument(
        "--max-batches-per-epoch", type=int, default=0,
        help="debug/smoke limit; zero uses the complete filtered training set",
    )
    parser.add_argument("--lr", type=float, default=1.5e-4)
    parser.add_argument("--weight-decay", type=float, default=0.05)
    parser.add_argument("--seed", type=int, default=SEED)
    args = parser.parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required")
    if not 0 < args.mask_ratio < 1:
        raise ValueError("--mask-ratio must be between zero and one")
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.set_float32_matmul_precision("high")
    output = args.output_dir or (args.data_root / "mae")
    output.mkdir(parents=True, exist_ok=True)
    store = FilteredTokenStore(
        args.cities, args.token_root, args.data_root,
        args.validation_images, args.seed,
        train_manifest_root=args.train_manifest_root,
        validation_manifest_root=args.validation_manifest_root,
        epoch_images_per_city=args.epoch_images_per_city,
    )
    model = FeatureMAE(
        width=args.width,
        encoder_layers=args.encoder_layers,
        decoder_width=args.decoder_width,
        decoder_layers=args.decoder_layers,
        mask_ratio=args.mask_ratio,
    ).to("cuda")
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=args.lr, weight_decay=args.weight_decay, betas=(0.9, 0.95)
    )
    batches_per_epoch = math.ceil(store.train_images_per_city / args.images_per_city)
    if args.max_batches_per_epoch:
        batches_per_epoch = min(batches_per_epoch, args.max_batches_per_epoch)
    optimizer_steps_per_epoch = math.ceil(batches_per_epoch / args.gradient_accumulation)
    total_optimizer_steps = args.epochs * optimizer_steps_per_epoch
    warmup_steps = max(1, int(total_optimizer_steps * 0.05))

    def schedule(step: int) -> float:
        if step < warmup_steps:
            return (step + 1) / warmup_steps
        progress = (step - warmup_steps) / max(1, total_optimizer_steps - warmup_steps)
        return 0.5 * (1.0 + math.cos(math.pi * min(progress, 1.0)))

    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, schedule)
    last_path = output / "checkpoint_last.pt"
    best_path = output / "model_best.pt"
    start_epoch, best_loss, stale, history, optimizer_step = 0, float("inf"), 0, [], 0
    if last_path.exists():
        checkpoint = torch.load(last_path, map_location="cuda", weights_only=False)
        saved_config = checkpoint.get("config", {})
        current_config = _config(args)
        identity_keys = (
            "width", "encoder_layers", "decoder_width", "decoder_layers",
            "mask_ratio", "seed", "train_manifest_root",
            "validation_manifest_root", "epoch_images_per_city",
        )
        mismatches = {
            key: (saved_config.get(key), current_config.get(key))
            for key in identity_keys
            if key in saved_config and saved_config.get(key) != current_config.get(key)
        }
        if mismatches:
            raise ValueError(f"checkpoint configuration mismatch: {mismatches}")
        model.load_state_dict(checkpoint["model"])
        optimizer.load_state_dict(checkpoint["optimizer"])
        scheduler.load_state_dict(checkpoint["scheduler"])
        start_epoch = checkpoint["epoch"] + 1
        best_loss = checkpoint["best_loss"]
        stale = checkpoint["stale"]
        history = checkpoint["history"]
        optimizer_step = checkpoint["optimizer_step"]
        print(f"resuming at epoch {start_epoch + 1}", flush=True)
    print(
        f"Feature-MAE: {sum(p.numel() for p in model.parameters()):,} parameters, "
        f"batch={len(args.cities) * args.images_per_city}, microbatch={args.microbatch}, "
        f"mask={args.mask_ratio:.0%}",
        flush=True,
    )
    started = time.time()
    torch.cuda.reset_peak_memory_stats()
    rng = np.random.default_rng(args.seed + start_epoch)
    end_epoch = args.epochs
    if args.stop_after_epoch:
        end_epoch = min(end_epoch, args.stop_after_epoch)
    for epoch in range(start_epoch, end_epoch):
        model.train()
        optimizer.zero_grad(set_to_none=True)
        loss_sum = 0.0
        image_sum = 0
        for batch_index, array in enumerate(
            store.train_batches(args.images_per_city, rng), 1
        ):
            if batch_index > batches_per_epoch:
                break
            micro_count = math.ceil(len(array) / args.microbatch)
            for start in range(0, len(array), args.microbatch):
                batch = torch.from_numpy(array[start : start + args.microbatch]).to(
                    "cuda", non_blocking=True
                )
                with torch.autocast("cuda", dtype=torch.bfloat16):
                    prediction, mask = model(batch)
                    loss = masked_cosine_loss(batch, prediction, mask)
                (loss / (args.gradient_accumulation * micro_count)).backward()
                loss_sum += float(loss.detach()) * len(batch)
                image_sum += len(batch)
            if batch_index % args.gradient_accumulation == 0 or batch_index == batches_per_epoch:
                nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad(set_to_none=True)
                optimizer_step += 1
            if batch_index % 250 == 0 or batch_index == batches_per_epoch:
                print(
                    f"  epoch {epoch+1:02d} batch {batch_index:,}/{batches_per_epoch:,} "
                    f"loss={loss_sum/image_sum:.5f} "
                    f"gpu={torch.cuda.max_memory_allocated()/2**30:.2f}GiB",
                    flush=True,
                )
        validation_loss = validate(
            model, store, max(1, args.images_per_city), args.microbatch
        )
        row = {
            "epoch": epoch + 1,
            "train_loss": loss_sum / image_sum,
            "validation_loss": validation_loss,
            "learning_rate": optimizer.param_groups[0]["lr"],
            "elapsed_seconds": time.time() - started,
        }
        history.append(row)
        if validation_loss < best_loss - 1e-5:
            best_loss, stale = validation_loss, 0
            _atomic_torch_save(
                {"model": model.state_dict(), "epoch": epoch, "metrics": row,
                 "config": _config(args)},
                best_path,
            )
        else:
            stale += 1
        _atomic_torch_save(
            {
                "model": model.state_dict(), "optimizer": optimizer.state_dict(),
                "scheduler": scheduler.state_dict(), "epoch": epoch,
                "optimizer_step": optimizer_step, "best_loss": best_loss,
                "stale": stale, "history": history, "config": _config(args),
            },
            last_path,
        )
        print(
            f"epoch {epoch+1:02d}: train={row['train_loss']:.6f} "
            f"val={validation_loss:.6f}",
            flush=True,
        )
        if stale >= args.patience:
            print(f"early stopping after {stale} stale epochs", flush=True)
            break

    with (output / "history.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(history[0]))
        writer.writeheader()
        writer.writerows(history)
    best = torch.load(best_path, map_location="cpu", weights_only=False)
    report = {
        "model": "DINOv3 ViT-B/16 feature masked autoencoder",
        "input": "14x14 normalized DINOv3 patch tokens",
        "cities": args.cities,
        "clean_images_by_city": store.counts,
        "clean_images_total": sum(store.counts.values()),
        "train_images_by_city": store.training_counts,
        "validation_images_by_city": store.validation_counts,
        "epoch_images_per_city": store.train_images_per_city,
        "input_dim": INPUT_DIM,
        "latent_width": args.width,
        "patch_grid": [14, 14],
        "mask_ratio": args.mask_ratio,
        "encoder_layers": args.encoder_layers,
        "decoder_layers": args.decoder_layers,
        "best_epoch": best["epoch"] + 1,
        "best_metrics": best["metrics"],
        "peak_gpu_gib": torch.cuda.max_memory_allocated() / 2**30,
        "heatmap_contract": (
            f"encode_full(images) -> [B,196,{args.width}], "
            f"reshape to [B,14,14,{args.width}]"
        ),
    }
    (output / "training_report.json").write_text(json.dumps(report, indent=2) + "\n")


if __name__ == "__main__":
    main()
