from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

from sae_experiments.models.base_sae import build_sae


def log(*args: object) -> None:
    print(f"[{time.strftime('%H:%M:%S')}]", *args, flush=True)


def cosine_recon_loss(x: torch.Tensor, xhat: torch.Tensor) -> torch.Tensor:
    return (1 - F.cosine_similarity(x, xhat, dim=-1)).mean()


def load_features(path: Path, load_ram: bool):
    z = np.load(path, mmap_mode=None if load_ram else "r")
    if z.ndim != 2:
        raise ValueError(f"expected feature matrix [N,D], got {z.shape}")
    return z


def make_batch(features, indices: torch.Tensor, device: str) -> torch.Tensor:
    idx = indices.cpu().numpy()
    x = torch.from_numpy(np.asarray(features[idx], dtype=np.float32))
    return x.to(device, non_blocking=True)


def train(args: argparse.Namespace) -> None:
    sample_path = Path(args.sample_path)
    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    ckpt = outdir / f"{args.model_type}_w{args.width}_k{args.k}.pt"
    report_path = outdir / "training_report.json"

    features = load_features(sample_path, args.load_ram)
    n, d = int(features.shape[0]), int(features.shape[1])
    if args.max_patches and args.max_patches < n:
        n = int(args.max_patches)
    if args.input_dim not in (0, d):
        raise ValueError(f"input_dim={args.input_dim} does not match features.shape[-1]={d}")
    if args.width <= 0:
        raise ValueError("--width must be positive")

    device = "cuda" if torch.cuda.is_available() and not args.cpu else "cpu"
    model_cfg = {
        "type": args.model_type,
        "latent_dim": args.width,
        "k": args.k,
        "decoder_unit_norm": True,
    }
    sae = build_sae(input_dim=d, config=model_cfg).to(device)
    opt = torch.optim.AdamW(sae.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    log(f"train {args.model_type} W={args.width} k={args.k} on {n:,}/{features.shape[0]:,} patches D={d} dev={device}")

    history = []
    steps_per_epoch = (n + args.batch_size - 1) // args.batch_size
    for epoch in range(args.epochs):
        perm = torch.randperm(n)
        total = 0.0
        active_total = 0.0
        seen = 0
        for step, start in enumerate(range(0, n, args.batch_size), start=1):
            batch_idx = perm[start:start + args.batch_size]
            x = make_batch(features, batch_idx, device)
            opt.zero_grad(set_to_none=True)
            acts, xhat = sae(x)
            loss = cosine_recon_loss(x, xhat)
            loss.backward()
            if args.grad_clip > 0:
                torch.nn.utils.clip_grad_norm_(sae.parameters(), args.grad_clip)
            opt.step()
            sae.normalize_decoder_()
            bs = int(x.shape[0])
            total += float(loss.item()) * bs
            active_total += float((acts > 0).sum(dim=1).float().mean().item()) * bs
            seen += bs
            if args.log_every and step % args.log_every == 0:
                log(
                    f"  ep{epoch + 1:03d}/{args.epochs} step {step:04d}/{steps_per_epoch} "
                    f"loss={total / seen:.5f} active={active_total / seen:.2f}"
                )
        row = {
            "epoch": epoch + 1,
            "train_loss": round(total / seen, 6),
            "active_dims_patch_mean": round(active_total / seen, 3),
        }
        history.append(row)
        log(f"epoch {epoch + 1:03d}: loss={row['train_loss']:.5f} active={row['active_dims_patch_mean']:.2f}")
        torch.save(
            {
                "state": {k: v.detach().cpu() for k, v in sae.state_dict().items()},
                "D": d,
                "K": args.width,
                "k": args.k,
                "topk": args.k,
                "model_type": args.model_type,
                "sample_path": str(sample_path),
                "n_train_patches": n,
                "epoch": epoch + 1,
                "history": history,
                "final_train_loss": history[-1]["train_loss"],
                "final_active_dims_patch_mean": history[-1]["active_dims_patch_mean"],
            },
            ckpt,
        )
        report_path.write_text(json.dumps({"checkpoint": str(ckpt), "history": history}, indent=2))

    log(f"done -> {ckpt}")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Train SAE from cached DINOv3 feature matrix.")
    p.add_argument("--sample-path", required=True)
    p.add_argument("--outdir", required=True)
    p.add_argument("--model-type", choices=["topk", "batch_topk", "jumprelu"], default="batch_topk")
    p.add_argument("--input-dim", type=int, default=0, help="0 = infer from features")
    p.add_argument("--width", type=int, default=1024)
    p.add_argument("--k", type=int, default=8)
    p.add_argument("--epochs", type=int, default=60)
    p.add_argument("--batch-size", type=int, default=16384)
    p.add_argument("--lr", type=float, default=3e-4)
    p.add_argument("--weight-decay", type=float, default=1e-5)
    p.add_argument("--grad-clip", type=float, default=1.0)
    p.add_argument("--max-patches", type=int, default=0)
    p.add_argument("--load-ram", action="store_true")
    p.add_argument("--cpu", action="store_true")
    p.add_argument("--log-every", type=int, default=100)
    return p.parse_args()


def main() -> None:
    train(parse_args())


if __name__ == "__main__":
    main()

