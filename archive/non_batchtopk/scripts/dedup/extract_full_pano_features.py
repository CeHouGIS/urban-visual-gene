#!/usr/bin/env python3
"""Extract 4-heading features for the FULL download=1 pano library of a city.

Per pano: 4 images (0/90/180/270) -> model -> concat (4*embed_dim,) -> L2 norm.
Model is the shared loader from stage1 (DINOv2 torch.hub OR DINOv3 HF), default
DINOv3 ViT-L/16 (embed_dim 1024 -> 4096-d concat).

Resumable RAM-array output (checkpointed every 40 batches):
  outputs/full_feats/<city>_feats.f16.npy   (N, 4*embed_dim) float16, row order = meta
  outputs/full_feats/<city>_meta.parquet    (pano_id, lat, lon)   same order
  outputs/full_feats/<city>.progress        #panos done (for resume)

  python -m scripts.dedup.extract_full_pano_features HongKong [--model dinov3_vitl16] [--batch-panos 96]

Env: needs ~/.cache/huggingface/token (gated DINOv3) and, for transformers'
scipy import, LD_LIBRARY_PATH=<conda_env>/lib (newer libstdc++).
"""
import scripts._env  # noqa: F401  (locks thread pools before numpy/torch)
import argparse
import os
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader, Subset

from scripts.core.cities import load_panos, img_root, path_style
from scripts.pipeline.stage1_extract_pano_features import (
    _load_model, _load_image, _img_path_hk, _img_path_vienna)

ROOT = Path(__file__).resolve().parents[2]
FEAT = ROOT / "outputs" / "full_feats"
FEAT.mkdir(parents=True, exist_ok=True)


class PanoDS(Dataset):
    """Loads a pano's 4 views in a worker process (parallel decode keeps GPU fed)."""
    def __init__(self, ids, pathfn, root, transform):
        self.ids, self.pathfn, self.root, self.tf = ids, pathfn, root, transform

    def __len__(self):
        return len(self.ids)

    def __getitem__(self, i):
        ts = [_load_image(self.pathfn(self.root, self.ids[i], h), self.tf)
              for h in (0, 90, 180, 270)]
        if all(t is not None for t in ts):
            return i, torch.stack(ts), True
        return i, torch.zeros(4, 3, 224, 224), False


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("city")
    ap.add_argument("--model", default="dinov3_vitl16")
    ap.add_argument("--batch-panos", type=int, default=96)
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--limit", type=int, default=None, help="cap #panos (smoke test)")
    args = ap.parse_args()

    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = _load_model(args.model, dev)
    # bf16 on GPU: ~fp16 speed but fp32 dynamic range, so DINOv3's layernorm/
    # attention do NOT overflow to NaN the way pure fp16 (model.half()) does.
    lowp = dev.type == "cuda"
    if lowp:
        model.model.to(torch.bfloat16)
    D = 4 * model.embed_dim                 # 4096 for DINOv3 ViT-L/16

    df = load_panos(args.city, None).reset_index(drop=True)   # all download=1
    if args.limit:
        df = df.head(args.limit).reset_index(drop=True)
    ids = df["pano_id"].astype(str).tolist()
    N = len(ids)
    pathfn = _img_path_hk if path_style(args.city) == "hongkong" else _img_path_vienna
    root = img_root(args.city)
    df[["pano_id", "lat", "lon"]].to_parquet(FEAT / f"{args.city}_meta.parquet", index=False)

    outp = FEAT / f"{args.city}_feats.f16.npy"
    prog = FEAT / f"{args.city}.progress"
    start = int(prog.read_text()) if (prog.exists() and outp.exists()) else 0
    feats = np.load(outp) if start > 0 else np.zeros((N, D), np.float16)
    if feats.shape != (N, D):              # model/scale changed -> restart clean
        feats = np.zeros((N, D), np.float16); start = 0

    def checkpoint(done):
        tmp = outp.with_suffix(".tmp.npy")
        np.save(tmp, feats); os.replace(tmp, outp)
        prog.write_text(str(done))

    print(f"[feat] {args.city}: {N:,} panos, model={args.model}, D={D}, "
          f"resume@{start}, dev={dev}, workers={args.workers}", flush=True)

    ds = Subset(PanoDS(ids, pathfn, root, model.transform), list(range(start, N)))
    # timeout: a hung Lustre read in a worker raises after N s (instead of a
    # silent forever-stall) so the job fails fast and a resubmit resumes from
    # the last checkpoint rather than burning the whole walltime.
    loader = DataLoader(ds, batch_size=args.batch_panos, num_workers=args.workers,
                        prefetch_factor=4, pin_memory=True, timeout=300)
    done = start; b = 0
    for gidx, imgs, valid in loader:           # imgs: (B,4,3,224,224)
        B = imgs.shape[0]
        x = imgs.view(B * 4, 3, 224, 224).to(dev, non_blocking=True)
        if lowp:
            x = x.to(torch.bfloat16)
        with torch.no_grad():
            emb = model(x).float().cpu().numpy().reshape(B, D)
        if b == 0 and not np.isfinite(emb).all():
            raise RuntimeError(f"[feat] {args.city}: non-finite features in first "
                               f"batch (precision/dtype issue) — aborting fast")
        emb /= (np.linalg.norm(emb, axis=1, keepdims=True) + 1e-9)
        gi = gidx.numpy(); vv = valid.numpy()
        for k in range(B):
            feats[gi[k]] = emb[k].astype(np.float16) if vv[k] else 0
        done = int(gi.max()) + 1; b += 1
        if b % 20 == 0:
            checkpoint(done)
            print(f"[feat] {args.city}: {done:,}/{N:,}", flush=True)
    checkpoint(N)
    n_missing = int((~np.isfinite(feats).all(1) | (np.abs(feats).sum(1) == 0)).sum())
    print(f"[feat] {args.city}: done -> {outp.name}  (missing-all-4: {n_missing:,}/{N:,})", flush=True)


if __name__ == "__main__":
    main()
