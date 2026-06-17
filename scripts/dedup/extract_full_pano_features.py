#!/usr/bin/env python3
"""Extract DINOv2 4-heading features for the FULL on-disk pano library (for dedup).
Per pano: 4 images (0/90/180/270) -> DINOv2-ViT-B/14 -> concat (3072,) -> L2 norm.

Resumable memmap output:
  outputs/full_feats/<city>_feats.f16.npy  (N, 3072) float16
  outputs/full_feats/<city>_ids.txt        kept ids (panos with all 4 images), same order
  outputs/full_feats/<city>.progress        #panos done (for resume)

    /opt/conda/bin/python3 scripts/extract_full_pano_features.py <City> [batch_panos]
"""
import scripts._env  # noqa: F401
import os, sys
from pathlib import Path
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader, Subset
from scripts.pipeline.stage1_extract_pano_features import (_load_model, _load_image,
                                                  _img_path_vienna, _img_path_hk)


class PanoDS(Dataset):
    """Loads a pano's 4 views in a worker process (parallel decode keeps GPU fed)."""
    def __init__(self, ids, pathfn, img_root):
        self.ids, self.pathfn, self.img_root = ids, pathfn, img_root
    def __len__(self): return len(self.ids)
    def __getitem__(self, i):
        ts = [_load_image(self.pathfn(self.img_root, self.ids[i], h)) for h in (0, 90, 180, 270)]
        if all(t is not None for t in ts):
            return i, torch.stack(ts), True
        return i, torch.zeros(4, 3, 224, 224), False

ROOT = Path(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
DATA = ROOT / "dashboard" / "data"
FEAT = ROOT / "outputs" / "full_feats"; FEAT.mkdir(parents=True, exist_ok=True)
CFG = {"Vienna": ("data/SVIs/GSV/images/Austria/Vienna", _img_path_vienna),
       "HongKong": ("data/SVIs/GSV/images/China/HongKong", _img_path_hk)}


def main():
    city = sys.argv[1]
    bp = int(sys.argv[2]) if len(sys.argv) > 2 else 96       # panos per batch (×4 imgs)
    img_root, pathfn = CFG[city]; img_root = ROOT / img_root
    ids = DATA.joinpath(f"panos_{city}_ids.txt").read_text().split("\n")
    N = len(ids)
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = _load_model("dinov2_vitb14", dev)
    half = dev.type == "cuda"
    if half: model.half()      # explicit fp16 (faster, avoids autocast Float/Half mismatch)
    outp = FEAT / f"{city}_feats.f16.npy"; prog_f = FEAT / f"{city}.progress"
    start = int(prog_f.read_text()) if (prog_f.exists() and outp.exists()) else 0
    feats = np.load(outp) if start > 0 else np.zeros((N, 3072), np.float16)   # RAM array (resume-aware)
    def checkpoint(done):
        tmp = outp.with_suffix(".tmp.npy"); np.save(tmp, feats); os.replace(tmp, outp)
        prog_f.write_text(str(done))
    print(f"[feat] {city}: {N:,} panos, resume@{start}, batch={bp}, dev={dev} (8 workers)", flush=True)

    ds = Subset(PanoDS(ids, pathfn, img_root), list(range(start, N)))
    loader = DataLoader(ds, batch_size=bp, num_workers=8, prefetch_factor=4, pin_memory=True)
    done = start; b = 0
    for gidx, imgs, valid in loader:           # imgs:(B,4,3,224,224)
        B = imgs.shape[0]
        x = imgs.view(B * 4, 3, 224, 224).to(dev, non_blocking=True)
        if half: x = x.half()
        with torch.no_grad():
            emb = model(x).float().cpu().numpy().reshape(B, 4 * 768)
        emb /= (np.linalg.norm(emb, axis=1, keepdims=True) + 1e-9)
        gi = gidx.numpy(); vv = valid.numpy()
        for k in range(B):
            feats[gi[k]] = emb[k].astype(np.float16) if vv[k] else 0
        done = int(gi.max()) + 1; b += 1
        if b % 40 == 0:
            checkpoint(done); print(f"[feat] {city}: {done:,}/{N:,}", flush=True)
    checkpoint(N)
    print(f"[feat] {city}: done -> {outp.name}", flush=True)


if __name__ == "__main__":
    main()
