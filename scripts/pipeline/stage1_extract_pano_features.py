"""Stage 1 — Extract 4-heading DINOv2 embeddings and concat into (4D,) per pano.

Image path convention (auto-detected):
  Vienna  : <root>/<pid[0]>/<pid[1]>/<pid>/<pid>_<heading>.jpg
  HongKong: <root>/<pid[0].lower>/<pid[1].lower>/<pid[2].lower>/<pid>_<heading>.jpg

Exportable:
  extract_pano_features(pano_df, img_root, model_name, batch_size, device) -> (df, report)
"""
from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Literal

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from PIL import Image
from torchvision import transforms

from scripts.core.io_utils import assert_l2_normed, checkpoint, save_report

HEADINGS = [0, 90, 180, 270]

# Gated HF DINOv3 checkpoints (need an accepted-licence HF token in
# ~/.cache/huggingface/token; loaded via transformers AutoModel).
DINOV3_HF_IDS = {
    "dinov3_vitl16": "facebook/dinov3-vitl16-pretrain-lvd1689m",
}

_IMAGENET_MEAN = [0.485, 0.456, 0.406]
_IMAGENET_STD = [0.229, 0.224, 0.225]

# DINOv2 (torch.hub): shortest-edge resize 256 + center-crop 224.
_TRANSFORM_DINOV2 = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(_IMAGENET_MEAN, _IMAGENET_STD),
])
# DINOv3 (matches its HF AutoImageProcessor): direct resize to 224x224, no crop.
_TRANSFORM_DINOV3 = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(_IMAGENET_MEAN, _IMAGENET_STD),
])
# Back-compat default (DINOv2 recipe); per-model transform comes from the loader.
TRANSFORM = _TRANSFORM_DINOV2


def _img_path_vienna(img_root: Path, pid: str, heading: int) -> Path:
    return img_root / pid[0] / pid[1] / pid / f"{pid}_{heading}.jpg"


def _img_path_hk(img_root: Path, pid: str, heading: int) -> Path:
    return img_root / pid[0].lower() / pid[1].lower() / pid[2].lower() / f"{pid}_{heading}.jpg"


def _load_image(path: Path, transform=TRANSFORM) -> torch.Tensor | None:
    """Return (3,224,224) tensor or None if file missing/corrupt."""
    try:
        img = Image.open(path).convert("RGB")
        return transform(img)
    except Exception:
        return None


class _Extractor:
    """Uniform feature extractor over DINOv2 (torch.hub) and DINOv3 (HF).

    Callable: x (N,3,224,224) -> (N, embed_dim) global/CLS feature. Carries the
    matching preprocessing transform and embed_dim so callers stay model-agnostic.
    """
    def __init__(self, model, kind: str, embed_dim: int, transform):
        self.model, self.kind, self.embed_dim, self.transform = model, kind, embed_dim, transform

    def to(self, device):
        self.model.to(device); return self

    def half(self):
        self.model.half(); return self

    def __call__(self, x):
        if self.kind == "dinov3":
            return self.model(pixel_values=x).pooler_output   # (N, hidden_size)
        return self.model(x)                                   # dinov2 CLS feature


def _load_model(model_name: str, device: torch.device) -> "_Extractor":
    if model_name in ("dinov2_vitb14", "dinov2_vitl14"):
        model = torch.hub.load("facebookresearch/dinov2", model_name,
                               pretrained=True, verbose=False).eval().to(device)
        return _Extractor(model, "dinov2", model.embed_dim, _TRANSFORM_DINOV2)
    if model_name in DINOV3_HF_IDS:
        from transformers import AutoModel
        model = AutoModel.from_pretrained(DINOV3_HF_IDS[model_name]).eval().to(device)
        return _Extractor(model, "dinov3", model.config.hidden_size, _TRANSFORM_DINOV3)
    raise ValueError(f"Unknown model: {model_name}")


def extract_pano_features(
    pano_df: pd.DataFrame,
    img_root: str | Path,
    model_name: str = "dinov2_vitb14",
    batch_size: int = 32,
    device: str | None = None,
    path_style: Literal["vienna", "hongkong", "auto"] = "auto",
    max_panos: int | None = None,
    img_root_fallback: str | None = None,
) -> tuple[pd.DataFrame, dict]:
    """
    Extract 4-heading DINOv2 embeddings per pano, concat to (4D,), L2 norm.

    Parameters
    ----------
    pano_df    : DataFrame with columns [pano_id, city, lat, lon]
    img_root   : root directory of images
    path_style : 'vienna' | 'hongkong' | 'auto'
    max_panos  : limit for testing (None = all)
    """
    img_root = Path(img_root)
    fb_root  = Path(img_root_fallback) if img_root_fallback else None
    _device  = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))

    print(f"Loading {model_name} on {_device}...")
    model = _load_model(model_name, _device)
    D_single = model.embed_dim   # 768 for vitb14, 1024 for vitl14
    D_total  = D_single * 4

    # Detect path style
    if path_style == "auto":
        city = str(pano_df["city"].iloc[0]).lower()
        path_style = "hongkong" if "hong" in city or "hk" in city else "vienna"
    path_fn = _img_path_hk if path_style == "hongkong" else _img_path_vienna

    if max_panos is not None:
        pano_df = pano_df.head(max_panos)

    pano_ids   = pano_df["pano_id"].tolist()
    n          = len(pano_ids)
    results    = []
    n_missing  = 0
    n_partial  = 0

    print(f"Extracting features for {n} panos (batch={batch_size})...")

    for start in range(0, n, batch_size):
        batch_pids = pano_ids[start: start + batch_size]
        batch_embs = []

        # Collect all (pid, heading) images for this batch
        for pid in batch_pids:
            heading_tensors = []
            missing_headings = 0
            for h in HEADINGS:
                p = path_fn(img_root, pid, h)
                if fb_root is not None and not p.exists():
                    p = path_fn(fb_root, pid, h)   # fall back to NAS for missing images
                t = _load_image(p, model.transform)
                if t is None:
                    t = torch.zeros(3, 224, 224)
                    missing_headings += 1
                heading_tensors.append(t)
            if missing_headings == 4:
                n_missing += 1
            elif missing_headings > 0:
                n_partial += 1
            batch_embs.append(heading_tensors)  # list of 4 tensors per pano

        # Stack: (B*4, 3, 224, 224) and extract in one forward pass
        flat = torch.stack(
            [t for pano_ts in batch_embs for t in pano_ts]
        ).to(_device)

        with torch.no_grad():
            feats = model(flat)   # (B*4, D_single)

        feats = feats.cpu().float()
        B = len(batch_pids)
        feats = feats.reshape(B, 4, D_single)         # (B, 4, D_single)
        concat = feats.reshape(B, D_total)            # (B, 4*D_single)
        normed = F.normalize(concat, dim=1)           # L2 norm

        for i, pid in enumerate(batch_pids):
            results.append(normed[i].numpy())

        if (start // batch_size) % 10 == 0:
            print(f"  {start + len(batch_pids)}/{n} panos done")

    # Build output DataFrame
    out_df = pano_df[["pano_id", "city", "lat", "lon"]].copy().reset_index(drop=True)
    if "heading" in pano_df.columns:
        out_df["heading"] = pano_df["heading"].values
    else:
        out_df["heading"] = 0.0

    emb_array = np.stack(results, axis=0).astype(np.float32)
    assert_l2_normed(emb_array, name="pano_embedding")

    out_df["pano_embedding"] = list(emb_array)
    out_df["feature_model"]  = model_name

    report = {
        "n_panos":          n,
        "n_missing_all":    n_missing,
        "n_partial":        n_partial,
        "model":            model_name,
        "D_single":         D_single,
        "D_total":          D_total,
        "device":           str(_device),
        "path_style":       path_style,
    }

    checkpoint(n_missing / max(n, 1) < 0.10,
               f"too many panos missing all images: {n_missing}/{n}")

    return out_df, report


if __name__ == "__main__":
    import argparse, sqlite3

    ap = argparse.ArgumentParser()
    ap.add_argument("--city",       required=True,  choices=["Vienna", "HongKong"])
    ap.add_argument("--model",      default="dinov2_vitb14")
    ap.add_argument("--batch-size", type=int, default=32)
    ap.add_argument("--max-panos",  type=int, default=None)
    ap.add_argument("--output-dir", default=None)
    args = ap.parse_args()

    # Prefer the local data mirror (./data, populated by scripts/copy_data.py);
    # fall back to the NAS source if the local copy is absent.
    _local = Path(__file__).resolve().parent.parent.parent / "data/SVIs/GSV"
    _nas   = Path("/workplace/nas_data/houce/urban_visual_gene/SVIs/GSV")
    NAS    = _local if _local.exists() else _nas

    if args.city == "Vienna":
        pano_df = pd.read_parquet(NAS / "metadata/Austria/Vienna/panoids.parquet")
        pano_df = pano_df.rename(columns={"panoid": "pano_id"})
        pano_df["city"] = "Vienna"
        img_root   = NAS / "images/Austria/Vienna"
        path_style = "vienna"
        out_dir    = Path(args.output_dir or "outputs/Austria/Vienna")
    else:
        con = sqlite3.connect(NAS / "metadata/China/HongKong/meta/China_HongKong.db")
        pano_df = pd.read_sql("SELECT panoid as pano_id, lat, lon, year FROM gsv WHERE download=1", con)
        con.close()
        pano_df["city"] = "HongKong"
        img_root   = NAS / "images/China/HongKong"
        path_style = "hongkong"
        out_dir    = Path(args.output_dir or "outputs/China/HongKong")

    out_dir.mkdir(parents=True, exist_ok=True)

    feat_df, report = extract_pano_features(
        pano_df, img_root,
        model_name=args.model,
        batch_size=args.batch_size,
        max_panos=args.max_panos,
        path_style=path_style,
    )

    feat_df.to_parquet(out_dir / "pano_features.parquet", index=False)
    save_report(out_dir / "stage_reports/stage1_report.json", report)
    print(json.dumps(report, indent=2))
    print(f"Saved {len(feat_df)} pano features → {out_dir}/pano_features.parquet")
