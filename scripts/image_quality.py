"""Cheap per-image quality features + heuristic auto-labels (no torch).

Targets four defect families: over-exposure/glare, blur/defocus,
under-exposure/corrupt, occlusion/no-information. Features are computed on a
fixed-size grayscale+RGB crop so they are comparable across images.

  python -m scripts.image_quality --city Vienna --max-panos 1500
  -> outputs/quality/Vienna_features.parquet  (one row per pano x heading)
"""
from __future__ import annotations

import scripts._env  # noqa: F401

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.ndimage import laplace

from scripts.cities import img_root, img_root_fallback, load_panos, path_style
from scripts.stage1_extract_pano_features import _img_path_hk, _img_path_vienna

HEADINGS = [0, 90, 180, 270]
SIZE = 256

FEAT_COLS = ["mean_b", "std_b", "frac_sat", "frac_dark", "lap_var",
             "edge_density", "entropy", "dom_frac", "colorfulness",
             "rg_mean", "yb_mean"]


def image_features(rgb: np.ndarray) -> dict:
    """rgb: HxWx3 float in [0,1]. Returns cheap quality descriptors."""
    R, G, B = rgb[..., 0], rgb[..., 1], rgb[..., 2]
    g = 0.299 * R + 0.587 * G + 0.114 * B           # luminance
    hist, _ = np.histogram(g, bins=64, range=(0, 1), density=True)
    p = hist / (hist.sum() + 1e-12)
    entropy = float(-(p * np.log2(p + 1e-12)).sum())
    lap = laplace(g)
    rg = R - G
    yb = 0.5 * (R + G) - B
    colorfulness = float(np.sqrt(rg.std() ** 2 + yb.std() ** 2)
                         + 0.3 * np.sqrt(rg.mean() ** 2 + yb.mean() ** 2))
    return dict(
        mean_b=float(g.mean()),
        std_b=float(g.std()),
        frac_sat=float((g > 0.96).mean()),          # over-exposure
        frac_dark=float((g < 0.04).mean()),         # under-exposure
        lap_var=float(lap.var()),                   # sharpness (low = blur)
        edge_density=float((np.abs(lap) > 0.04).mean()),
        entropy=entropy,                            # info content (low = blank)
        dom_frac=float(hist.max() / (hist.sum() + 1e-12)),  # single-tone share
        colorfulness=colorfulness,
        rg_mean=float(rg.mean()),                   # colour cast (glare)
        yb_mean=float(yb.mean()),
    )


def heuristic_label(f: dict) -> tuple[str, list[str]]:
    """Conservative auto-label: 'bad' / 'good' / 'ambiguous' + defect reasons.

    Thresholds set from the Vienna feature distribution (p1/5/95/99). Only clear
    cases are labeled bad/good; the rest are 'ambiguous' and excluded from
    training (the trained model later resolves them).
    """
    # High-PRECISION rules only: cheap global features cannot separate a valid
    # low-texture open scene (highway + big sky) from genuine blur/occlusion, so
    # we do NOT auto-flag those (they false-positive on open roads). We flag the
    # defects that ARE reliably separable: over-exposure, colour glare, very
    # dark, and near-uniform/corrupt frames.
    reasons = []
    if f["frac_sat"] > 0.30 or f["mean_b"] > 0.80:
        reasons.append("overexposed")
    # magenta/pink glare is a strong colour CAST (R>>G, high B), not just
    # colourfulness — rg_mean>0.10 is the extreme tail (p99~0.05).
    if f["rg_mean"] > 0.10 and f["yb_mean"] < 0.02:
        reasons.append("glare")
    if f["mean_b"] < 0.15 or f["frac_dark"] > 0.50:
        reasons.append("dark")
    if f["dom_frac"] > 0.45 or f["std_b"] < 0.05:
        reasons.append("corrupt")
    if reasons:
        return "bad", reasons
    # clearly good: mid brightness, no colour cast, not a single tone
    if (0.28 <= f["mean_b"] <= 0.72 and f["frac_sat"] < 0.15
            and abs(f["rg_mean"]) < 0.06 and f["dom_frac"] < 0.22):
        return "good", []
    return "ambiguous", []


def _load(city, pid, h):
    fn = _img_path_hk if path_style(city) == "hongkong" else _img_path_vienna
    p = fn(img_root(city), pid, h)
    if not p.exists():
        fb = img_root_fallback(city)
        if fb is not None:
            p = fn(fb, pid, h)
    if not p.exists():
        return None
    from PIL import Image
    return np.asarray(Image.open(p).convert("RGB").resize((SIZE, SIZE)),
                      dtype=np.float32) / 255.0


def filter_panos(pano_df, city, model_path="models/quality_model.joblib",
                 bad_headings=1, threshold=0.5):
    """Drop bad-quality panos before feature extraction.

    Scores each pano's 4 headings with the trained quality model and removes any
    pano with >= bad_headings headings classified bad. Returns (kept_df, report).
    If the model is missing, returns the input unchanged.
    """
    import joblib
    if not Path(model_path).exists():
        return pano_df, {"excluded": 0, "note": "no quality model; pass-through"}
    bundle = joblib.load(model_path)
    clf, cols = bundle["model"], bundle["features"]

    n_bad = {}
    for pid in pano_df["pano_id"]:
        bad = 0
        for h in HEADINGS:
            rgb = _load(city, pid, h)
            if rgb is None:
                continue
            f = image_features(rgb)
            if clf.predict_proba(np.array([[f[c] for c in cols]]))[0, 1] > threshold:
                bad += 1
        n_bad[pid] = bad
    keep = pano_df[pano_df["pano_id"].map(lambda p: n_bad.get(p, 0) < bad_headings)]
    rep = {"n_panos": len(pano_df), "excluded": len(pano_df) - len(keep),
           "bad_headings_threshold": bad_headings}
    return keep.reset_index(drop=True), rep


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--city", required=True, choices=["Vienna", "HongKong"])
    ap.add_argument("--max-panos", type=int, default=1500)
    args = ap.parse_args()
    out = Path("outputs/quality"); out.mkdir(parents=True, exist_ok=True)

    panos = load_panos(args.city, args.max_panos)
    rows, n_missing = [], 0
    for i, pid in enumerate(panos["pano_id"]):
        for h in HEADINGS:
            rgb = _load(args.city, pid, h)
            if rgb is None:
                n_missing += 1
                continue
            rows.append({"pano_id": pid, "heading": h, **image_features(rgb)})
        if (i + 1) % 300 == 0:
            print(f"  {i+1}/{len(panos)} panos", flush=True)

    df = pd.DataFrame(rows)
    df.to_parquet(out / f"{args.city}_features.parquet", index=False)
    print(f"{args.city}: {len(df)} images ({n_missing} missing) -> "
          f"{out/(args.city+'_features.parquet')}")
    # quick distribution peek to help set thresholds
    for c in ["mean_b", "frac_sat", "frac_dark", "lap_var", "entropy",
              "dom_frac", "colorfulness"]:
        q = df[c].quantile([0.01, 0.05, 0.5, 0.95, 0.99]).round(4).tolist()
        print(f"  {c:13s} p1/5/50/95/99 = {q}")


if __name__ == "__main__":
    main()
