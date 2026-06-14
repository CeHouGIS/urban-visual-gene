"""Isolated Stage 1 runner (uses torch) — extract DINOv2 pano features.

Runs in its own process: torch must never share a process with the
geopandas/scipy stages (OpenMP/MKL clash → native segfaults).

  python -m scripts.run_stage1 --city HongKong --max-panos 500
"""
from __future__ import annotations

import scripts._env  # noqa: F401  (sets thread limits before numpy/torch)

import argparse
from pathlib import Path

from scripts.cities import (img_root, img_root_fallback, load_panos, out_dir,
                            path_style)
from scripts.io_utils import save_report
from scripts.stage1_extract_pano_features import extract_pano_features


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--city", required=True, choices=["Vienna", "HongKong"])
    ap.add_argument("--max-panos", type=int, default=None)
    ap.add_argument("--model", default="dinov2_vitb14")
    ap.add_argument("--batch-size", type=int, default=32)
    ap.add_argument("--out", default=None, help="override output dir")
    ap.add_argument("--exclude-bad", action="store_true",
                    help="drop low-quality panos via models/quality_model.joblib")
    args = ap.parse_args()

    out = Path(args.out) if args.out else out_dir(args.city)
    (out / "stage_reports").mkdir(parents=True, exist_ok=True)

    pano_df = load_panos(args.city, args.max_panos)
    if args.exclude_bad:
        from scripts.image_quality import filter_panos
        pano_df, qrep = filter_panos(pano_df, args.city)
        print(f"  quality filter: excluded {qrep['excluded']}/{qrep['n_panos']} panos")
    feat_df, r1 = extract_pano_features(
        pano_df, img_root(args.city),
        model_name=args.model, batch_size=args.batch_size,
        path_style=path_style(args.city),
        img_root_fallback=img_root_fallback(args.city),
    )
    feat_df.to_parquet(out / "pano_features.parquet", index=False)
    save_report(out / "stage_reports/stage1_report.json", r1)
    print(f"Stage 1 ✓ {len(feat_df)} panos, D={r1['D_total']}, "
          f"missing={r1['n_missing_all']}")


if __name__ == "__main__":
    main()
