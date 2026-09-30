#!/usr/bin/env python3
"""Bridge full_feats/<city>_feats.f16.npy (+ _meta.parquet) -> pano_features.parquet.

Produces the exact schema Stage 2/3 consume (pano_id, city, lat, lon, heading,
pano_embedding, feature_model), dropping panos whose 4 images were all missing
(all-zero rows). Written in 100k-row chunks (mmap read + ParquetWriter row
groups) so it is memory-bounded AND each row group's list column stays under
Arrow's int32 offset limit even at 4096-d for huge cities.

  python -m scripts.dedup.build_full_pano_features HongKong [--model dinov3_vitl16]
"""
import scripts._env  # noqa: F401
import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

from scripts.core.cities import out_dir

ROOT = Path(__file__).resolve().parents[2]
FEAT = ROOT / "outputs" / "full_feats"
CH = 100_000   # rows per chunk / row group


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("city")
    ap.add_argument("--model", default="dinov3_vitl16")
    ap.add_argument("--out", default=None, help="override output parquet path")
    args = ap.parse_args()

    feats = np.load(FEAT / f"{args.city}_feats.f16.npy", mmap_mode="r")  # (N, D) f16
    meta = pd.read_parquet(FEAT / f"{args.city}_meta.parquet")           # pano_id, lat, lon
    N, D = feats.shape
    assert len(meta) == N, f"meta {len(meta)} != feats {N}"
    out = Path(args.out) if args.out else out_dir(args.city) / "pano_features.parquet"
    out.parent.mkdir(parents=True, exist_ok=True)

    writer = None
    kept = dropped = 0
    cols = ["pano_id", "city", "lat", "lon", "heading", "pano_embedding", "feature_model"]
    for s in range(0, N, CH):
        e = min(s + CH, N)
        blk = np.asarray(feats[s:e], dtype=np.float32)
        keep = np.isfinite(blk).all(1) & (np.abs(blk).sum(1) > 0)   # drop missing/NaN panos
        dropped += int((~keep).sum())
        if not keep.any():
            continue
        blk = blk[keep]
        sub = meta.iloc[s:e][keep].copy().reset_index(drop=True)
        sub["city"] = args.city
        sub["heading"] = 0.0
        sub["feature_model"] = args.model
        sub["pano_embedding"] = list(blk)
        tbl = pa.Table.from_pandas(sub[cols], preserve_index=False)
        if writer is None:
            writer = pq.ParquetWriter(out, tbl.schema)
        writer.write_table(tbl)
        kept += len(sub)
        print(f"[bridge] {args.city}: {e:,}/{N:,} (kept {kept:,}, dropped {dropped:,})", flush=True)
    if writer is not None:
        writer.close()
    print(f"[bridge] {args.city}: wrote {kept:,} panos x {D}-d -> {out} "
          f"({pq.read_metadata(out).num_row_groups} row groups)", flush=True)


if __name__ == "__main__":
    main()
