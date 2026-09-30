"""Copy the data this project needs into ./data (mirrors NAS GSV layout).

Copies ALL metadata + ONLY the sampled images used by the 500-pano experiment
(Vienna + Hong Kong, random_state=42). Run from project root.
"""
from __future__ import annotations

import shutil
import sqlite3
import sys
from pathlib import Path

import pandas as pd

NAS = Path("/workplace/nas_data/houce/urban_visual_gene/SVIs/GSV")
DST = Path("data/SVIs/GSV")          # local mirror inside the project
HEADINGS = [0, 90, 180, 270]
MAX_PANOS = 500
SEED = 42


def _copy(src: Path, dst: Path):
    dst.parent.mkdir(parents=True, exist_ok=True)
    if dst.exists():
        return
    shutil.copy2(src, dst)


def copy_meta():
    print("[meta] Vienna metadata ...")
    vdir = "metadata/Austria/Vienna"
    for f in ["panoids.parquet", "panoids.csv",
              "edges.shp", "edges.dbf", "edges.shx", "edges.prj", "edges.cpg",
              "nodes.shp", "nodes.dbf", "nodes.shx", "nodes.prj", "nodes.cpg"]:
        src = NAS / vdir / f
        if src.exists():
            _copy(src, DST / vdir / f)

    print("[meta] Hong Kong metadata ...")
    _copy(NAS / "metadata/China/HongKong/meta/China_HongKong.db",
          DST / "metadata/China/HongKong/meta/China_HongKong.db")
    _copy(NAS / "metadata/China/HongKong/road/China_Hong_Kong.geojson",
          DST / "metadata/China/HongKong/road/China_Hong_Kong.geojson")


def _vienna_path(root: Path, pid: str, h: int) -> Path:
    return root / pid[0] / pid[1] / pid / f"{pid}_{h}.jpg"


def _hk_path(root: Path, pid: str, h: int) -> Path:
    return root / pid[0].lower() / pid[1].lower() / pid[2].lower() / f"{pid}_{h}.jpg"


def vienna_pano_ids() -> list[str]:
    # Reuse the exact sample already materialised by the experiment if present.
    feat = Path("outputs/Austria/Vienna/pano_features.parquet")
    if feat.exists():
        return pd.read_parquet(feat, columns=["pano_id"])["pano_id"].tolist()
    df = pd.read_parquet(NAS / "metadata/Austria/Vienna/panoids.parquet")
    df = df.rename(columns={"panoid": "pano_id"})
    return df.sample(MAX_PANOS, random_state=SEED)["pano_id"].tolist()


def hongkong_pano_ids() -> list[str]:
    con = sqlite3.connect(NAS / "metadata/China/HongKong/meta/China_HongKong.db")
    df = pd.read_sql("SELECT panoid as pano_id, lat, lon FROM gsv WHERE download=1", con)
    con.close()
    return df.sample(MAX_PANOS, random_state=SEED)["pano_id"].tolist()


def copy_images(ids: list[str], src_root: Path, dst_root: Path, path_fn):
    ok = miss = 0
    for pid in ids:
        for h in HEADINGS:
            src = path_fn(src_root, pid, h)
            if src.exists():
                _copy(src, path_fn(dst_root, pid, h))
                ok += 1
            else:
                miss += 1
    print(f"  copied={ok}, missing={miss}  ({len(ids)} panos x4)")


def main():
    copy_meta()

    print("[img] Vienna sampled images ...")
    vids = vienna_pano_ids()
    copy_images(vids, NAS / "images/Austria/Vienna",
                DST / "images/Austria/Vienna", _vienna_path)

    print("[img] Hong Kong sampled images ...")
    hids = hongkong_pano_ids()
    copy_images(hids, NAS / "images/China/HongKong",
                DST / "images/China/HongKong", _hk_path)

    size = sum(f.stat().st_size for f in Path("data").rglob("*") if f.is_file())
    print(f"\nDone. ./data total size = {size/1e6:.1f} MB")


if __name__ == "__main__":
    sys.exit(main())
