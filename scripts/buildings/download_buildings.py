#!/usr/bin/env python3
"""Download GlobalBuildingAtlas (GBA.ODbLPolygon) building footprints for a city:
fetch the 5x5deg HuggingFace tile, clip to the city's bbox (from dashboard meta),
write data/buildings/<city>/buildings.parquet.

    /opt/conda/bin/python3 scripts/download_buildings.py <City>
"""
import os, sys, json
from pathlib import Path

ROOT = Path(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
TILES = {  # city -> (repo subfolder/file)
    "Vienna":   "europe/e015_n50_e020_n45.geojson",
    "HongKong": "asiaeast/e110_n25_e115_n20.geojson",
}
REPO = "zhu-xlab/GBA.ODbLPolygon"


def main():
    city = sys.argv[1]
    meta = json.load(open(ROOT / "dashboard/data/meta.json"))
    b = meta["cities"][city]["bounds"]            # [lonmin,latmin,lonmax,latmax]
    buf = 0.005
    bbox = (b[0] - buf, b[1] - buf, b[2] + buf, b[3] + buf)
    outdir = ROOT / "data/buildings" / city; outdir.mkdir(parents=True, exist_ok=True)
    tdir = ROOT / "data/buildings/_tiles"; tdir.mkdir(parents=True, exist_ok=True)

    from huggingface_hub import hf_hub_download
    print(f"[bld] {city}: downloading tile {TILES[city]} …", flush=True)
    path = hf_hub_download(repo_id=REPO, filename=TILES[city], repo_type="dataset",
                           local_dir=str(tdir))
    print(f"[bld] {city}: tile at {path} ({os.path.getsize(path)/1e9:.2f} GB); clipping bbox {bbox}…", flush=True)

    import pyogrio
    from pyproj import Transformer
    # GBA tiles are EPSG:3857 (metres) — convert the lon/lat bbox before filtering
    info = pyogrio.read_info(path)
    crs = str(info.get("crs") or "EPSG:4326")
    if "3857" in crs:
        tr = Transformer.from_crs(4326, 3857, always_xy=True)
        x0, y0 = tr.transform(bbox[0], bbox[1]); x1, y1 = tr.transform(bbox[2], bbox[3])
        qbbox = (x0, y0, x1, y1)
    else:
        qbbox = bbox
    gdf = pyogrio.read_dataframe(path, bbox=qbbox)
    if gdf.crs and gdf.crs.to_epsg() != 4326:
        gdf = gdf.to_crs(4326)                     # store as lon/lat for the dashboard
    gdf.to_parquet(outdir / "buildings.parquet")
    print(f"[bld] {city}: {len(gdf):,} buildings -> {outdir/'buildings.parquet'} | cols={list(gdf.columns)}", flush=True)


if __name__ == "__main__":
    main()
