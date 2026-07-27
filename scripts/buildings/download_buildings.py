#!/usr/bin/env python3
"""Download GlobalBuildingAtlas (GBA.ODbLPolygon) building footprints for a city:
fetch the 5x5deg HuggingFace tile, clip to the city's bbox (from dashboard meta),
write data/buildings/<city>/buildings.parquet.

    /opt/conda/bin/python3 scripts/download_buildings.py <City>
"""
import os, sys, json
from pathlib import Path

ROOT = Path(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
TILES = {  # city -> GBA 5x5deg tile (repo subfolder/file)
    "Vienna":    "europe/e015_n50_e020_n45.geojson",
    "HongKong":  "asiaeast/e110_n25_e115_n20.geojson",
    "Singapore": "oceania/e100_n05_e105_n00.geojson",
    "Amsterdam": "europe/e000_n55_e005_n50.geojson",
    "CapeTown":  "africa/e015_s30_e020_s35.geojson",
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
    # ALL GBA tiles are actually EPSG:3857 (metres) — but some tiles mislabel their
    # CRS as 4326 in the file metadata (observed: Singapore, Amsterdam), which made
    # the old "3857" string check use a lon/lat bbox on metre coords -> 0 features.
    # So: always query + reproject as 3857, ignoring the (unreliable) metadata.
    tr = Transformer.from_crs(4326, 3857, always_xy=True)
    x0, y0 = tr.transform(bbox[0], bbox[1]); x1, y1 = tr.transform(bbox[2], bbox[3])
    gdf = pyogrio.read_dataframe(path, bbox=(x0, y0, x1, y1))
    gdf = gdf.set_crs(3857, allow_override=True).to_crs(4326)   # -> lon/lat for the dashboard
    gdf.to_parquet(outdir / "buildings.parquet")
    print(f"[bld] {city}: {len(gdf):,} buildings -> {outdir/'buildings.parquet'} | cols={list(gdf.columns)}", flush=True)


if __name__ == "__main__":
    main()
