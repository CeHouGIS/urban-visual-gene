#!/usr/bin/env python3
"""Export the FULL on-disk pano index per city (every pano that has images), so
clicking a node maps to a truly nearby street view (panos are now denser than nodes).

Output (compact, lazy-loaded by the dashboard):
  panos_<city>_xy.bin   Float32 [lon, lat, ...]   one pano per pair
  panos_<city>_ids.txt  newline-joined pano ids    (same order)
The image URL is rebuilt in JS from the id via the per-city path rule.
"""
import os, sqlite3
import numpy as np
import pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
OUT = os.path.join(ROOT, "dashboard", "data")
IMG = "data/SVIs/GSV/images"


def disk_panoids(city_path):
    """All pano ids that have a _0.jpg under the city's image tree."""
    base = os.path.join(ROOT, IMG, city_path)
    ids = []
    for dp, _, files in os.walk(base):
        for f in files:
            if f.endswith("_0.jpg"):
                ids.append(f[:-6])   # strip "_0.jpg"
    return set(ids)


def write(city, rows):
    xy = np.empty(len(rows) * 2, np.float32)
    for i, (_, lon, lat) in enumerate(rows):
        xy[i * 2] = lon; xy[i * 2 + 1] = lat
    xy.tofile(os.path.join(OUT, f"panos_{city}_xy.bin"))
    open(os.path.join(OUT, f"panos_{city}_ids.txt"), "w").write("\n".join(r[0] for r in rows))
    print(f"[panos-full] {city}: {len(rows):,} panos -> xy.bin + ids.txt")


def main():
    # Vienna: coords from panoids.parquet, keep those with images on disk
    vp = pd.read_parquet(os.path.join(ROOT, "data/SVIs/GSV/metadata/Austria/Vienna/panoids.parquet"),
                         columns=["panoid", "lat", "lon"]).drop_duplicates("panoid")
    have = disk_panoids("Austria/Vienna")
    rows = [(p, float(lo), float(la)) for p, la, lo in zip(vp.panoid, vp.lat, vp.lon) if p in have]
    write("Vienna", rows)

    # HongKong: on-disk panoids joined to coords from the sqlite db
    have = disk_panoids("China/HongKong")
    con = sqlite3.connect(os.path.join(ROOT, "data/SVIs/GSV/metadata/China/HongKong/meta/China_HongKong.db"))
    coord = {p: (lo, la) for p, la, lo in con.execute("SELECT panoid,lat,lon FROM gsv")}
    rows = [(p, float(coord[p][0]), float(coord[p][1])) for p in have if p in coord]
    write("HongKong", rows)


if __name__ == "__main__":
    main()
