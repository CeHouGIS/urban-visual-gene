#!/usr/bin/env python3
"""Write the DROPPED-by-dedup pano set (full minus kept) for the dashboard's
'show excluded points (gray)' toggle.
Output: panos_<city>_dropped_xy.bin (Float32 lon,lat) + _dropped_ids.txt"""
import os, sys
from pathlib import Path
import numpy as np

ROOT = Path(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
DATA = ROOT / "dashboard" / "data"


def main():
    for city in (sys.argv[1:] or ["Vienna", "HongKong"]):
        full_ids = (DATA / f"panos_{city}_ids.full.txt").read_text().split("\n")
        full_xy = np.fromfile(DATA / f"panos_{city}_xy.full.bin", np.float32).reshape(-1, 2)
        kept = set((DATA / f"panos_{city}_ids.txt").read_text().split("\n"))
        drop_idx = [i for i, pid in enumerate(full_ids) if pid not in kept]
        full_xy[drop_idx].tofile(DATA / f"panos_{city}_dropped_xy.bin")
        open(DATA / f"panos_{city}_dropped_ids.txt", "w").write("\n".join(full_ids[i] for i in drop_idx))
        print(f"[dropped] {city}: {len(drop_idx):,} excluded / {len(full_ids):,} full")


if __name__ == "__main__":
    main()
