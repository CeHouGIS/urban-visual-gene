#!/usr/bin/env python3
"""Flag node<->nearest-pano binding lines that pass through a building footprint
(occluded sight-lines). Writes <city>_bindblock.bin (Uint8 per node: 1=blocked).
The dashboard then drops blocked lines.

    /opt/conda/bin/python3 scripts/filter_binding_buildings.py <City>
"""
import os, sys
from pathlib import Path
import numpy as np
import geopandas as gpd
import shapely

ROOT = Path(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
DATA = ROOT / "dashboard" / "data"


def main():
    city = sys.argv[1]
    pos = np.fromfile(DATA / f"{city}_pos.bin", np.float32).reshape(-1, 2)      # node lon,lat
    bind = np.fromfile(DATA / f"{city}_bind.bin", np.float32).reshape(-1, 3)    # panoLon,panoLat,dist
    N = len(pos); assert len(bind) == N
    blocked = np.zeros(N, np.uint8)

    cand = np.where(bind[:, 2] <= 100)[0]            # only drawn lines (≤100m)
    print(f"[bld-filter] {city}: {N:,} nodes, {len(cand):,} drawn lines", flush=True)

    bld = gpd.read_parquet(DATA.parent.parent / "data/buildings" / city / "buildings.parquet")
    geoms = bld.geometry.values
    tree = shapely.STRtree(geoms)
    print(f"[bld-filter] {city}: {len(geoms):,} buildings indexed", flush=True)

    # segments node -> pano for candidate nodes
    starts = pos[cand]                                # (M,2)
    ends = bind[cand, :2]                             # (M,2)
    lines = shapely.linestrings(np.stack([starts, ends], axis=1))   # (M,) LineString

    # which lines intersect any building
    pairs = tree.query(lines, predicate="intersects")  # (2, K): [line_idx, building_idx]
    hit_local = np.unique(pairs[0])
    blocked[cand[hit_local]] = 1
    blocked.tofile(DATA / f"{city}_bindblock.bin")
    nb = int(blocked.sum())
    print(f"[bld-filter] {city}: {nb:,}/{len(cand):,} lines cross a building "
          f"({nb/max(1,len(cand))*100:.1f}%) -> dropped; {len(cand)-nb:,} clear", flush=True)


if __name__ == "__main__":
    main()
