"""Shared city data loaders (torch-free).

Resolves panoramas, road network and output dir per city, preferring the local
data mirror (./data) over the NAS source. Kept free of torch and geopandas at
import time so it is safe to use from any stage runner.
"""
from __future__ import annotations

import sqlite3
from pathlib import Path

import pandas as pd

# Prefer the local data mirror (./data, populated by scripts/copy_data.py);
# fall back to the NAS source if the local copy is absent.
_LOCAL = Path(__file__).resolve().parent.parent.parent / "data/SVIs/GSV"
_NAS = Path("/workplace/nas_data/houce/urban_visual_gene/SVIs/GSV")
DATA_ROOT = _LOCAL if _LOCAL.exists() else _NAS

MAX_PANOS_SEED = 42

CITY_CONFIG = {
    "Vienna": {
        "panoids":  "metadata/Austria/Vienna/panoids.parquet",
        "roads":    "metadata/Austria/Vienna/edges.shp",
        "images":   "images/Austria/Vienna",
        "path_style": "vienna",
        "out_dir":  "outputs/Austria/Vienna",
    },
    "HongKong": {
        "db":       "metadata/China/HongKong/meta/China_HongKong.db",
        "roads":    "metadata/China/HongKong/road/HongKong_China_roads.geojson",
        "images":   "images/China/HongKong",
        "path_style": "hongkong",
        "out_dir":  "outputs/China/HongKong",
    },
    "Singapore": {
        "db":       "metadata/Singapore/Singapore/meta/Singapore_Singapore.db",
        "roads":    "metadata/Singapore/Singapore/road/Singapore_Singapore_roads.geojson",
        "images":   "images/Singapore/Singapore",
        "path_style": "hongkong",
        "out_dir":  "outputs/Singapore/Singapore",
    },
    "Amsterdam": {
        "db":       "metadata/Netherlands/Amsterdam/meta/Netherlands_Amsterdam.db",
        "roads":    "metadata/Netherlands/Amsterdam/road/Amsterdam_Netherlands_roads.geojson",
        "images":   "images/Netherlands/Amsterdam",
        "path_style": "hongkong",
        "out_dir":  "outputs/Netherlands/Amsterdam",
    },
    "CapeTown": {
        "db":       "metadata/SouthAfrica/CapeTown/meta/SouthAfrica_CapeTown.db",
        "roads":    "metadata/SouthAfrica/CapeTown/road/CapeTown_SouthAfrica_roads.geojson",
        "images":   "images/SouthAfrica/CapeTown",
        "path_style": "hongkong",
        "out_dir":  "outputs/SouthAfrica/CapeTown",
    },
}


def out_dir(city: str) -> Path:
    return Path(CITY_CONFIG[city]["out_dir"])


def img_root(city: str) -> Path:
    return DATA_ROOT / CITY_CONFIG[city]["images"]


def img_root_fallback(city: str) -> Path | None:
    """NAS image root, used when the local mirror is incomplete (e.g. HK is only
    partially copied locally). None if we are already reading from NAS."""
    if DATA_ROOT == _NAS:
        return None
    nas_imgs = _NAS / CITY_CONFIG[city]["images"]
    return nas_imgs if nas_imgs.exists() else None


def path_style(city: str) -> str:
    return CITY_CONFIG[city]["path_style"]


def load_panos(city: str, max_panos: int | None,
               sampling: str = "random") -> pd.DataFrame:
    """Return pano metadata DataFrame (pano_id, lat, lon, city).

    sampling="random": uniform random subsample (default, reproducible seed).
    sampling="greedy": keep the panos pre-selected by `scripts.sampling.select_panos`
    (max road-network coverage), in greedy rank order — far fewer downstream
    nodes need interpolation. Falls back to random if the selection is missing.
    """
    cfg = CITY_CONFIG[city]
    if city == "Vienna":
        df = pd.read_parquet(DATA_ROOT / cfg["panoids"]).rename(columns={"panoid": "pano_id"})
    else:
        con = sqlite3.connect(DATA_ROOT / cfg["db"])
        df = pd.read_sql("SELECT panoid as pano_id, lat, lon FROM gsv WHERE download=1", con)
        con.close()
    df["city"] = city
    if sampling == "greedy":
        sel_path = out_dir(city) / "selected_pano_ids.parquet"
        if sel_path.exists():
            sel = pd.read_parquet(sel_path).sort_values("rank")
            if max_panos:
                sel = sel.head(max_panos)
            order = {pid: r for r, pid in enumerate(sel["pano_id"])}
            df = df[df["pano_id"].isin(order)].copy()
            df["_rank"] = df["pano_id"].map(order)
            return df.sort_values("_rank").drop(columns="_rank").reset_index(drop=True)
        print(f"[load_panos] no greedy selection at {sel_path}; "
              f"falling back to random")
    if max_panos:
        df = df.sample(max_panos, random_state=MAX_PANOS_SEED).reset_index(drop=True)
    return df


def load_roads(city: str):
    """Return the road-network GeoDataFrame (geopandas imported lazily)."""
    import geopandas as gpd
    cfg = CITY_CONFIG[city]
    roads = gpd.read_file(DATA_ROOT / cfg["roads"]).rename(columns={"osmid": "road_id"})
    roads["road_id"] = roads["road_id"].astype(str)
    return roads
