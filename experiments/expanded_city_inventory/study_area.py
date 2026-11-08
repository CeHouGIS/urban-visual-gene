"""Shared GHSL urban-centre boundaries for the 300-city coverage atlas."""
from __future__ import annotations

import csv
from functools import lru_cache
from pathlib import Path

import geopandas as gpd
from pyproj import CRS, Transformer
from shapely.ops import transform


ROOT = Path(__file__).resolve().parents[2]
CITY_CENTERS = (ROOT / "experiments/global307_baseline_w1024_k8/configs"
                / "final_global_307_cities.csv")
GHSL_DATA = Path(__file__).resolve().parent / "results/ghsl"
GHSL_BOUNDARIES = GHSL_DATA / "matched_city_boundaries.gpkg"
GHSL_MATCH_AUDIT = GHSL_DATA / "city_boundary_matches.csv"
GHSL_VERSION = "GHS-UCDB GLOBE R2024A V1.2"
BOUNDARY_METHOD = f"GHSL {GHSL_VERSION} urban-centre boundary"


@lru_cache(maxsize=1)
def centers() -> dict[str, tuple[float, float]]:
    with CITY_CENTERS.open(newline="", encoding="utf-8") as stream:
        return {f"{row['country']}/{row['city']}":
                (float(row["center_lat"]), float(row["center_lon"]))
                for row in csv.DictReader(stream)}


def center_for(city: str) -> tuple[float, float]:
    try:
        return centers()[city]
    except KeyError as error:
        raise ValueError(f"No GeoNames center configured for {city}") from error


@lru_cache(maxsize=1)
def boundary_table() -> gpd.GeoDataFrame:
    return gpd.read_file(GHSL_BOUNDARIES, layer="city_boundaries").set_index("city")


def boundary_info(city: str) -> dict | None:
    table = boundary_table()
    if city not in table.index:
        return None
    return table.loc[city].drop(labels="geometry").to_dict()


def boundary_for(city: str, target_crs="EPSG:4326"):
    table = boundary_table()
    if city not in table.index:
        return None
    geometry = table.loc[city].geometry
    target = CRS.from_user_input(target_crs)
    if target == CRS.from_user_input(table.crs):
        return geometry
    transformer = Transformer.from_crs(table.crs, target, always_xy=True)
    return transform(transformer.transform, geometry)


def boundary_bbox_wgs84(city: str) -> tuple[float, float, float, float] | None:
    geometry = boundary_for(city, "EPSG:4326")
    return tuple(geometry.bounds) if geometry is not None else None
