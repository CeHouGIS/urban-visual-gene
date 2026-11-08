"""Match all 300 study city centres to GHSL urban-centre polygons."""
from __future__ import annotations

import csv
import json
import re
import shutil
import unicodedata
import urllib.request
import zipfile
from pathlib import Path

import geopandas as gpd
from pyproj import Transformer
from shapely.geometry import Point
from shapely.strtree import STRtree

from render_google_coverage import CITY_MAP
from study_area import CITY_CENTERS


ROOT = Path(__file__).resolve().parent
DATA = ROOT / "results/ghsl"
SOURCE = DATA / "GHS_UCDB_GLOBE_R2024A.gpkg"
ZIP_SOURCE = DATA / "GHS_UCDB_GLOBE_R2024A_V1_2.zip"
SOURCE_URL = ("https://jeodpp.jrc.ec.europa.eu/ftp/jrc-opendata/GHSL/"
              "GHS_UCDB_GLOBE_R2024A/GHS_UCDB_GLOBE_R2024A/V1-2/"
              "GHS_UCDB_GLOBE_R2024A_V1_2.zip")
BOUNDARIES = DATA / "matched_city_boundaries.gpkg"
AUDIT = DATA / "city_boundary_matches.csv"
LAYER = "GHSL_UCDB_THEME_GENERAL_CHARACTERISTICS_GLOBE_R2024A"
FIELDS = ["city", "status", "ghsl_id", "ghsl_name", "ghsl_country_un",
          "ghsl_country_gad", "ghsl_area_km2", "ghsl_population_2025",
          "match_method", "center_lat", "center_lon", "center_boundary_distance_km",
          "nearest_ghsl_name", "nearest_boundary_distance_km", "boundary_area_km2"]


def normalize_name(value: str) -> str:
    folded = unicodedata.normalize("NFKD", value).encode("ascii", "ignore").decode().casefold()
    return re.sub(r"[^a-z0-9]", "", folded)


def ensure_source() -> None:
    if SOURCE.is_file():
        return
    DATA.mkdir(parents=True, exist_ok=True)
    if not ZIP_SOURCE.is_file():
        urllib.request.urlretrieve(SOURCE_URL, ZIP_SOURCE)
    with zipfile.ZipFile(ZIP_SOURCE) as archive:
        member = next((name for name in archive.namelist()
                       if name.endswith("GHS_UCDB_GLOBE_R2024A.gpkg")), None)
        if member is None:
            raise RuntimeError(f"GHSL package has no expected GeoPackage: {ZIP_SOURCE}")
        with archive.open(member) as source, SOURCE.open("wb") as destination:
            shutil.copyfileobj(source, destination)


def run() -> None:
    ensure_source()
    city_map = json.loads(CITY_MAP.read_text())
    with CITY_CENTERS.open(newline="", encoding="utf-8") as stream:
        centers = {f"{row['country']}/{row['city']}": row
                   for row in csv.DictReader(stream)}
    cities = sorted(set(city_map.values()))
    if not set(cities).issubset(centers):
        raise ValueError(f"Missing GeoNames centre(s): {sorted(set(cities) - set(centers))[:5]}")

    boundaries = gpd.read_file(SOURCE, layer=LAYER)
    boundaries = boundaries[["ID_UC_G0", "GC_UCN_MAI_2025", "GC_CNT_UNN_2025",
                             "GC_CNT_GAD_2025", "GC_UCA_KM2_2025", "GC_POP_TOT_2025",
                             "geometry"]]
    tree = STRtree(boundaries.geometry.to_numpy())
    to_source = Transformer.from_crs("EPSG:4326", boundaries.crs, always_xy=True)
    matched = []
    audit = []
    for city in cities:
        ref = centers[city]
        lat, lon = float(ref["center_lat"]), float(ref["center_lon"])
        x, y = to_source.transform(lon, lat)
        point = Point(x, y)
        candidates = tree.query(point, predicate="intersects")
        candidates = [int(index) for index in candidates
                      if boundaries.geometry.iloc[int(index)].covers(point)]
        match_method = "Study city centre covered by GHSL UCDB polygon"
        if not candidates:
            nearest = tree.query_nearest(point, all_matches=True, return_distance=True)
            nearest_ids, nearest_distances = nearest
            nearest_pairs = sorted(zip(map(int, nearest_ids), map(float, nearest_distances)),
                                   key=lambda pair: pair[1])
            nearest_index, nearest_distance = nearest_pairs[0]
            city_name = city.rsplit("/", 1)[-1]
            same_name = [index for index in range(len(boundaries))
                         if normalize_name(str(boundaries.iloc[index]["GC_UCN_MAI_2025"]))
                         == normalize_name(city_name)]
            nearby_same_name = [index for index in same_name
                                if point.distance(boundaries.geometry.iloc[index]) <= 10_000]
            if nearby_same_name:
                candidates = [min(nearby_same_name,
                                 key=lambda index: point.distance(boundaries.geometry.iloc[index]))]
                match_method = "exact normalized name and centre within 10 km of GHSL UCDB polygon"
                index = candidates[0]
                nearest_index = index
                nearest_distance = point.distance(boundaries.geometry.iloc[index])
        else:
            nearest_index, nearest_distance = candidates[0], 0.0
        if len(candidates) == 1:
            index = candidates[0]
            source = boundaries.iloc[index]
            geometry = source.geometry
            area_km2 = float(geometry.area) / 1_000_000
            row = {
                "city": city, "status": "matched", "ghsl_id": source["ID_UC_G0"],
                "ghsl_name": source["GC_UCN_MAI_2025"],
                "ghsl_country_un": source["GC_CNT_UNN_2025"],
                "ghsl_country_gad": source["GC_CNT_GAD_2025"],
                "ghsl_area_km2": source["GC_UCA_KM2_2025"],
                "ghsl_population_2025": source["GC_POP_TOT_2025"],
                "match_method": match_method,
                "center_lat": lat, "center_lon": lon,
                "center_boundary_distance_km": round(float(nearest_distance) / 1000, 3),
                "nearest_ghsl_name": source["GC_UCN_MAI_2025"],
                "nearest_boundary_distance_km": round(float(nearest_distance) / 1000, 3),
                "boundary_area_km2": round(area_km2, 3),
            }
            matched.append({"city": city, "ghsl_id": row["ghsl_id"],
                            "ghsl_name": row["ghsl_name"],
                            "ghsl_country_un": row["ghsl_country_un"],
                            "ghsl_country_gad": row["ghsl_country_gad"],
                            "ghsl_area_km2": row["ghsl_area_km2"],
                            "ghsl_population_2025": row["ghsl_population_2025"],
                            "match_method": row["match_method"],
                            "center_lat": lat, "center_lon": lon,
                            "center_boundary_distance_km": row["center_boundary_distance_km"],
                            "boundary_area_km2": area_km2,
                            "geometry": geometry})
        else:
            row = {"city": city, "status": "unmatched" if not candidates else "ambiguous",
                   "ghsl_id": "", "ghsl_name": "", "ghsl_country_un": "",
                   "ghsl_country_gad": "", "ghsl_area_km2": "",
                   "ghsl_population_2025": "", "match_method": "",
                   "center_lat": lat, "center_lon": lon,
                   "center_boundary_distance_km": "",
                   "nearest_ghsl_name": boundaries.iloc[int(nearest_index)]["GC_UCN_MAI_2025"],
                   "nearest_boundary_distance_km": round(float(nearest_distance) / 1000, 3),
                   "boundary_area_km2": ""}
        audit.append(row)

    DATA.mkdir(parents=True, exist_ok=True)
    with AUDIT.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=FIELDS, lineterminator="\n")
        writer.writeheader()
        writer.writerows(audit)
    if matched:
        gpd.GeoDataFrame(matched, geometry="geometry", crs=boundaries.crs).to_file(
            BOUNDARIES, layer="city_boundaries", driver="GPKG")
    counts = {}
    for row in audit:
        counts[row["status"]] = counts.get(row["status"], 0) + 1
    print(f"GHSL GHS-UCDB GLOBE R2024A V1.2: {counts}")
    print(AUDIT)
    print(BOUNDARIES)


if __name__ == "__main__":
    run()
