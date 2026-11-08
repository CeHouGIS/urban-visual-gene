"""Recover geocoordinates for on-disk panoramas omitted from city parquet files.

This reads source city pickles, then joins their coordinate-free panoids to the
global metadata database once. It never infers a location from a filename.
"""
from __future__ import annotations

import csv
import json
import sys
from pathlib import Path

import duckdb
import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq


ROOT = Path(__file__).resolve().parents[2]
CITY_MAP = ROOT / "experiments/global307_baseline_w1024_k8/configs/final_global_300_google_city_map.json"
METADATA = Path("/host/root/mnt/nas/huangyj/GoogleSV/metadata")
GLOBAL_DB = METADATA / "panoids.duckdb"
DISK_COUNTS = Path(__file__).resolve().parent / "results/google_actual_images.csv"
OUTPUT = Path(__file__).resolve().parent / "results/google_50m_coverage/recovered"


def load_legacy_pickle(path: Path) -> pd.DataFrame:
    # These source pickles were written with NumPy 2; the current environment
    # has NumPy 1 and the array internals were renamed between versions.
    sys.modules.setdefault("numpy._core", np.core)
    sys.modules.setdefault("numpy._core.numeric", np.core.numeric)
    return pd.read_pickle(path)


def valid_locations(frame: pd.DataFrame) -> pd.DataFrame:
    lat = pd.to_numeric(frame["lat"], errors="coerce")
    lon = pd.to_numeric(frame["lon"], errors="coerce")
    valid = frame["panoid"].notna() & lat.between(-90, 90) & lon.between(-180, 180)
    return pd.DataFrame({"panoid": frame.loc[valid, "panoid"].astype(str),
                         "lat": lat.loc[valid].astype(float),
                         "lon": lon.loc[valid].astype(float)}).drop_duplicates("panoid")


def prepare_city(connection: duckdb.DuckDBPyConnection, city: str,
                 disk_count: int, output: Path):
    source = METADATA / city / "panoids_all.p"
    if not source.is_file():
        return {"city": city, "source": "no_source_pickle", "source_available": 0,
                "source_geolocated": 0, "global_candidates": 0}
    frame = load_legacy_pickle(source)
    flag = next((name for name in ("exists", "download", "marker")
                 if name in frame.columns), None)
    if flag is None or not {"panoid", "lat", "lon"}.issubset(frame.columns):
        raise ValueError(f"Unexpected source pickle schema: {source}")
    frame = frame.loc[frame[flag].eq(True), ["panoid", "lat", "lon"]]
    frame = frame.loc[frame["panoid"].notna()].drop_duplicates("panoid")
    located = valid_locations(frame)
    supplement = output / "source" / (city.replace("/", "__") + ".parquet")
    supplement.parent.mkdir(parents=True, exist_ok=True)
    pq.write_table(pa.Table.from_pandas(located, preserve_index=False), supplement,
                   compression="zstd")
    source_valid = set(located["panoid"])
    missing = frame.loc[~frame["panoid"].isin(source_valid), ["panoid"]]
    if not missing.empty:
        connection.register("city_missing", missing)
        connection.execute("INSERT INTO missing SELECT ?, panoid FROM city_missing", [city])
        connection.unregister("city_missing")
    connection.execute("INSERT INTO completed VALUES (?, ?, ?, ?, ?)",
                       [city, len(frame), len(located), len(missing), disk_count])
    return {"city": city, "source": "pickle", "source_available": len(frame),
            "source_geolocated": len(located), "global_candidates": len(missing)}


def run(output: Path = OUTPUT):
    output.mkdir(parents=True, exist_ok=True)
    cities = sorted(set(json.loads(CITY_MAP.read_text()).values()))
    with DISK_COUNTS.open(newline="", encoding="utf-8") as stream:
        disk = {row["city"]: int(row["disk_panoids_any_image"]) for row in csv.DictReader(stream)}
    if set(disk) != set(cities):
        raise ValueError("City map and disk inventory differ")
    connection = duckdb.connect(str(output / "staging.duckdb"))
    connection.execute("PRAGMA disable_progress_bar")
    connection.execute("CREATE TABLE IF NOT EXISTS missing(city VARCHAR, panoid VARCHAR)")
    connection.execute("CREATE TABLE IF NOT EXISTS completed(city VARCHAR PRIMARY KEY, "
                       "source_available BIGINT, source_geolocated BIGINT, "
                       "global_candidates BIGINT, disk_panoids BIGINT)")
    done = {row[0] for row in connection.execute("SELECT city FROM completed").fetchall()}
    for index, city in enumerate(cities, 1):
        if city in done:
            continue
        connection.begin()
        try:
            record = prepare_city(connection, city, disk[city], output)
            connection.commit()
        except BaseException:
            connection.rollback()
            raise
        print(f"source {index}/{len(cities)} {city}: {record['source_geolocated']:,} "
              f"located, {record['global_candidates']:,} to query", flush=True)
    if (output / "global").exists():
        print("Global coordinate recovery already exists", flush=True)
        return
    connection.execute(f"ATTACH '{GLOBAL_DB}' AS global_source (READ_ONLY)")
    destination = output / "global"
    print("Joining coordinate-free source IDs to global metadata", flush=True)
    connection.execute("""
        COPY (
          SELECT DISTINCT missing.city, missing.panoid, metadata.lat, metadata.lon
          FROM missing JOIN global_source.metadata AS metadata USING (panoid)
          WHERE metadata.lat BETWEEN -90 AND 90
            AND metadata.lon BETWEEN -180 AND 180
        ) TO ? (FORMAT PARQUET, COMPRESSION ZSTD, PARTITION_BY (city))
    """, [str(destination)])
    print(destination, flush=True)


if __name__ == "__main__":
    run()
