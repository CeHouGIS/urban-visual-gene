"""Classify 50 m road sections by nearby geolocated Google panorama points.

Roads are Overture transportation segments within the observed panorama extent.
The road network is a separate dataset from the imagery, and the classification
uses only panorama IDs with known coordinates. Every output city is checkpointed.
"""
from __future__ import annotations

import argparse
import csv
import math
import subprocess
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq
from PIL import Image, ImageDraw
from pyproj import Transformer
from scipy.spatial import cKDTree
from shapely import wkb
from shapely.geometry import LineString, MultiLineString, box
from shapely.ops import transform

from render_google_coverage import (CITY_MAP, DISK_COUNTS, IMAGE_H, IMAGE_W,
                                    MAP_BOX, RECOVERY, combined_locations,
                                    font, nice_distance, projection_for)
import json


RELEASE = "2026-10-21.0"
OUTPUT = Path(__file__).resolve().parent / "results/google_road_coverage"
SECTION_M = 50.0
SEARCH_RADIUS_M = 50.0
BOUNDARY_PAD_M = 150.0
DRIVEABLE = {
    "motorway", "motorway_link", "trunk", "trunk_link", "primary",
    "primary_link", "secondary", "secondary_link", "tertiary",
    "tertiary_link", "residential", "unclassified", "service",
    "living_street", "track", "road",
}
FIELDS = ["city", "status", "road_source", "located_panoids", "disk_panoids",
          "road_segments", "sections", "covered_sections", "uncovered_sections",
          "covered_road_km", "uncovered_road_km", "covered_road_percent",
          "section_m", "radius_m", "image"]


def bbox_from_points(lat, lon):
    # 150 m pad avoids cutting roads exactly at the outermost panorama point.
    midlat = float(np.median(lat))
    dy = BOUNDARY_PAD_M / 111_320
    dx = BOUNDARY_PAD_M / (111_320 * max(0.1, math.cos(math.radians(midlat))))
    return (float(lon.min() - dx), float(lat.min() - dy),
            float(lon.max() + dx), float(lat.max() + dy))


def download_roads(city: str, bbox, output: Path):
    target = output / "overture" / (city.replace("/", "__") + ".parquet")
    if target.is_file() and target.stat().st_size:
        try:
            pq.ParquetFile(target)
            return target
        except Exception:
            target.unlink()
    target.parent.mkdir(parents=True, exist_ok=True)
    command = ["overturemaps", "download", "--bbox=" + ",".join(f"{n:.7f}" for n in bbox),
               "-t", "segment", "-f", "geoparquet", "-r", RELEASE,
               "-o", str(target), "--request_timeout", "240"]
    for attempt in range(3):
        result = subprocess.run(command, capture_output=True, text=True, timeout=900)
        if result.returncode == 0 and target.is_file() and target.stat().st_size:
            return target
        message = (result.stderr or result.stdout)[-800:]
        if attempt == 2 or not any(key in message.lower() for key in ("429", "rate limit", "too many requests")):
            raise RuntimeError(f"Overture download failed for {city}: {message}")
        time.sleep(900)
    raise AssertionError("unreachable")


def split_line(line: LineString):
    """Yield straight road sections no longer than 50 m, preserving vertices."""
    coords = np.asarray(line.coords, dtype=np.float64)
    for p, q in zip(coords[:-1], coords[1:]):
        length = float(np.linalg.norm(q[:2] - p[:2]))
        if length <= 0:
            continue
        steps = math.ceil(length / SECTION_M)
        fractions = np.linspace(0, 1, steps + 1)
        points = p[:2] + fractions[:, None] * (q[:2] - p[:2])
        for start, end in zip(points[:-1], points[1:]):
            yield start, end


def geometry_lines(geometry):
    if isinstance(geometry, LineString):
        yield geometry
    elif isinstance(geometry, MultiLineString):
        yield from geometry.geoms


def classify_geometry(lines, projector, clip, tree, draw_red, draw_green, pixel):
    covered_m = uncovered_m = 0.0
    covered_sections = uncovered_sections = 0
    road_segments = 0
    for shape in lines:
        if shape.is_empty:
            continue
        projected = transform(projector.transform, shape)
        clipped = projected.intersection(clip)
        for line in geometry_lines(clipped):
            starts = []
            ends = []
            for start, end in split_line(line):
                starts.append(start)
                ends.append(end)
            if not starts:
                continue
            road_segments += 1
            starts = np.asarray(starts)
            ends = np.asarray(ends)
            midpoints = (starts + ends) / 2
            distances = tree.query(midpoints, k=1, workers=1)[0]
            lengths = np.linalg.norm(ends - starts, axis=1)
            matched = distances <= SEARCH_RADIUS_M
            covered_m += float(lengths[matched].sum())
            uncovered_m += float(lengths[~matched].sum())
            covered_sections += int(matched.sum())
            uncovered_sections += int((~matched).sum())
            for start, end, has_image in zip(starts, ends, matched):
                coords = (*pixel(start), *pixel(end))
                (draw_green if has_image else draw_red).line(coords, fill="#218B5A" if has_image else "#D15559", width=2)
    return road_segments, covered_sections, uncovered_sections, covered_m, uncovered_m


def render_roads(city: str, path: Path, lat: np.ndarray, lon: np.ndarray,
                 disk_count: int, network: Path):
    epsg = projection_for(lat, lon)
    projector = Transformer.from_crs("EPSG:4326", f"EPSG:{epsg}", always_xy=True)
    x, y = projector.transform(lon, lat)
    tree = cKDTree(np.column_stack([x, y]))
    xmin, ymin, xmax, ymax = (float(x.min() - BOUNDARY_PAD_M), float(y.min() - BOUNDARY_PAD_M),
                              float(x.max() + BOUNDARY_PAD_M), float(y.max() + BOUNDARY_PAD_M))
    clip = box(xmin, ymin, xmax, ymax)
    canvas = Image.new("RGB", (IMAGE_W, IMAGE_H), "#fff")
    green = Image.new("RGBA", (IMAGE_W, IMAGE_H))
    base = ImageDraw.Draw(canvas)
    base.text((65, 30), city.replace("/", "  /  "), font=font(30, True), fill="#253039")
    base.text((65, 77), f"50 m road sections  /  {len(lat):,} located panoids", font=font(18), fill="#54646D")
    left, top, right, bottom = MAP_BOX
    base.rectangle(MAP_BOX, fill="#F3F5F4", outline="#D5DEDE", width=2)
    scale = min((right - left - 30) / (xmax - xmin), (bottom - top - 30) / (ymax - ymin))
    map_w = (xmax - xmin) * scale
    map_h = (ymax - ymin) * scale
    px0 = left + (right - left - map_w) / 2
    py0 = top + (bottom - top - map_h) / 2
    def pixel(point):
        return (round(px0 + (point[0] - xmin) * scale),
                round(py0 + (ymax - point[1]) * scale))
    red_draw, green_draw = ImageDraw.Draw(canvas), ImageDraw.Draw(green)
    totals = np.zeros(5, dtype=np.float64)
    parquet = pq.ParquetFile(network)
    for batch in parquet.iter_batches(batch_size=2048, columns=["subtype", "class", "geometry"]):
        data = batch.to_pydict()
        lines = (wkb.loads(raw) for subtype, road_class, raw in
                 zip(data["subtype"], data["class"], data["geometry"])
                 if subtype == "road" and road_class in DRIVEABLE and raw)
        totals += classify_geometry(lines, projector, clip, tree, red_draw, green_draw, pixel)
    canvas = Image.alpha_composite(canvas.convert("RGBA"), green).convert("RGB")
    draw = ImageDraw.Draw(canvas)
    scale_m = nice_distance((xmax - xmin) * 0.18)
    px = round(scale_m * scale)
    sy = bottom - 35
    draw.line((left + 25, sy, left + 25 + px, sy), fill="#303B42", width=3)
    draw.text((left + 25, sy - 27), f"{scale_m/1000:g} km" if scale_m >= 1000 else f"{scale_m} m",
              font=font(16), fill="#303B42")
    draw.line((65, 779, 85, 779), fill="#218B5A", width=5)
    draw.text((95, 769), "within 50 m of image", font=font(17), fill="#35444A")
    draw.line((395, 779, 415, 779), fill="#D15559", width=5)
    draw.text((425, 769), "no located image within 50 m", font=font(17), fill="#35444A")
    covered_m, uncovered_m = totals[3:]
    share = covered_m / (covered_m + uncovered_m) if covered_m + uncovered_m else 0
    draw.text((65, 811), f"Mapped road length: {share:.1%} covered  |  Overture Maps {RELEASE}  |  Study-area extent",
              font=font(14), fill="#64737A")
    path.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(path, optimize=True)
    return {"road_segments": int(totals[0]), "sections": int(totals[1] + totals[2]),
            "covered_sections": int(totals[1]), "uncovered_sections": int(totals[2]),
            "covered_road_km": round(covered_m / 1000, 3),
            "uncovered_road_km": round(uncovered_m / 1000, 3),
            "covered_road_percent": round(100 * share, 2)}


def process_city(city: str, disk_row: dict, output: Path):
    count = int(disk_row["disk_panoids_any_image"])
    lat, lon = combined_locations(city, count, RECOVERY)
    image = "maps/" + city.replace("/", "__") + ".png"
    row = {key: "" for key in FIELDS}
    row.update({"city": city, "road_source": f"Overture Maps {RELEASE}",
                "located_panoids": len(lat), "disk_panoids": count,
                "section_m": SECTION_M, "radius_m": SEARCH_RADIUS_M,
                "image": image})
    if len(lat) / count < .01:
        row["status"] = "insufficient_coordinates"
        (output / image).parent.mkdir(parents=True, exist_ok=True)
        Image.new("RGB", (IMAGE_W, IMAGE_H), "#F3F5F4").save(output / image)
    else:
        source = download_roads(city, bbox_from_points(lat, lon), output)
        row.update(render_roads(city, output / image, lat, lon, count, source))
        row["status"] = "mapped" if row["sections"] else "no_roads"
    return row


def run(output: Path = OUTPUT, only: str | None = None, workers: int = 3):
    if workers < 1:
        raise ValueError("workers must be positive")
    cities = sorted(set(json.loads(CITY_MAP.read_text()).values()))
    if only:
        if only not in cities:
            raise ValueError(only)
        cities = [only]
    with DISK_COUNTS.open(newline="", encoding="utf-8") as stream:
        disk = {r["city"]: r for r in csv.DictReader(stream)}
    output.mkdir(parents=True, exist_ok=True)
    progress = output / "progress.csv"
    prior = {}
    if progress.exists():
        with progress.open(newline="", encoding="utf-8") as stream:
            prior = {r["city"]: r for r in csv.DictReader(stream)}
    remaining = [city for city in cities if city not in prior or not (output / prior[city]["image"]).is_file()]
    failed = []
    with progress.open("a", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=FIELDS)
        if not prior:
            writer.writeheader()
        with ProcessPoolExecutor(max_workers=workers) as pool:
            futures = {pool.submit(process_city, city, disk[city], output): city
                       for city in remaining}
            for future in as_completed(futures):
                city = futures[future]
                try:
                    row = future.result()
                except Exception as exc:
                    failed.append(city)
                    print(f"FAILED {city}: {exc}", flush=True)
                    continue
                writer.writerow(row)
                stream.flush()
                prior[city] = row
                print(f"{len(prior)}/{len(set(json.loads(CITY_MAP.read_text()).values()))} "
                      f"{city}: {row['status']}, {row['sections']} sections", flush=True)
    with (output / "city_road_coverage.csv").open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=FIELDS)
        writer.writeheader()
        writer.writerows(sorted(prior.values(), key=lambda r: r["city"]))
    if failed:
        raise RuntimeError(f"{len(failed)} cities failed and can be retried: {failed}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=OUTPUT)
    parser.add_argument("--city")
    parser.add_argument("--workers", type=int, default=3)
    args = parser.parse_args()
    run(args.output, args.city, args.workers)
