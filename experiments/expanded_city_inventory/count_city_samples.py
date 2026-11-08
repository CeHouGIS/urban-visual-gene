"""Inventory Google four-view candidates and Baidu panorama coverage by city."""
from __future__ import annotations

import argparse
import csv
import json
import math
import os
from collections import defaultdict
from pathlib import Path

import pyarrow.parquet as pq


ROOT = Path(__file__).resolve().parents[2]
GOOGLE_MAP = (ROOT / "experiments/global307_baseline_w1024_k8/configs"
              / "final_global_300_google_city_map.json")
GOOGLE_METADATA = Path("/host/root/mnt/nas/huangyj/GoogleSV/metadata")
BAIDU_ROOT = Path("/host/root/mnt/nas/liyong/baidusv_china")
OUTPUT = Path(__file__).resolve().parent / "results"
NON_MAINLAND = {"Aomen", "Macau", "Xianggang", "HongKong", "Taiwan"}


def google_counts():
    city_map = json.loads(GOOGLE_MAP.read_text())
    for index, (name, city) in enumerate(sorted(city_map.items()), 1):
        source = GOOGLE_METADATA / city / "panoids.parquet"
        row = {"platform": "GoogleSV", "city": city, "province": "",
               "raw_records": 0, "eligible_records": 0,
               "unique_eligible_panoids": 0, "disk_image_files": "",
               "unique_disk_panoids": "", "eligible_on_disk": "",
               "count_basis": "metadata availability flag; four images when #images exists"}
        if not source.is_file():
            row["count_basis"] = "missing metadata"
            yield row
            continue
        parquet = pq.ParquetFile(source)
        columns = parquet.schema.names
        flag = next((x for x in ("exists", "download", "marker") if x in columns), None)
        if flag is None:
            raise ValueError(f"No availability flag: {source}")
        has_four = "#images" in columns
        wanted = ["panoid", "lat", "lon", flag]
        if has_four:
            wanted.append("#images")
        ids = set()
        for batch in parquet.iter_batches(batch_size=65536, columns=wanted):
            data = batch.to_pydict()
            count = len(data["panoid"])
            row["raw_records"] += count
            for i in range(count):
                pid, lat, lon = data["panoid"][i], data["lat"][i], data["lon"][i]
                if (data[flag][i] is not True or not pid or lat is None or lon is None
                        or not math.isfinite(lat) or not math.isfinite(lon)
                        or not -90 <= lat <= 90 or not -180 <= lon <= 180
                        or (has_four and (data["#images"][i] or 0) < 4)):
                    continue
                row["eligible_records"] += 1
                ids.add(pid)
        row["unique_eligible_panoids"] = len(ids)
        if not has_four:
            row["count_basis"] += "; four-image presence not independently verified"
        print(f"Google {index}/{len(city_map)} {city}: {len(ids):,}", flush=True)
        yield row


def baidu_index():
    rows = defaultdict(int)
    successes = defaultdict(int)
    ids = defaultdict(set)
    source = BAIDU_ROOT / "h3_samples_nationwide.csv"
    with source.open(newline="", encoding="utf-8-sig") as stream:
        for record in csv.DictReader(stream):
            key = (record["province"].strip(), record["city"].strip())
            if not all(key):
                continue
            rows[key] += 1
            pid = record["panoid"].strip()
            if record["status"] != "成功" or not pid:
                continue
            try:
                lat, lon = float(record["wgs_lat"]), float(record["wgs_lon"])
            except ValueError:
                continue
            if math.isfinite(lat) and math.isfinite(lon) and -90 <= lat <= 90 and -180 <= lon <= 180:
                successes[key] += 1
                ids[key].add(pid)
    return rows, successes, ids


def baidu_counts():
    rows, successes, ids = baidu_index()
    root = BAIDU_ROOT / "images"
    disk_keys = {(p.parent.name, p.name): p for p in root.glob("*/*") if p.is_dir()}
    for index, key in enumerate(sorted(disk_keys.keys() | rows.keys()), 1):
        province, city = key
        unique_disk = set()
        file_count = 0
        matched = set()
        directory = disk_keys.get(key)
        if directory:
            for current, _, files in os.walk(directory):
                for filename in files:
                    if not filename.lower().endswith((".jpg", ".jpeg")):
                        continue
                    file_count += 1
                    pid = Path(filename).stem
                    unique_disk.add(pid)
                    if pid in ids[key]:
                        matched.add(pid)
        print(f"Baidu {index}/{len(disk_keys.keys() | rows.keys())} "
              f"{province}/{city}: {len(unique_disk):,} disk, {len(matched):,} indexed",
              flush=True)
        yield {"platform": "BaiduSV", "city": f"China/{province}/{city}",
               "province": province, "raw_records": rows[key],
               "eligible_records": successes[key],
               "unique_eligible_panoids": len(ids[key]),
               "disk_image_files": file_count,
               "unique_disk_panoids": len(unique_disk),
               "eligible_on_disk": len(matched),
               "count_basis": ("non-mainland; " if province in NON_MAINLAND else "")
               + "one equirectangular file per panoid; indexed intersection has coordinates"}


def write_ranked_tables(destination: Path):
    with destination.open(newline="", encoding="utf-8") as stream:
        rows = list(csv.DictReader(stream))
    groups = [
        ("google_city_counts.csv", [row for row in rows if row["platform"] == "GoogleSV"],
         "unique_eligible_panoids"),
        ("china_mainland_city_counts.csv",
         [row for row in rows if row["platform"] == "BaiduSV"
          and row["province"] not in NON_MAINLAND], "eligible_on_disk"),
    ]
    for filename, group, field in groups:
        ranked = sorted(group, key=lambda row: (-int(row[field]), row["city"]))
        with (destination.parent / filename).open("w", newline="", encoding="utf-8") as stream:
            writer = csv.DictWriter(stream, fieldnames=["rank", "city", "sample_count",
                                                      "count_basis", "province"])
            writer.writeheader()
            for rank, row in enumerate(ranked, 1):
                writer.writerow({"rank": rank, "city": row["city"],
                                 "sample_count": row[field],
                                 "count_basis": row["count_basis"],
                                 "province": row["province"]})


def run(output: Path):
    output.mkdir(parents=True, exist_ok=True)
    fields = ["platform", "city", "province", "raw_records", "eligible_records",
              "unique_eligible_panoids", "disk_image_files", "unique_disk_panoids",
              "eligible_on_disk", "count_basis"]
    destination = output / "city_sample_counts.csv"
    partial = destination.with_suffix(".partial.csv")
    with partial.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        for row in google_counts():
            writer.writerow(row)
        for row in baidu_counts():
            writer.writerow(row)
    partial.replace(destination)
    write_ranked_tables(destination)
    print(destination, flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=OUTPUT)
    run(parser.parse_args().output)
