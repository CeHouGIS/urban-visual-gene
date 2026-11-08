"""Render 50 m coverage footprints for the 300 Google street-view cities.

Only panoids with coordinates and an available-image flag can be mapped. The
per-city inventory keeps the on-disk panoid total alongside the mapped count;
it must not be interpreted as a city-boundary coverage percentage.
"""
from __future__ import annotations

import argparse
import csv
import html
import json
import math
import shutil
from pathlib import Path
from urllib.parse import quote

import numpy as np
import pandas as pd
import pyarrow.parquet as pq
from PIL import Image, ImageDraw, ImageFont
from pyproj import Transformer
from scipy.ndimage import binary_dilation


ROOT = Path(__file__).resolve().parents[2]
CITY_MAP = ROOT / "experiments/global307_baseline_w1024_k8/configs/final_global_300_google_city_map.json"
METADATA = Path("/host/root/mnt/nas/huangyj/GoogleSV/metadata")
DISK_COUNTS = Path(__file__).resolve().parent / "results/google_actual_images.csv"
OUTPUT = Path(__file__).resolve().parent / "results/google_50m_coverage"
RECOVERY = OUTPUT / "recovered"
RADIUS_M = 50
MIN_CELL_M = 10
MAX_RASTER_CELLS = 150_000_000
IMAGE_W, IMAGE_H = 1200, 850
MAP_BOX = (65, 118, 1135, 740)
FIELDS = ["city", "disk_panoids_any_image", "disk_jpg_files", "located_panoids",
          "located_to_disk_ratio", "coverage_km2_estimate", "raster_cell_m",
          "raster_width", "raster_height", "epsg", "status", "image"]


def font(size: int, bold: bool = False):
    name = "DejaVuSans-Bold.ttf" if bold else "DejaVuSans.ttf"
    return ImageFont.truetype(name, size)


def read_locations(path: Path) -> pd.DataFrame:
    parquet = pq.ParquetFile(path)
    columns = parquet.schema.names
    flag = next((name for name in ("exists", "download", "marker") if name in columns), None)
    if flag is None:
        raise ValueError(f"Missing image-availability flag: {path}")
    selected = ["panoid", "lat", "lon", flag]
    if "#images" in columns:
        selected.append("#images")
    frame = parquet.read(columns=selected).to_pandas()
    if frame.empty:
        return pd.DataFrame(columns=["panoid", "lat", "lon"])
    valid = frame[flag].eq(True) & frame["panoid"].notna()
    if "#images" in columns:
        valid &= frame["#images"].fillna(0).ge(4)
    frame = frame.loc[valid, ["panoid", "lat", "lon"]].drop_duplicates("panoid")
    frame["lat"] = pd.to_numeric(frame["lat"], errors="coerce")
    frame["lon"] = pd.to_numeric(frame["lon"], errors="coerce")
    finite = (frame["lat"].between(-90, 90) & frame["lon"].between(-180, 180))
    return frame.loc[finite]


def combined_locations(city: str, disk_count: int, recovery: Path | None):
    sources = [read_locations(METADATA / city / "panoids.parquet")]
    if recovery is not None:
        source = recovery / "source" / (city.replace("/", "__") + ".parquet")
        if source.is_file():
            extra = pq.read_table(source).to_pandas()
            # Some source pickles mark images available when the files are no
            # longer on disk. Do not use such a source to expand the map.
            if len(extra) <= disk_count:
                sources.append(extra)
        global_city = recovery / "global" / f"city={quote(city, safe='')}"
        if global_city.is_dir():
            sources.extend(pq.read_table(part).to_pandas()
                           for part in sorted(global_city.glob("*.parquet")))
    frame = pd.concat(sources, ignore_index=True).drop_duplicates("panoid")
    if len(frame) > disk_count:
        raise ValueError(f"More located IDs than disk panoids in {city}: {len(frame)} > {disk_count}")
    return frame["lat"].to_numpy(dtype=np.float64), frame["lon"].to_numpy(dtype=np.float64)


def projection_for(lat: np.ndarray, lon: np.ndarray) -> int:
    middle_lat, middle_lon = float(np.median(lat)), float(np.median(lon))
    if not -80 <= middle_lat <= 84:
        raise ValueError("UTM is unavailable at this latitude")
    zone = min(60, max(1, math.floor((middle_lon + 180) / 6) + 1))
    return (32600 if middle_lat >= 0 else 32700) + zone


def footprint(x: np.ndarray, y: np.ndarray, radius_m: float = RADIUS_M):
    if len(x) != len(y) or len(x) == 0:
        raise ValueError("Coordinates must be paired and nonempty")
    xmin, xmax = float(np.min(x) - radius_m), float(np.max(x) + radius_m)
    ymin, ymax = float(np.min(y) - radius_m), float(np.max(y) + radius_m)
    cell_m = max(MIN_CELL_M, math.sqrt((xmax - xmin) * (ymax - ymin) / MAX_RASTER_CELLS))
    cell_m = math.ceil(cell_m / 2) * 2
    width = max(1, math.ceil((xmax - xmin) / cell_m))
    height = max(1, math.ceil((ymax - ymin) / cell_m))
    occupied = np.zeros((height, width), dtype=bool)
    col = np.clip(((x - xmin) / cell_m).astype(np.int64), 0, width - 1)
    row = np.clip(((ymax - y) / cell_m).astype(np.int64), 0, height - 1)
    occupied[row, col] = True
    reach = math.ceil(radius_m / cell_m)
    offset = np.arange(-reach, reach + 1)
    dy, dx = np.meshgrid(offset, offset, indexing="ij")
    disk = (dx * cell_m) ** 2 + (dy * cell_m) ** 2 <= radius_m**2
    covered = binary_dilation(occupied, structure=disk)
    area_km2 = float(covered.sum()) * cell_m**2 / 1_000_000
    return covered, (xmin, ymin, xmax, ymax), cell_m, area_km2


def nice_distance(meters: float) -> int:
    magnitude = 10 ** math.floor(math.log10(max(meters, 1)))
    return int(max(v * magnitude for v in (0.1, 0.2, 0.5, 1, 2, 5, 10) if v * magnitude <= meters))


def render_map(city: str, image_path: Path, covered: np.ndarray | None,
               extent: tuple[float, float, float, float] | None,
               located: int, disk: int, cell_m: float | None,
               insufficient: bool = False):
    canvas = Image.new("RGB", (IMAGE_W, IMAGE_H), "#ffffff")
    draw = ImageDraw.Draw(canvas)
    draw.text((65, 32), city.replace("/", "  /  "), fill="#202a32", font=font(30, True))
    draw.text((65, 79), f"{located:,} located panoids  /  {disk:,} on disk",
              fill="#566573", font=font(18))
    left, top, right, bottom = MAP_BOX
    draw.rectangle(MAP_BOX, fill="#f1f4f4", outline="#d4dce0", width=2)
    if covered is None or extent is None:
        title = "Too few coordinates for a city coverage map" if insufficient else "Location coordinates unavailable"
        detail = (f"Only {located / disk:.2%} of on-disk panoids can be located."
                  if insufficient else "No 50 m footprint can be mapped for this city.")
        draw.text((left + 38, top + 46), title,
                  fill="#57636c", font=font(24))
        draw.text((left + 38, top + 86), detail,
                  fill="#687680", font=font(17))
    else:
        xmin, ymin, xmax, ymax = extent
        real_w, real_h = xmax - xmin, ymax - ymin
        fit = min((right - left - 30) / real_w, (bottom - top - 30) / real_h)
        display_w, display_h = max(1, round(real_w * fit)), max(1, round(real_h * fit))
        display_left = left + ((right - left) - display_w) // 2
        display_top = top + ((bottom - top) - display_h) // 2
        mask = Image.fromarray((covered * 255).astype(np.uint8), mode="L")
        mask = mask.resize((display_w, display_h), Image.Resampling.LANCZOS)
        tint = Image.new("RGB", (display_w, display_h), "#247d88")
        canvas.paste(tint, (display_left, display_top), mask)
        draw = ImageDraw.Draw(canvas)
        draw.rectangle((display_left, display_top, display_left + display_w,
                        display_top + display_h), outline="#c2cbd0", width=1)
        scale_m = nice_distance(real_w * 0.18)
        length_px = round(scale_m * fit)
        base_y = bottom - 38
        draw.line((left + 25, base_y, left + 25 + length_px, base_y), fill="#2e3940", width=4)
        draw.line((left + 25, base_y - 5, left + 25, base_y + 5), fill="#2e3940", width=2)
        draw.line((left + 25 + length_px, base_y - 5, left + 25 + length_px, base_y + 5),
                  fill="#2e3940", width=2)
        label = f"{scale_m / 1000:g} km" if scale_m >= 1000 else f"{scale_m} m"
        draw.text((left + 25, base_y - 30), label, fill="#2e3940", font=font(16))
    draw = ImageDraw.Draw(canvas)
    if covered is not None:
        draw.rectangle((65, 778, 85, 798), fill="#247d88")
        draw.text((96, 776), "within 50 m of a located street-view point", fill="#394650", font=font(17))
    draw.text((65, 817), "Projected metric raster; coverage is approximate and is not clipped to a municipal boundary.",
              fill="#687680", font=font(14))
    image_path.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(image_path, optimize=True)


def process_city(city: str, disk: dict[str, str], output: Path, recovery: Path | None):
    disk_count = int(disk["disk_panoids_any_image"])
    lat, lon = combined_locations(city, disk_count, recovery)
    located = len(lat)
    image_name = city.replace("/", "__") + ".png"
    image_path = output / "maps" / image_name
    row = {"city": city, "disk_panoids_any_image": disk["disk_panoids_any_image"],
           "disk_jpg_files": disk["actual_jpg_files"], "located_panoids": located,
           "located_to_disk_ratio": round(located / int(disk["disk_panoids_any_image"]), 6)
           if int(disk["disk_panoids_any_image"]) else 0,
           "coverage_km2_estimate": "", "raster_cell_m": "", "raster_width": "",
           "raster_height": "", "epsg": "", "status": "no_coordinates", "image": f"maps/{image_name}"}
    if not located or located / disk_count < 0.01:
        insufficient = bool(located)
        row["status"] = "insufficient_coordinates" if insufficient else "no_coordinates"
        render_map(city, image_path, None, None, located, disk_count, None,
                   insufficient=insufficient)
        return row
    epsg = projection_for(lat, lon)
    x, y = Transformer.from_crs("EPSG:4326", f"EPSG:{epsg}", always_xy=True).transform(lon, lat)
    finite = np.isfinite(x) & np.isfinite(y)
    x, y = x[finite], y[finite]
    covered, extent, cell_m, area_km2 = footprint(x, y)
    render_map(city, image_path, covered, extent, len(x),
               int(disk["disk_panoids_any_image"]), cell_m)
    row.update({"located_panoids": len(x), "coverage_km2_estimate": round(area_km2, 3),
                "raster_cell_m": cell_m, "raster_width": covered.shape[1],
                "raster_height": covered.shape[0], "epsg": epsg, "status": "mapped"})
    return row


def write_report(rows: list[dict], output: Path, road_report: Path | None = None):
    with (output / "city_coverage.csv").open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=FIELDS, lineterminator="\n")
        writer.writeheader()
        writer.writerows(sorted(rows, key=lambda row: row["city"]))
    roads = {}
    if road_report is not None and road_report.is_file():
        shutil.copy2(road_report, output / "city_road_coverage.csv")
        with road_report.open(newline="", encoding="utf-8") as stream:
            roads = {row["city"]: row for row in csv.DictReader(stream)}
        road_dir = output / "road-maps"
        road_dir.mkdir(exist_ok=True)
        for row in roads.values():
            if row.get("status") != "mapped":
                continue
            source = road_report.parent / row["image"]
            target = road_dir / Path(row["image"]).name
            if source.is_file() and (not target.is_file() or source.stat().st_mtime > target.stat().st_mtime):
                shutil.copy2(source, target)
    cards = []
    for row in sorted(rows, key=lambda item: item["city"]):
        city = html.escape(row["city"])
        src = quote(row["image"])
        road = roads.get(row["city"], {})
        road_name = row["city"].replace("/", "__") + ".png"
        road_src = "road-maps/" + quote(road_name)
        has_roads = road.get("status") == "mapped" and (output / "road-maps" / road_name).is_file()
        ratio = 100 * float(row["located_to_disk_ratio"])
        area = row["coverage_km2_estimate"]
        note = (f" · ~{float(area):,.1f} km²" if area != "" else
                " · too few to map" if row["status"] == "insufficient_coordinates" else
                " · coordinates unavailable")
        detail = f"{int(row['located_panoids']):,} located · {ratio:.1f}% relative to disk count" + note
        road_detail = (f"<p>覆盖道路 {float(road['covered_road_percent']):.1f}% · "
                       f"绿 {float(road['covered_road_km']):,.1f} km / "
                       f"红 {float(road['uncovered_road_km']):,.1f} km</p>"
                       if has_roads else "<p>路网覆盖无法判定</p>")
        road_button = (f'<button type="button" class="mode active" data-src="{road_src}" '
                       f'data-title="{city} · 路网分段">路网分段</button>' if has_roads else "")
        active = "" if has_roads else " active"
        initial_src = road_src if has_roads else src
        initial_title = f"{city} · " + ("路网分段" if has_roads else "50 m 范围")
        cards.append(f'<article data-search="{city.lower()}"><div class="modes">{road_button}'
                     f'<button type="button" class="mode{active}" data-src="{src}" '
                     f'data-title="{city} · 50 m 范围">50 m 范围</button></div>'
                     f'<button type="button" class="preview" aria-label="打开 {initial_title}">'
                     f'<img loading="lazy" src="{initial_src}" alt="{initial_title}"></button>'
                     f'<h2>{city}</h2>{road_detail}<p>{detail}</p></article>')
    mapped = sum(row["status"] == "mapped" for row in rows)
    total_located = sum(int(row["located_panoids"]) for row in rows)
    total_disk = sum(int(row["disk_panoids_any_image"]) for row in rows)
    road_count = sum(row.get("status") == "mapped" for row in roads.values())
    page = f'''<!doctype html><html lang="zh-CN"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1"><title>300 城街景路网覆盖</title>
<style>body{{margin:0;background:#f6f8f7;color:#263238;font:16px Arial,sans-serif}}
header{{background:white;border-bottom:1px solid #d5dedd;padding:25px max(24px,4vw)}}
h1{{font-size:28px;margin:0 0 12px}}header p{{max-width:1100px;color:#55636b;line-height:1.5;margin:7px 0}}
input{{margin-top:16px;width:min(430px,90%);padding:11px;border:1px solid #aab9bb;font:inherit}}
main{{max-width:1600px;margin:auto;padding:24px;display:grid;grid-template-columns:repeat(auto-fill,minmax(min(100%,360px),1fr));gap:20px}}
article{{background:white;border:1px solid #dbe2e2;padding:12px;min-width:0}}article h2{{font-size:18px;margin:10px 0 4px;overflow-wrap:anywhere}}article p{{font-size:14px;color:#5a6a71;margin:3px 0;overflow-wrap:anywhere}}
.modes{{display:flex;border-bottom:1px solid #dbe2e2;min-height:41px;gap:14px}}.mode{{background:transparent;border:0;border-bottom:2px solid transparent;color:#56656b;cursor:pointer;font:inherit;padding:7px 3px}}.mode.active{{border-color:#237d68;color:#1e594b;font-weight:700}}
.preview{{display:block;background:transparent;border:0;padding:0;width:100%;cursor:zoom-in}}.preview img{{display:block;width:100%;aspect-ratio:1200/850;object-fit:contain}}
dialog{{width:min(96vw,1500px);max-height:96vh;border:0;padding:12px;background:#fff;box-shadow:0 12px 36px #0005}}dialog::backdrop{{background:#182428b8}}.viewer-bar{{display:flex;justify-content:space-between;align-items:center;gap:16px;padding:2px 4px 10px}}
#viewer-title{{font-size:17px;overflow-wrap:anywhere}}#viewer-close{{border:1px solid #b9c8c8;background:#fff;color:#263238;padding:8px 12px;cursor:pointer;font:inherit;flex:none}}#viewer-image{{display:block;max-width:100%;max-height:calc(96vh - 74px);margin:auto;object-fit:contain}}
@media(max-width:480px){{#viewer-title{{font-size:14px}}#viewer-close{{font-size:14px}}}}
</style></head><body><header><h1>300 城街景路网覆盖</h1>
<p>{len(rows)} 个城市 · 已生成 {road_count} 个路网分段图 · {mapped} 个 50 m 范围图 · {total_located:,} 个可定位全景 / {total_disk:,} 个磁盘全景。其余城市的路网仍在处理。</p>
<p>绿色表示道路分段中点距离可定位街景点不超过 50 m，红色表示超过 50 m；每段最长 50 m。路网来自 Overture Maps，范围为街景点观测范围，并非城市行政边界。街景点依据元数据中的图像可用标记，尚未逐一与磁盘 JPG 核对；缺少经纬度的图像无法参与判定。达拉斯、休斯敦、印第安纳波利斯的坐标不足。</p>
<input id="search" type="search" placeholder="搜索城市或国家" aria-label="搜索城市或国家">
<p><a href="city_coverage.csv">下载全景统计 CSV</a> · <a href="city_road_coverage.csv">下载道路分段统计 CSV</a></p></header>
<main id="grid">{''.join(cards)}</main>
<dialog id="viewer"><div class="viewer-bar"><strong id="viewer-title"></strong><button id="viewer-close" type="button">← 返回图集</button></div><img id="viewer-image" alt="城市覆盖图"></dialog>
<script>
const viewer=document.querySelector('#viewer');let opener;
document.querySelector('#search').addEventListener('input',e=>{{const s=e.target.value.toLowerCase().trim();document.querySelectorAll('article').forEach(a=>a.hidden=!a.dataset.search.includes(s))}});
document.querySelectorAll('.mode').forEach(button=>button.addEventListener('click',()=>{{const card=button.closest('article');card.querySelectorAll('.mode').forEach(b=>b.classList.toggle('active',b===button));const img=card.querySelector('.preview img');img.src=button.dataset.src;img.alt=button.dataset.title;card.querySelector('.preview').setAttribute('aria-label','打开 '+button.dataset.title)}}));
document.querySelectorAll('.preview').forEach(button=>button.addEventListener('click',()=>{{opener=button;const img=button.querySelector('img');document.querySelector('#viewer-image').src=img.src;document.querySelector('#viewer-title').textContent=img.alt;viewer.showModal()}}));
document.querySelector('#viewer-close').addEventListener('click',()=>viewer.close());viewer.addEventListener('close',()=>{{document.querySelector('#viewer-image').removeAttribute('src');opener?.focus()}});
</script>
</body></html>'''
    (output / "index.html").write_text(page, encoding="utf-8")


def run(output: Path, city_filter: str | None = None, recovery: Path | None = None,
        road_report: Path | None = None):
    city_map = json.loads(CITY_MAP.read_text())
    cities = sorted(set(city_map.values()))
    if city_filter:
        if city_filter not in cities:
            raise ValueError(f"Unknown city: {city_filter}")
        cities = [city_filter]
    with DISK_COUNTS.open(newline="", encoding="utf-8") as stream:
        disk = {row["city"]: row for row in csv.DictReader(stream)}
    if set(city_map.values()) != set(disk):
        raise ValueError("City map and disk inventory differ")
    output.mkdir(parents=True, exist_ok=True)
    checkpoint = output / "progress.csv"
    if checkpoint.exists():
        with checkpoint.open(newline="", encoding="utf-8") as stream:
            prior = {row["city"]: row for row in csv.DictReader(stream)}
    else:
        prior = {}
    with checkpoint.open("a", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=FIELDS)
        if not prior:
            writer.writeheader()
        for city in cities:
            if city in prior and (output / prior[city]["image"]).is_file():
                continue
            row = process_city(city, disk[city], output, recovery)
            writer.writerow(row)
            stream.flush()
            prior[city] = row
            print(f"{len(prior)}/{len(city_map)} {city}: {row['status']}, "
                  f"{int(row['located_panoids']):,} located", flush=True)
    if not city_filter and len(prior) != len(city_map):
        raise ValueError("Incomplete coverage atlas")
    write_report(list(prior.values()), output, road_report)
    print(output / "index.html", flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=OUTPUT)
    parser.add_argument("--city", help="Pilot one city, then resume the full atlas")
    parser.add_argument("--recovery", type=Path,
                        help="Optional recovered source and global coordinate directory")
    parser.add_argument("--road-report", type=Path, help="Road-section report and road-maps/ assets")
    parser.add_argument("--refresh-site-only", action="store_true",
                        help="Rebuild the existing webpage without processing panorama maps")
    args = parser.parse_args()
    if args.refresh_site_only:
        with (args.output / "progress.csv").open(newline="", encoding="utf-8") as stream:
            write_report(list(csv.DictReader(stream)), args.output, args.road_report)
    else:
        run(args.output, args.city, args.recovery, args.road_report)
