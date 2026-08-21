#!/usr/bin/env python3
"""Create publication-ready global and Hong Kong street-view sampling maps."""

import csv
import json
import sqlite3
import argparse
from pathlib import Path

import geopandas as gpd
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.lines import Line2D
from PIL import Image
from shapely.geometry import shape


HOME = Path("/global/scratch/users/cehou")
REPO = HOME / "urban-visual-gene"
SVI = HOME / "data/SVIs"
OUT = REPO / "formal/sampling_maps"
RESEARCH = {
    "HongKong", "Singapore", "Amsterdam", "CapeTown", "Paris", "SaoPaulo",
    "MexicoCity", "Sydney", "Jakarta", "Dhaka", "NewDelhi", "Manila",
}
WORLD = Path("/global/scratch/users/cehou/conda_envs/svi/lib/python3.11/site-packages/pyogrio/tests/fixtures/naturalearth_lowres/naturalearth_lowres.shp")
HK_IMAGE = SVI / "GSV/images/China/HongKong/1/1/m/11mNrzjtNZDAdojKFQKHRQ_90.jpg"
HK_LON, HK_LAT = 114.1693463268765, 22.3062129383001


def aggregate_cities():
    rows = list(csv.DictReader((SVI / "gsv_download_list.csv").open()))
    stats = []
    for row in rows:
        city_root = SVI / "GSV/metadata" / row["countryname"] / row["cityname"]
        lon = lat = None
        boundary_files = list((city_root / "boundary").glob("*.geojson"))
        if boundary_files:
            geom = shape(json.loads(boundary_files[0].read_text())["features"][0]["geometry"]).centroid
            lon, lat = geom.x, geom.y
        if lon is None:
            stats.append({**row, "lon": "", "lat": "", "success": 0, "failed": 0, "status": "unverified"})
            continue
        meta_dir = city_root / "meta"
        db = next(meta_dir.glob("*.db"), None) if meta_dir.exists() else None
        has_metadata = False
        if db is not None:
            try:
                con = sqlite3.connect(f"file:{db}?mode=ro", uri=True)
                has_metadata = con.execute("SELECT 1 FROM gsv LIMIT 1").fetchone() is not None
                con.close()
            except sqlite3.Error:
                has_metadata = False
        success, failed = int(has_metadata), int(not has_metadata)
        status = "failed" if failed else ("research" if row["cityname"] in RESEARCH else "collected_other")
        stats.append({**row, "lon": lon, "lat": lat, "success": int(success or 0),
                      "failed": int(failed or 0), "metadata_status": ">0" if has_metadata else "0", "status": status})
    with (OUT / "global_sampling_status.csv").open("w", newline="") as f:
        cols = ["cityname", "countryname", "lon", "lat", "metadata_status", "success", "failed", "status"]
        w = csv.DictWriter(f, fieldnames=cols, extrasaction="ignore", lineterminator="\n")
        w.writeheader(); w.writerows(stats)
    return stats


def global_map(stats):
    world = gpd.read_file(WORLD)
    fig, ax = plt.subplots(figsize=(16, 8.7), facecolor="white")
    ax.set_facecolor("white")
    world.plot(ax=ax, color="#e7ebf0", edgecolor="#b8c1cc", linewidth=.35)
    ax.text(-168, -47, "LAND WITHOUT A PLANNED CITY SAMPLE", color="#7b8794", fontsize=8,
            weight="bold", alpha=.8)

    valid = [r for r in stats if r["lon"] != ""]
    other = [r for r in valid if r["status"] == "collected_other"]
    study = [r for r in valid if r["status"] == "research"]
    ax.scatter([r["lon"] for r in other], [r["lat"] for r in other], s=25,
               c="#3976d3", alpha=.82, edgecolors="white", linewidths=.35, zorder=4)
    ax.scatter([r["lon"] for r in study], [r["lat"] for r in study], s=105,
               marker="*", c="#00a878", edgecolors="white", linewidths=.7, zorder=6)

    failed = [r for r in valid if r["status"] == "failed"]
    ax.scatter([r["lon"] for r in failed], [r["lat"] for r in failed], s=48,
               marker="x", c="#d93654", linewidths=1.35, alpha=.95, zorder=7)

    offsets = {"Singapore": (3, -7), "Jakarta": (3, -8), "Dhaka": (3, 6),
               "NewDelhi": (-20, 7), "HongKong": (3, 7), "Manila": (3, 7),
               "Amsterdam": (3, 7), "Paris": (-15, -10)}
    for r in study:
        dx, dy = offsets.get(r["cityname"], (3, 6))
        label = {"CapeTown":"Cape Town", "SaoPaulo":"São Paulo", "MexicoCity":"Mexico City",
                 "NewDelhi":"New Delhi", "HongKong":"Hong Kong"}.get(r["cityname"], r["cityname"])
        ax.annotate(label, (r["lon"], r["lat"]), xytext=(dx, dy), textcoords="offset points",
                    fontsize=7.5, color="#334155", zorder=7)

    ax.set_title("Global Street-View Sampling Footprint", loc="left", color="#172033",
                 fontsize=23, weight="bold", pad=16)
    ax.text(-179, 88, f"12 study cities · {sum(r['status']!='failed' for r in valid)} cities with metadata · {len(failed)} failed / zero-metadata cities",
            color="#64748b", fontsize=10, va="bottom")
    legend = [
        Line2D([0],[0], marker="*", color="none", markerfacecolor="#00a878", markeredgecolor="white", markersize=12, label="Current 12-city research sample"),
        Line2D([0],[0], marker="o", color="none", markerfacecolor="#3976d3", markeredgecolor="white", markersize=7, label="Retrieved, outside current study"),
        Line2D([0],[0], marker="x", color="#d93654", markersize=8, label="Failed city (metadata count = 0)"),
        Line2D([0],[0], marker="s", color="none", markerfacecolor="#e7ebf0", markeredgecolor="#b8c1cc", markersize=9, label="No planned city sample / unverified"),
    ]
    ax.legend(handles=legend, loc="lower left", ncol=2, facecolor="white", edgecolor="#cbd5e1",
              labelcolor="#334155", fontsize=9, framealpha=.97)
    ax.set_xlim(-180, 180); ax.set_ylim(-58, 90); ax.set_axis_off()
    fig.tight_layout()
    fig.savefig(OUT / "global_streetview_sampling.png", dpi=240, bbox_inches="tight", facecolor=fig.get_facecolor())
    plt.close(fig)


def hong_kong_map():
    base = SVI / "GSV/metadata/China/HongKong"
    roads = gpd.read_file(base / "road/HongKong_China_roads.geojson")
    boundary = gpd.read_file(base / "boundary/HongKong_China_boundary.geojson")
    db = next((base / "meta").glob("*.db"))
    con = sqlite3.connect(str(db))
    points = np.asarray(con.execute("SELECT lon,lat,download FROM gsv WHERE selected=1").fetchall())
    con.close()
    ok, fail = points[points[:,2] == 1], points[points[:,2] == 0]
    rng = np.random.default_rng(42)
    ok = ok[rng.choice(len(ok), min(45000, len(ok)), replace=False)]
    fail = fail[rng.choice(len(fail), min(18000, len(fail)), replace=False)]

    fig = plt.figure(figsize=(16, 8.5), facecolor="#070b14")
    gs = fig.add_gridspec(1, 2, width_ratios=[1.65, 1], wspace=.03)
    ax = fig.add_subplot(gs[0,0]); photo = fig.add_subplot(gs[0,1])
    ax.set_facecolor("#08111f")
    boundary.plot(ax=ax, facecolor="#111c2c", edgecolor="#65748a", linewidth=.7)
    roads.plot(ax=ax, color="#46556b", linewidth=.18, alpha=.42)
    ax.scatter(ok[:,0], ok[:,1], s=.55, c="#25d7ff", alpha=.28, linewidths=0, rasterized=True)
    ax.scatter(fail[:,0], fail[:,1], s=1.1, c="#ff667d", alpha=.38, linewidths=0, rasterized=True)
    ax.scatter([HK_LON], [HK_LAT], s=155, facecolors="none", edgecolors="#f6c945", linewidths=2.2, zorder=8)
    ax.scatter([HK_LON], [HK_LAT], s=20, c="#f6c945", zorder=9)
    ax.annotate("Selected street view\nMong Kok / Kowloon", (HK_LON, HK_LAT), xytext=(34, 30),
                textcoords="offset points", color="#f6c945", fontsize=9, weight="bold",
                arrowprops=dict(arrowstyle="-", color="#f6c945", lw=1.1))
    ax.set_title("Hong Kong: Sampling Along the Street Network", loc="left", color="#eef5ff",
                 fontsize=19, weight="bold", pad=13)
    ax.text(.0, 1.01, f"{len(points):,} planned locations · {int((points[:,2]==1).sum()):,} retrieved · {int((points[:,2]==0).sum()):,} failed",
            transform=ax.transAxes, color="#9badc5", fontsize=9)
    ax.set_axis_off()

    photo.set_facecolor("#0d1626")
    photo.imshow(Image.open(HK_IMAGE))
    photo.set_title("Representative sampled scene", color="#eef5ff", fontsize=15, weight="bold", pad=12)
    photo.text(.02, -.055, "22.3062°N, 114.1693°E · Kowloon · March 2022 · heading 90°",
               transform=photo.transAxes, color="#9badc5", fontsize=9)
    photo.set_xticks([]); photo.set_yticks([])
    for spine in photo.spines.values(): spine.set_color("#31425d")
    handles = [
        Line2D([0],[0], marker="o", color="none", markerfacecolor="#25d7ff", markersize=6, label="Retrieved location"),
        Line2D([0],[0], marker="o", color="none", markerfacecolor="#ff667d", markersize=6, label="Failed query"),
        Line2D([0],[0], marker="o", color="none", markerfacecolor="none", markeredgecolor="#f6c945", markersize=9, label="Selected scene"),
    ]
    ax.legend(handles=handles, loc="lower left", facecolor="#0d1626", edgecolor="#31425d",
              labelcolor="#dbe4f0", fontsize=9)
    fig.savefig(OUT / "hongkong_streetview_sampling.png", dpi=240, bbox_inches="tight", facecolor=fig.get_facecolor())
    plt.close(fig)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--global-only", action="store_true", help="Skip the Hong Kong detail figure")
    args = parser.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    city_stats = aggregate_cities()
    global_map(city_stats)
    if not args.global_only:
        hong_kong_map()
    print(f"Saved figures to {OUT}")
