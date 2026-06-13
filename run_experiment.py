"""End-to-end experiment runner for Vienna and Hong Kong.

Usage:
  python run_experiment.py --city Vienna --max-panos 5000
  python run_experiment.py --city HongKong --max-panos 5000
  python run_experiment.py --city Vienna   # full run
"""
from __future__ import annotations

import argparse
import json
import sqlite3
import sys
import time
from pathlib import Path

import geopandas as gpd
import numpy as np
import pandas as pd

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent))

from scripts.io_utils import save_report
from scripts.stage1_extract_pano_features import extract_pano_features
from scripts.stage2_map_match_road import map_match
from scripts.stage3_build_road_context_features import build_road_context_features
from scripts.stage4_train_road_basis_model import train_basis_model
from scripts.stage5_infer_road_basis_activation import infer_activation
from scripts.stage6_extract_road_units import extract_road_units

# Prefer the local data mirror (./data, populated by scripts/copy_data.py);
# fall back to the NAS source if the local copy is absent.
_LOCAL = Path(__file__).parent / "data/SVIs/GSV"
_NAS = Path("/workplace/nas_data/houce/urban_visual_gene/SVIs/GSV")
NAS = _LOCAL if _LOCAL.exists() else _NAS


def load_vienna(max_panos: int | None):
    pano_df = pd.read_parquet(NAS / "metadata/Austria/Vienna/panoids.parquet")
    pano_df = pano_df.rename(columns={"panoid": "pano_id"})
    pano_df["city"] = "Vienna"
    if max_panos:
        pano_df = pano_df.sample(max_panos, random_state=42)
    img_root = NAS / "images/Austria/Vienna"
    roads    = gpd.read_file(NAS / "metadata/Austria/Vienna/edges.shp")
    roads    = roads.rename(columns={"osmid": "road_id"})
    roads["road_id"] = roads["road_id"].astype(str)
    return pano_df, img_root, roads, "vienna", Path("outputs/Austria/Vienna")


def load_hongkong(max_panos: int | None):
    con = sqlite3.connect(NAS / "metadata/China/HongKong/meta/China_HongKong.db")
    pano_df = pd.read_sql(
        "SELECT panoid as pano_id, lat, lon FROM gsv WHERE download=1", con
    )
    con.close()
    pano_df["city"] = "HongKong"
    if max_panos:
        pano_df = pano_df.sample(max_panos, random_state=42)
    img_root = NAS / "images/China/HongKong"
    roads    = gpd.read_file(
        NAS / "metadata/China/HongKong/road/China_Hong_Kong.geojson"
    )
    roads    = roads.rename(columns={"osmid": "road_id"})
    roads["road_id"] = roads["road_id"].astype(str)
    return pano_df, img_root, roads, "hongkong", Path("outputs/China/HongKong")


def run(city: str, max_panos: int | None, model: str, K: int,
        batch_size: int, epochs: int, skip_stage1: bool):
    t0 = time.time()
    print(f"\n{'='*60}")
    print(f"  MRLU Pipeline — {city}  (max_panos={max_panos or 'ALL'})")
    print(f"{'='*60}\n")

    if city == "Vienna":
        pano_df, img_root, roads, path_style, out_dir = load_vienna(max_panos)
    else:
        pano_df, img_root, roads, path_style, out_dir = load_hongkong(max_panos)

    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "stage_reports").mkdir(exist_ok=True)

    # ── Stage 1: Feature extraction ──────────────────────────────────────────
    feat_path = out_dir / "pano_features.parquet"
    if skip_stage1 and feat_path.exists():
        print("[ Stage 1 ] Skipping — loading existing features")
        feat_df = pd.read_parquet(feat_path)
    else:
        print("[ Stage 1 ] Extracting DINOv2 4-heading features...")
        feat_df, r1 = extract_pano_features(
            pano_df, img_root,
            model_name=model,
            batch_size=batch_size,
            path_style=path_style,
        )
        feat_df.to_parquet(feat_path, index=False)
        save_report(out_dir / "stage_reports/stage1_report.json", r1)
        print(f"  ✓ {len(feat_df)} panos, D={r1['D_total']}, "
              f"missing={r1['n_missing_all']}, t={time.time()-t0:.0f}s")

    # ── Stage 2: Map-match + road graph ──────────────────────────────────────
    # Clip road network to pano bounding box + 500m buffer to avoid building
    # nodes for the entire city when only a subset of panos is used.
    buf_deg = 500.0 / 111_000.0
    lat_min, lat_max = feat_df["lat"].min() - buf_deg, feat_df["lat"].max() + buf_deg
    lon_min, lon_max = feat_df["lon"].min() - buf_deg, feat_df["lon"].max() + buf_deg
    roads_clipped = roads.cx[lon_min:lon_max, lat_min:lat_max].copy()
    print(f"  Road segments in coverage area: {len(roads_clipped)} / {len(roads)}")

    print("\n[ Stage 2 ] Map-matching panos to road network...")
    t2 = time.time()
    matched, nodes, edges, r2 = map_match(
        feat_df, roads_clipped,
        max_dist_m=30.0,
        node_spacing_m=25.0,
    )
    matched.to_parquet(out_dir / "road_matched_panos.parquet",  index=False)
    nodes.to_parquet(out_dir   / "road_graph_nodes.parquet",    index=False)
    edges.to_parquet(out_dir   / "road_graph_edges.parquet",    index=False)
    save_report(out_dir / "stage_reports/stage2_report.json", r2)
    print(f"  ✓ matched={r2['n_panos_matched']}/{r2['n_panos_total']} "
          f"({r2['matched_ratio']*100:.1f}%), nodes={r2['n_road_nodes']}, "
          f"lcc={r2['lcc_ratio']*100:.1f}%, t={time.time()-t2:.0f}s")

    # ── Stage 3: Build Z_road ─────────────────────────────────────────────────
    print("\n[ Stage 3 ] Building road context features Z_road...")
    t3 = time.time()
    ctx_df, r3 = build_road_context_features(
        matched, nodes, edges,
        search_radius_m=100.0,
        kernel_sigma_m=30.0,
        context_radius_m=100.0,
    )
    ctx_df.to_parquet(out_dir / "road_context_features.parquet", index=False)
    save_report(out_dir / "stage_reports/stage3_report.json", r3)
    print(f"  ✓ {r3['n_road_nodes']} nodes, coverage={r3['coverage_ratio']*100:.1f}%, "
          f"D={r3['embedding_dim']}, t={time.time()-t3:.0f}s")

    # ── Stage 4: Train basis model (only covered nodes) ──────────────────────
    # Training on all 237K+ road nodes is infeasible; use only nodes with actual
    # pano coverage (total_weight > 0). The full ctx_df is still used for Stage 5.
    ctx_covered = ctx_df[ctx_df["n_panos"] > 0].copy().reset_index(drop=True)
    print(f"\n[ Stage 4 ] Training sparse autoencoder (K={K}, epochs={epochs})...")
    print(f"  Using {len(ctx_covered)}/{len(ctx_df)} covered road nodes for training")
    t4 = time.time()
    basis_model, r4 = train_basis_model(
        ctx_covered, edges,
        K=K, hidden=512,
        epochs=epochs,
        lambda_sparse=5e-3,
        lambda_spatial=1e-3,
        lambda_div=1e-3,
        output_dir=Path("models"),
    )
    save_report(out_dir / "stage_reports/stage4_report.json", r4)
    print(f"  ✓ loss {r4['initial_train_loss']:.4f} → {r4['final_train_loss']:.4f}, "
          f"t={time.time()-t4:.0f}s")

    # ── Stage 5: Infer activation A ───────────────────────────────────────────
    print("\n[ Stage 5 ] Inferring basis activation A...")
    t5 = time.time()
    act_df, r5 = infer_activation(ctx_df, basis_model)
    act_df.to_parquet(out_dir / "road_basis_activation.parquet", index=False)
    save_report(out_dir / "stage_reports/stage5_report.json", r5)
    print(f"  ✓ median_active={r5['median_active_basis']:.1f}/{K}, "
          f"mean_recon={r5['mean_recon_error']:.4f}, t={time.time()-t5:.0f}s")

    # ── Stage 6: Extract MRLU ─────────────────────────────────────────────────
    # road_graph_nodes carries no pano count; attach n_panos from Z_road so the
    # min_panos filter (and unit pano stats) reflect real visual coverage.
    npano = ctx_df.drop_duplicates("road_node_id").set_index("road_node_id")["n_panos"]
    nodes = nodes.copy()
    nodes["n_panos"] = nodes["road_node_id"].map(npano).fillna(0).astype(int)

    print("\n[ Stage 6 ] Extracting road landscape units...")
    t6 = time.time()
    b_gdf, u_gdf, stats, r6 = extract_road_units(
        act_df, nodes, edges,
        boundary_quantile=0.90,
        min_road_nodes=3,
        min_road_length_m=50.0,
        min_panos=3,
    )
    if len(b_gdf) > 0:
        b_gdf.to_file(out_dir / "road_activation_boundaries.geojson", driver="GeoJSON")
    if len(u_gdf) > 0:
        u_gdf.to_file(out_dir / "minimum_road_landscape_units.geojson", driver="GeoJSON")
    stats.to_csv(out_dir / "unit_statistics.csv", index=False)
    save_report(out_dir / "stage_reports/stage6_report.json", r6)
    print(f"  ✓ boundaries={r6['n_boundary_edges']}, units={r6['n_units_after_filter']}, "
          f"t={time.time()-t6:.0f}s")

    total = time.time() - t0
    print(f"\n{'='*60}")
    print(f"  {city} done in {total:.0f}s  —  {r6['n_units_after_filter']} MRLU extracted")
    print(f"  Output: {out_dir}/")
    print(f"{'='*60}\n")

    return stats


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--city",        required=True, choices=["Vienna", "HongKong", "both"])
    ap.add_argument("--max-panos",   type=int, default=None,
                    help="Limit panos for quick test (default: all)")
    ap.add_argument("--model",       default="dinov2_vitb14")
    ap.add_argument("--K",           type=int, default=100)
    ap.add_argument("--batch-size",  type=int, default=32)
    ap.add_argument("--epochs",      type=int, default=100)
    ap.add_argument("--skip-stage1", action="store_true",
                    help="Skip feature extraction if pano_features.parquet exists")
    args = ap.parse_args()

    cities = ["Vienna", "HongKong"] if args.city == "both" else [args.city]
    for c in cities:
        run(c, args.max_panos, args.model, args.K,
            args.batch_size, args.epochs, args.skip_stage1)
