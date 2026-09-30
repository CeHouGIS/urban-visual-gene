"""Pipeline pre-step — choose panos by greedy max road-coverage.

Runs BEFORE Stage 1 (in its own geo process, no torch). Densifies the road
network into ~25 m coverage nodes, then greedily selects `--max-panos` panos
covering the most nodes. Writes `selected_pano_ids.parquet` (pano_id, rank),
which `load_panos(..., sampling="greedy")` reads so Stage 1 only extracts
features for the chosen, well-spread panos.

  python -m scripts.sampling.select_panos --city Vienna --max-panos 500
"""
from __future__ import annotations

import scripts._env  # noqa: F401  (thread pinning before numpy/scipy)

import argparse

import pandas as pd

from scripts.core.cities import load_panos, load_roads, out_dir
from scripts.core.io_utils import save_report
from scripts.sampling.pano_sampling import select_greedy_pano_ids
from scripts.core.road_graph_utils import sample_road_nodes


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--city", required=True, choices=["Vienna", "HongKong"])
    ap.add_argument("--max-panos", type=int, required=True)
    ap.add_argument("--radius-m", type=float, default=100.0)
    ap.add_argument("--node-spacing-m", type=float, default=25.0)
    ap.add_argument("--max-candidates", type=int, default=80_000)
    args = ap.parse_args()
    out = out_dir(args.city)
    out.mkdir(parents=True, exist_ok=True)

    panos = load_panos(args.city, None)
    roads = load_roads(args.city)
    # clip roads to the pano bounding box (+500 m buffer) so coverage nodes only
    # span where panos actually exist — mirrors Stage 2's clip.
    lon0, lat0 = panos["lon"].min(), panos["lat"].min()
    lon1, lat1 = panos["lon"].max(), panos["lat"].max()
    buf = 500.0 / 111_000.0
    roads = roads.cx[lon0 - buf:lon1 + buf, lat0 - buf:lat1 + buf].copy()
    nodes = sample_road_nodes(roads, spacing_m=args.node_spacing_m)
    print(f"  coverage nodes: {len(nodes)} from {len(roads)} road segments",
          flush=True)

    sel_ids, stats = select_greedy_pano_ids(
        panos, nodes["lon"].to_numpy(), nodes["lat"].to_numpy(),
        budget=args.max_panos, radius_m=args.radius_m,
        thin_cell_m=args.node_spacing_m, max_candidates=args.max_candidates,
    )
    sel_df = pd.DataFrame({"pano_id": sel_ids, "rank": range(len(sel_ids))})
    sel_df.to_parquet(out / "selected_pano_ids.parquet", index=False)
    save_report(out / "stage_reports/select_panos_report.json", stats)
    print(f"select_panos ✓ {stats['n_selected']} panos, "
          f"greedy coverage {stats['coverage_pct']}% "
          f"(interp {stats['interp_pct']}%)")


if __name__ == "__main__":
    main()
