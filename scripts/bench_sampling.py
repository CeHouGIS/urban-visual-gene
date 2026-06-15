"""Benchmark pano-sampling strategies by road-network coverage.

Compares the current `random` pano sampling against `greedy` max-coverage
selection (with spatial stratification), at several budgets N. Coverage = road
nodes within `radius_m` of any selected pano (same rule as Stage 3). Higher
coverage => fewer nodes that must be interpolated.

Pure geo (numpy/scipy only) — safe to run in its own process. Reads the saved
road_graph_nodes.parquet and the full candidate pano set; extracts NO features.

  python -m scripts.bench_sampling --city Vienna --budgets 500 2000 5000
"""
from __future__ import annotations

import scripts._env  # noqa: F401  (thread pinning before numpy/scipy)

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.spatial import cKDTree

from scripts.cities import load_panos, out_dir
from scripts.pano_sampling import (covered_count, greedy_select, grid_thin,
                                   to_xy)


def bench_city(city: str, budgets: list[int], radius_m: float,
               max_candidates: int, thin_cell_m: float, seed: int = 42):
    nodes = pd.read_parquet(out_dir(city) / "road_graph_nodes.parquet")
    lat0 = float(nodes["lat"].mean())
    node_xy = to_xy(nodes["lon"].to_numpy(), nodes["lat"].to_numpy(), lat0)
    n_nodes = len(node_xy)
    node_tree = cKDTree(node_xy)

    panos = load_panos(city, None)
    pano_xy_all = to_xy(panos["lon"].to_numpy(), panos["lat"].to_numpy(), lat0)
    n_cand_full = len(pano_xy_all)

    # candidate pool for greedy: grid-thin huge sets (spatial stratification)
    keep = grid_thin(pano_xy_all, thin_cell_m, max_candidates, seed)
    cand_xy = pano_xy_all[keep]
    # precompute each candidate's covered-node set
    hit = node_tree.query_ball_point(cand_xy, r=radius_m)
    cand_sets = [np.asarray(h, dtype=np.int64) for h in hit]

    rng = np.random.default_rng(seed)
    rows = []
    for N in budgets:
        # random baseline: same rule as load_panos(.sample)
        ridx = rng.choice(n_cand_full, size=min(N, n_cand_full), replace=False)
        rand_cov = covered_count(pano_xy_all[ridx], node_tree, n_nodes, radius_m)
        # greedy over thinned candidates
        gidx = greedy_select(cand_sets, n_nodes, N)
        greedy_cov = covered_count(cand_xy[gidx], node_tree, n_nodes, radius_m)
        rows.append({
            "city": city, "N": N, "n_nodes": n_nodes,
            "random_covered": rand_cov,
            "random_cov_pct": round(rand_cov / n_nodes * 100, 2),
            "random_interp_pct": round((1 - rand_cov / n_nodes) * 100, 2),
            "greedy_covered": greedy_cov,
            "greedy_cov_pct": round(greedy_cov / n_nodes * 100, 2),
            "greedy_interp_pct": round((1 - greedy_cov / n_nodes) * 100, 2),
            "coverage_gain_x": round(greedy_cov / max(rand_cov, 1), 2),
        })
        print(f"  N={N:>5}: random {rand_cov:>6} ({rand_cov/n_nodes*100:4.1f}%) | "
              f"greedy {greedy_cov:>6} ({greedy_cov/n_nodes*100:4.1f}%) | "
              f"x{greedy_cov/max(rand_cov,1):.2f}", flush=True)
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--city", required=True, choices=["Vienna", "HongKong"])
    ap.add_argument("--budgets", type=int, nargs="+", default=[500, 2000, 5000])
    ap.add_argument("--radius-m", type=float, default=100.0)
    ap.add_argument("--max-candidates", type=int, default=80_000,
                    help="cap greedy candidate pool after grid-thinning")
    ap.add_argument("--thin-cell-m", type=float, default=25.0)
    ap.add_argument("--out", default="outputs/sweep/sampling_bench.csv")
    args = ap.parse_args()

    print(f"[{args.city}] benchmarking random vs greedy coverage", flush=True)
    rows = bench_city(args.city, args.budgets, args.radius_m,
                      args.max_candidates, args.thin_cell_m)

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    df = pd.DataFrame(rows)
    if out.exists():
        prev = pd.read_csv(out)
        prev = prev[prev["city"] != args.city]
        df = pd.concat([prev, df], ignore_index=True)
    df.to_csv(out, index=False)
    print(f"\nwrote {out}")


if __name__ == "__main__":
    main()
