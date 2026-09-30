"""Pano-sampling strategies (random vs greedy max-coverage).

Shared geo primitives used by `bench_sampling` (offline benchmark) and
`select_panos` (pipeline pre-step). Greedy selection picks the panos that cover
the most road-network nodes within `radius_m`, so far fewer nodes have to be
interpolated downstream than under uniform-random sampling.

Pure geo (numpy/scipy/geopandas, NO torch).
"""
from __future__ import annotations

import heapq

import numpy as np
import pandas as pd
from scipy.spatial import cKDTree

DEG_TO_M = 111_000.0


def to_xy(lon: np.ndarray, lat: np.ndarray, lat0: float) -> np.ndarray:
    """Equirectangular lon/lat -> local metres about lat0 (good enough for a
    ~100 m coverage test at city scale)."""
    x = np.asarray(lon) * DEG_TO_M * np.cos(np.radians(lat0))
    y = np.asarray(lat) * DEG_TO_M
    return np.column_stack([x, y]).astype(np.float64)


def grid_thin(xy: np.ndarray, cell_m: float, cap: int, seed: int) -> np.ndarray:
    """Keep at most one point per `cell_m` grid cell (spatial stratification),
    so greedy has a spread candidate pool and stays tractable on huge sets.
    Returns indices into `xy`."""
    keys = np.floor(xy / cell_m).astype(np.int64)
    _, idx = np.unique(keys, axis=0, return_index=True)
    idx = np.sort(idx)
    if len(idx) > cap:
        rng = np.random.default_rng(seed)
        idx = np.sort(rng.choice(idx, size=cap, replace=False))
    return idx


def covered_count(sel_xy: np.ndarray, node_tree: cKDTree, n_nodes: int,
                  radius_m: float) -> int:
    if len(sel_xy) == 0:
        return 0
    hit = node_tree.query_ball_point(sel_xy, r=radius_m)
    covered = np.zeros(n_nodes, dtype=bool)
    for h in hit:
        if h:
            covered[h] = True
    return int(covered.sum())


def greedy_select(cand_sets, n_nodes: int, budget: int) -> list[int]:
    """Lazy-greedy (CELF) max-coverage: repeatedly take the candidate adding the
    most still-uncovered nodes, until `budget` candidates are picked. Returns
    indices into `cand_sets`, in selection order."""
    covered = np.zeros(n_nodes, dtype=bool)
    heap = [(-len(s), i) for i, s in enumerate(cand_sets)]
    heapq.heapify(heap)
    picked: list[int] = []
    while heap and len(picked) < budget:
        neg_gain, i = heapq.heappop(heap)
        s = cand_sets[i]
        gain = int((~covered[s]).sum()) if len(s) else 0
        if gain == 0:
            continue
        if heap and gain < -heap[0][0]:        # stale -> re-insert true gain
            heapq.heappush(heap, (-gain, i))
            continue
        covered[s] = True
        picked.append(i)
    return picked


def select_greedy_pano_ids(panos: pd.DataFrame, node_lon: np.ndarray,
                           node_lat: np.ndarray, budget: int,
                           radius_m: float = 100.0, thin_cell_m: float = 25.0,
                           max_candidates: int = 80_000, seed: int = 42):
    """Greedily choose `budget` pano rows maximising road-node coverage.

    Returns (selected_pano_ids: list[str], stats: dict). Order is greedy rank.
    """
    lat0 = float(np.mean(node_lat))
    node_xy = to_xy(node_lon, node_lat, lat0)
    n_nodes = len(node_xy)
    node_tree = cKDTree(node_xy)

    pano_xy = to_xy(panos["lon"].to_numpy(), panos["lat"].to_numpy(), lat0)
    keep = grid_thin(pano_xy, thin_cell_m, max_candidates, seed)
    cand_xy = pano_xy[keep]
    cand_ids = panos["pano_id"].to_numpy()[keep]

    hit = node_tree.query_ball_point(cand_xy, r=radius_m)
    cand_sets = [np.asarray(h, dtype=np.int64) for h in hit]

    picked = greedy_select(cand_sets, n_nodes, budget)
    sel_ids = [str(cand_ids[i]) for i in picked]
    greedy_cov = covered_count(cand_xy[picked], node_tree, n_nodes, radius_m)
    stats = {
        "budget": budget,
        "n_selected": len(sel_ids),
        "n_candidates": int(len(keep)),
        "n_nodes": n_nodes,
        "covered": greedy_cov,
        "coverage_pct": round(greedy_cov / max(n_nodes, 1) * 100, 2),
        "interp_pct": round((1 - greedy_cov / max(n_nodes, 1)) * 100, 2),
        "radius_m": radius_m,
    }
    return sel_ids, stats
