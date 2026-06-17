"""Unit tests for greedy max-coverage pano sampling (scripts.sampling.pano_sampling)."""
import numpy as np
import pandas as pd

from scripts.sampling.pano_sampling import greedy_select, select_greedy_pano_ids


def test_greedy_prefers_max_marginal_coverage():
    # 4 candidates over 6 nodes. {0,1,2} and {3,4,5} together cover everything
    # with 2 picks; the redundant overlapping sets must be skipped.
    cand_sets = [
        np.array([0, 1, 2]),     # 0: left half
        np.array([1, 2]),        # 1: subset of 0 (redundant)
        np.array([3, 4, 5]),     # 2: right half
        np.array([4, 5]),        # 3: subset of 2 (redundant)
    ]
    picked = greedy_select(cand_sets, n_nodes=6, budget=2)
    assert set(picked) == {0, 2}        # the two disjoint maximal sets


def test_greedy_stops_when_fully_covered():
    cand_sets = [np.array([0, 1]), np.array([2, 3]), np.array([0, 1, 2, 3])]
    picked = greedy_select(cand_sets, n_nodes=4, budget=3)
    # one set covers all 4 -> taken first; nothing left to add
    assert picked == [2]


def test_select_greedy_beats_first_n_on_coverage():
    # nodes along a line at x = 0,10,...,90 (10 nodes, ~1 m apart in deg terms)
    lat = np.zeros(10)
    lon = np.arange(10) * (10.0 / 111_000.0)        # 10 m spacing
    # 12 panos: first 6 all sit on node 0 (clustered); last 6 spread on 0..90
    p_lon = np.concatenate([np.zeros(6), np.arange(0, 100, 18) / 111_000.0])
    panos = pd.DataFrame({
        "pano_id": [f"p{i}" for i in range(len(p_lon))],
        "lon": p_lon, "lat": np.zeros(len(p_lon)),
    })
    sel, stats = select_greedy_pano_ids(
        panos, lon, lat, budget=3, radius_m=15.0, thin_cell_m=1.0,
        max_candidates=100,
    )
    assert stats["n_selected"] == 3
    # greedy must spread out, not stack on the clustered node-0 panos
    assert stats["covered"] >= 6
    assert len(set(sel)) == 3
