#!/usr/bin/env python3
"""Build assets for Dashboard 2 (feature space) and 3 (analysis).

    /opt/conda/bin/python3 scripts/build_dashboard_data2.py

Emits to dashboard/data/:
  <city>_embed.bin   Float32 [x, y, ...]   joint-UMAP 2D coord per node (same order as *_pos.bin)
  analysis.json      all sweep CSVs + computed 32x32 transition / basis-similarity matrices
"""
import os
for _v in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ.setdefault(_v, "1")
import json, math
import numpy as np
import pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
OUT = os.path.join(ROOT, "dashboard", "data")
SWEEP = os.path.join(ROOT, "outputs", "sweep")
CITIES = {"Vienna": "outputs/dedup_bld/Vienna", "HongKong": "outputs/dedup_bld/HongKong"}
ACT = [f"a_{i:03d}" for i in range(32)]


def clean(o):
    if isinstance(o, dict):  return {k: clean(v) for k, v in o.items()}
    if isinstance(o, list):  return [clean(v) for v in o]
    if isinstance(o, float) and (math.isnan(o) or math.isinf(o)): return None
    return o


def csv_records(name):
    p = os.path.join(SWEEP, name)
    if not os.path.exists(p): return []
    return clean(pd.read_csv(p).to_dict(orient="records"))


def transition_matrix(run):
    """32x32 row-normalized dominant-basis transition along road edges (full graph)."""
    act = pd.read_parquet(os.path.join(ROOT, run, "road_basis_activation_joint.parquet"),
                          columns=["road_node_id"] + ACT)
    dom = dict(zip(act["road_node_id"], act[ACT].to_numpy().argmax(1)))
    ed = pd.read_parquet(os.path.join(ROOT, run, "road_graph_edges.parquet"),
                         columns=["src_node_id", "dst_node_id"])
    T = np.zeros((32, 32), np.float64)
    s = ed["src_node_id"].map(dom).to_numpy()
    d = ed["dst_node_id"].map(dom).to_numpy()
    ok = ~(pd.isna(s) | pd.isna(d))
    for a, b in zip(s[ok].astype(int), d[ok].astype(int)):
        T[a, b] += 1
    row = T.sum(1, keepdims=True); row[row == 0] = 1
    return (T / row)


def main():
    # ---------- joint 2D UMAP over ALL nodes of both cities (PCA init) ----------
    # PCA init avoids both the spectral-layout crash AND the random-init manifold
    # tearing; it seeds the layout along PC1 (the axis best separating the cities).
    import umap
    stacks, order = [], []
    for city, run in CITIES.items():
        A = pd.read_parquet(os.path.join(ROOT, run, "road_basis_activation_joint.parquet"),
                            columns=ACT).to_numpy(np.float32)
        stacks.append(A); order.append((city, len(A)))
        print(f"[emb] {city}: {len(A):,} nodes")
    allA = np.vstack(stacks)

    mu = allA.mean(0); Xc = allA - mu
    cov_mat = (Xc.T @ Xc) / max(1, len(Xc) - 1)
    w, V = np.linalg.eigh(cov_mat)
    init2d = (Xc @ V[:, w.argsort()[::-1][:2]]).astype(np.float32)
    init2d = (init2d - init2d.mean(0)) / (init2d.std(0) + 1e-9) * 10.0

    print(f"[emb] fitting 2D UMAP on {len(allA):,} points (PCA init)…")
    reducer = umap.UMAP(n_neighbors=15, min_dist=0.12, metric="cosine",
                        n_components=2, init=init2d, verbose=True)
    emb = reducer.fit_transform(allA).astype(np.float32)
    mn, mx = emb.min(0), emb.max(0)
    emb = ((emb - mn) / (mx - mn + 1e-9) * 1000.0).astype(np.float32)
    off = 0
    for city, n in order:
        emb[off:off + n].tofile(os.path.join(OUT, f"{city}_embed.bin"))   # [x,y] per node
        off += n
        print(f"[emb] wrote {city}_embed.bin ({n:,} pts, 2D UMAP)")

    # ---------- analysis.json ----------
    analysis = {
        "summary": csv_records("summary.csv"),
        "visual_syntax": csv_records("visual_syntax.csv"),
        "basis_roles": csv_records("basis_roles.csv"),
        "k_sweep": csv_records("k_sweep.csv"),
        "baseline": csv_records("baseline_consistency.csv"),
        "basis_alignment": csv_records("basis_alignment.csv"),
        "basis_similarity": csv_records("basis_similarity.csv"),
        "sae_metrics": csv_records("sae_metrics.csv"),
        "spatial": csv_records("spatial_organization.csv"),
        "coherence": csv_records("unit_coherence.csv"),
        "sampling_bench": csv_records("sampling_bench.csv"),
        "interp": csv_records("interp_comparison.csv"),
    }
    # transition matrices (per city)
    analysis["transition"] = {}
    for city, run in CITIES.items():
        analysis["transition"][city] = transition_matrix(run).round(5).tolist()
        print(f"[trans] {city} transition matrix done")
    # basis similarity matrix from the formal basis (32x3072)
    try:
        X = np.load(os.path.join(ROOT, "outputs", "transfer", "joint", "road_landscape_basis.npy"))
        Xn = X / (np.linalg.norm(X, axis=1, keepdims=True) + 1e-9)
        S = np.abs(Xn @ Xn.T)
        analysis["basis_sim_matrix"] = S.round(4).tolist()
    except Exception as e:
        print("basis sim err", e)

    json.dump(clean(analysis), open(os.path.join(OUT, "analysis.json"), "w"),
              separators=(",", ":"), allow_nan=False)
    sz = os.path.getsize(os.path.join(OUT, "analysis.json"))
    print(f"[done] analysis.json = {sz/1024:.0f} KB")


if __name__ == "__main__":
    main()
