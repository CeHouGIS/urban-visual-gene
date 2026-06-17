#!/usr/bin/env python3
"""Build compact web assets for the Urban Visual Gene dashboard (Dashboard 1: map).

Run with the conda python that has pandas/pyarrow:
    /opt/conda/bin/python3 scripts/build_dashboard_data.py

Emits to dashboard/data/:
  <city>_pos.bin     Float32  [lon, lat, ...]            per road node
  <city>_rgb.bin     Uint8    [r, g, b, ...]             joint-PCA-RGB color per node
  <city>_attr.bin    Float32  [basis, entropy, active, recon, ...]  per node
  <city>_units.geojson         simplified MultiLineString units (streets) + props
  <city>_boundaries.geojson    simplified boundary segments
  meta.json                    counts, ranges, basis usage, labels, joint PCA info
"""
import os
# lock BLAS/OpenMP thread pools to 1 BEFORE importing numpy/pandas — avoids the
# Intel-vs-GNU OpenMP conflict that core-dumps on this box (see crash_report).
for _v in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ.setdefault(_v, "1")
import json
import numpy as np
import pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
OUT = os.path.join(ROOT, "dashboard", "data")
os.makedirs(OUT, exist_ok=True)

# city -> source run directory (largest-scale config chosen by user)
CITIES = {
    "Vienna":   "outputs/dedup_bld/Vienna",
    "HongKong": "outputs/dedup_bld/HongKong",
}
ACT_COLS = [f"a_{i:03d}" for i in range(32)]


def clean_nan(obj):
    """Recursively replace NaN/Inf floats with None — invalid in strict JSON,
    and browsers' JSON.parse rejects them (silently breaking the whole layer)."""
    import math
    if isinstance(obj, dict):
        return {k: clean_nan(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [clean_nan(v) for v in obj]
    if isinstance(obj, float) and (math.isnan(obj) or math.isinf(obj)):
        return None
    return obj


def coord_round(geom_coords, nd=5):
    """Round coordinate precision in-place for a nested coord list (~1m at nd=5)."""
    if isinstance(geom_coords[0], (int, float)):
        return [round(geom_coords[0], nd), round(geom_coords[1], nd)]
    return [coord_round(c, nd) for c in geom_coords]


def load_nodes(run):
    df = pd.read_parquet(os.path.join(ROOT, run, "road_basis_activation_joint.parquet"))
    A = df[ACT_COLS].to_numpy(dtype=np.float32)
    return df, A


def main():
    # ---- 1. load both cities, stack activations for a JOINT PCA (comparable colors) ----
    data = {}
    stack = []
    for city, run in CITIES.items():
        df, A = load_nodes(run)
        data[city] = {"df": df, "A": A}
        stack.append(A)
        print(f"[load] {city}: {len(df):,} nodes")

    allA = np.vstack(stack)
    # PCA via SVD on mean-centered activations -> 3 components
    mu = allA.mean(0)
    Xc = allA - mu
    # PCA via 32x32 covariance eigendecomposition (tiny + stable; avoids the big
    # SVD that intermittently SIGFPE-crashes under the OpenMP conflict on this box)
    n = Xc.shape[0]
    cov = (Xc.T @ Xc) / max(1, n - 1)    # (32, 32)
    w, V = np.linalg.eigh(cov)           # ascending eigenvalues
    top = w.argsort()[::-1][:3]
    comps = V[:, top].T                  # (3, 32)
    evr = w[top] / w.sum()
    print(f"[pca] explained variance ratio (3 comp) = {evr.round(4).tolist()}")

    # project all, then per-channel quantile-rank normalize to [0,255] over the JOINT dist
    proj_all = Xc @ comps.T              # (N,3)
    ranks = np.empty_like(proj_all)
    for c in range(3):
        order = proj_all[:, c].argsort()
        r = np.empty(n, dtype=np.float64)
        r[order] = np.linspace(0, 1, n)
        ranks[:, c] = r
    rgb_all = (ranks * 255).astype(np.uint8)

    # split back per city
    meta = {"cities": {}, "pca_explained_variance": [float(x) for x in evr],
            "config": {c: r for c, r in CITIES.items()}}
    off = 0
    for city in CITIES:
        df = data[city]["df"]; A = data[city]["A"]; m = len(df)
        rgb = rgb_all[off:off + m]; off += m

        lon = df["lon"].to_numpy(np.float32)
        lat = df["lat"].to_numpy(np.float32)
        pos = np.empty(m * 2, np.float32); pos[0::2] = lon; pos[1::2] = lat

        dom = A.argmax(1).astype(np.uint8)            # 0..31  exact
        ent = df["activation_entropy"].to_numpy(np.float32)
        act = df["active_basis_count"].to_numpy(np.float32)
        rec = df["reconstruction_error"].to_numpy(np.float32)
        # quantize entropy & recon to uint8 over their own range (range stored in meta)
        def q(x):
            lo, hi = float(x.min()), float(x.max())
            return np.clip(np.round((x - lo) / (hi - lo + 1e-12) * 255), 0, 255).astype(np.uint8)
        attr = np.empty(m * 4, np.uint8)              # [dom, active, ent_q, rec_q]  4 bytes/node
        attr[0::4] = dom
        attr[1::4] = np.clip(act, 0, 255).astype(np.uint8)
        attr[2::4] = q(ent)
        attr[3::4] = q(rec)

        pos.tofile(os.path.join(OUT, f"{city}_pos.bin"))
        rgb.astype(np.uint8).tofile(os.path.join(OUT, f"{city}_rgb.bin"))
        attr.tofile(os.path.join(OUT, f"{city}_attr.bin"))

        # ---- road EDGES as node-index pairs (the actual road network, for line rendering) ----
        nid2idx = {n: i for i, n in enumerate(df["road_node_id"].tolist())}
        ed = pd.read_parquet(os.path.join(ROOT, CITIES[city], "road_graph_edges.parquet"),
                             columns=["src_node_id", "dst_node_id"])
        si = ed["src_node_id"].map(nid2idx).to_numpy()
        di = ed["dst_node_id"].map(nid2idx).to_numpy()
        ok = ~(pd.isna(si) | pd.isna(di))
        si = si[ok].astype(np.int64); di = di[ok].astype(np.int64)
        # dedupe undirected edges (graph stores both directions)
        lo = np.minimum(si, di); hi = np.maximum(si, di)
        key = lo * np.int64(len(df)) + hi
        _, uniq = np.unique(key, return_index=True)
        edges = np.empty(len(uniq) * 2, np.uint32)
        edges[0::2] = lo[uniq]; edges[1::2] = hi[uniq]
        edges.tofile(os.path.join(OUT, f"{city}_edges.bin"))
        meta["cities"].setdefault(city, {})
        n_edges = len(uniq)

        # basis usage histogram (by node argmax)
        usage = np.bincount(A.argmax(1), minlength=32).tolist()

        meta["cities"][city] = {
            "n_nodes": int(m),
            "n_edges": int(n_edges),
            "center": [float(lon.mean()), float(lat.mean())],
            "bounds": [float(lon.min()), float(lat.min()), float(lon.max()), float(lat.max())],
            "basis_usage": usage,
            "entropy_range": [float(ent.min()), float(ent.max())],
            "active_range": [float(act.min()), float(act.max())],
            "recon_range": [float(rec.min()), float(rec.max())],
        }
        print(f"[nodes] {city}: wrote pos/rgb/attr ({m:,} nodes), {n_edges:,} road edges")

        # ---- units geojson (streets colored by category) ----
        ug = json.load(open(os.path.join(ROOT, CITIES[city], "minimum_road_landscape_units_joint.geojson")))
        # merge unit_confidence / stability from unit_statistics
        us = pd.read_csv(os.path.join(ROOT, CITIES[city], "unit_statistics_joint.csv")).set_index("unit_id")
        keep = ["unit_id", "dominant_basis_id", "road_length_m", "n_road_nodes",
                "n_panos", "activation_entropy", "unit_confidence", "mean_boundary_contrast"]
        feats = []
        for f in ug["features"]:
            p = f["properties"]; uid = p.get("unit_id")
            np_ = {k: p.get(k) for k in keep if k in p}
            if uid in us.index:
                np_["unit_confidence"] = float(us.loc[uid, "unit_confidence"])
                np_["stability_score"] = float(us.loc[uid, "stability_score"])
            f["properties"] = np_
            f["geometry"]["coordinates"] = coord_round(f["geometry"]["coordinates"])
            feats.append(f)
        ug["features"] = feats
        json.dump(clean_nan(ug), open(os.path.join(OUT, f"{city}_units.geojson"), "w"),
                  separators=(",", ":"), allow_nan=False)
        meta["cities"][city]["n_units"] = len(feats)
        print(f"[units] {city}: {len(feats):,} units")

        # ---- boundaries geojson (simplified) ----
        bp = os.path.join(ROOT, CITIES[city], "road_activation_boundaries_joint.geojson")
        bg = json.load(open(bp))
        for f in bg["features"]:
            f["properties"] = {k: f["properties"].get(k) for k in ["boundary_score", "activation_distance"]}
            f["geometry"]["coordinates"] = coord_round(f["geometry"]["coordinates"])
        json.dump(clean_nan(bg), open(os.path.join(OUT, f"{city}_boundaries.geojson"), "w"),
                  separators=(",", ":"), allow_nan=False)
        meta["cities"][city]["n_boundaries"] = len(bg["features"])
        print(f"[bound] {city}: {len(bg['features']):,} boundaries")

    # ---- formal-run semantic labels (reference only; indices differ from scale run) ----
    try:
        labels = json.load(open(os.path.join(ROOT, "outputs/sweep/basis_labels_Vienna.json")))
        meta["formal_basis_labels_vienna"] = labels
    except Exception:
        pass

    json.dump(clean_nan(meta), open(os.path.join(OUT, "meta.json"), "w"), indent=1, allow_nan=False)
    # report total asset size
    tot = sum(os.path.getsize(os.path.join(OUT, f)) for f in os.listdir(OUT))
    print(f"\n[done] dashboard/data total = {tot/1e6:.1f} MB")


if __name__ == "__main__":
    main()
