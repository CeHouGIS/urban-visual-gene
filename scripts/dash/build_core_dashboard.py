#!/usr/bin/env python3
"""Core-map dashboard builder (base pipeline outputs, no dedup/joint/buildings).

Reads the per-city run_experiment artefacts for the 4 downloaded cities and
emits the compact web assets the dashboard map + feature-space views need:

  dashboard/data/
    <city>_pos.bin        Float32 [lon,lat,...]                  per road node
    <city>_rgb.bin        Uint8   [r,g,b,...]   joint-PCA-RGB     per road node
    <city>_attr.bin       Uint8   [dom,active,ent_q,rec_q,...]    per road node
    <city>_edges.bin      Uint32  [i,j,...]   undirected node-index pairs
    <city>_embed.bin      Float32 [x,y,...]   joint-UMAP 2D       per road node
    <city>_units.geojson        simplified MRLU street MultiLineStrings + props
    <city>_boundaries.geojson   simplified boundary segments
    meta.json                   per-city counts/ranges/basis usage + joint PCA info
    analysis.json               per-city dominant-basis transition matrices

Run from repo root with the conda python:
    PROJ_DATA=... OMP_NUM_THREADS=1 python -m scripts.dash.build_core_dashboard
"""
import os
for _v in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ.setdefault(_v, "1")
import json
import math
import numpy as np
import pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
OUT = os.path.join(ROOT, "dashboard", "data")
os.makedirs(OUT, exist_ok=True)

# city -> base run_experiment output dir (no _joint suffix on artefacts)
CITIES = {
    "HongKong":  "outputs/China/HongKong",
    "Singapore": "outputs/Singapore/Singapore",
    "Amsterdam": "outputs/Netherlands/Amsterdam",
    "CapeTown":  "outputs/SouthAfrica/CapeTown",
}
K = 32
ACT_COLS = [f"a_{i:03d}" for i in range(K)]
UMAP_FIT_CAP = 200_000   # fit UMAP on at most this many points, transform the rest
PANO_DOT_CAP = 8_000     # cap pano dots per city on the map (subsample for rendering)


def clean_nan(obj):
    if isinstance(obj, dict):
        return {k: clean_nan(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [clean_nan(v) for v in obj]
    if isinstance(obj, float) and (math.isnan(obj) or math.isinf(obj)):
        return None
    return obj


def coord_round(c, nd=5):
    if isinstance(c[0], (int, float)):
        return [round(c[0], nd), round(c[1], nd)]
    return [coord_round(x, nd) for x in c]


def load_nodes(run):
    df = pd.read_parquet(os.path.join(ROOT, run, "road_basis_activation.parquet"))
    A = df[ACT_COLS].to_numpy(dtype=np.float32)
    return df, A


def joint_pca_rgb(allA):
    """3-component PCA via 32x32 covariance eigendecomp (stable), then per-channel
    quantile-rank normalise to [0,255] over the JOINT distribution -> comparable
    colours across cities."""
    mu = allA.mean(0)
    Xc = allA - mu
    n = Xc.shape[0]
    cov = (Xc.T @ Xc) / max(1, n - 1)
    w, V = np.linalg.eigh(cov)
    top = w.argsort()[::-1][:3]
    comps = V[:, top].T
    evr = w[top] / w.sum()
    proj = Xc @ comps.T
    ranks = np.empty_like(proj)
    for c in range(3):
        order = proj[:, c].argsort()
        r = np.empty(n, dtype=np.float64)
        r[order] = np.linspace(0, 1, n)
        ranks[:, c] = r
    rgb = (ranks * 255).astype(np.uint8)
    return rgb, [float(x) for x in evr]


def joint_umap(allA):
    """2D UMAP over all nodes (PCA init). For very large N, fit on a random
    subsample (<= UMAP_FIT_CAP) then transform the rest, so every node still
    gets a coordinate in the same row order."""
    import umap
    n = len(allA)
    mu = allA.mean(0)
    Xc = allA - mu
    cov = (Xc.T @ Xc) / max(1, n - 1)
    w, V = np.linalg.eigh(cov)
    pc2 = V[:, w.argsort()[::-1][:2]]

    if n <= UMAP_FIT_CAP:
        init2d = (Xc @ pc2).astype(np.float32)
        init2d = (init2d - init2d.mean(0)) / (init2d.std(0) + 1e-9) * 10.0
        print(f"[emb] fitting 2D UMAP on all {n:,} pts (PCA init)…")
        reducer = umap.UMAP(n_neighbors=15, min_dist=0.12, metric="cosine",
                            n_components=2, init=init2d, verbose=True)
        emb = reducer.fit_transform(allA).astype(np.float32)
    else:
        rng = np.random.default_rng(42)
        sel = rng.choice(n, UMAP_FIT_CAP, replace=False)
        sub = allA[sel]
        sinit = (Xc[sel] @ pc2).astype(np.float32)
        sinit = (sinit - sinit.mean(0)) / (sinit.std(0) + 1e-9) * 10.0
        print(f"[emb] fitting 2D UMAP on {UMAP_FIT_CAP:,}/{n:,} pts then transforming rest…")
        reducer = umap.UMAP(n_neighbors=15, min_dist=0.12, metric="cosine",
                            n_components=2, init=sinit, verbose=True)
        reducer.fit(sub)
        emb = np.empty((n, 2), np.float32)
        CH = 50_000
        for i in range(0, n, CH):
            emb[i:i + CH] = reducer.transform(allA[i:i + CH])
            print(f"[emb]  transformed {min(i+CH, n):,}/{n:,}")
    mn, mx = emb.min(0), emb.max(0)
    return ((emb - mn) / (mx - mn + 1e-9) * 1000.0).astype(np.float32)


def main():
    # ---- 1. load all cities; stack activations for joint PCA + UMAP ----
    data, stack = {}, []
    for city, run in CITIES.items():
        df, A = load_nodes(run)
        data[city] = {"df": df, "A": A}
        stack.append(A)
        print(f"[load] {city}: {len(df):,} nodes")
    allA = np.vstack(stack)

    rgb_all, evr = joint_pca_rgb(allA)
    print(f"[pca] explained variance ratio (3 comp) = {[round(x,4) for x in evr]}")
    emb_all = joint_umap(allA)

    meta = {"cities": {}, "pca_explained_variance": evr,
            "config": {c: r for c, r in CITIES.items()}}
    off = 0
    for city in CITIES:
        df = data[city]["df"]; A = data[city]["A"]; m = len(df)
        rgb = rgb_all[off:off + m]
        emb = emb_all[off:off + m]
        off += m

        lon = df["lon"].to_numpy(np.float32)
        lat = df["lat"].to_numpy(np.float32)
        pos = np.empty(m * 2, np.float32); pos[0::2] = lon; pos[1::2] = lat

        dom = A.argmax(1).astype(np.uint8)
        ent = df["activation_entropy"].to_numpy(np.float32)
        act = df["active_basis_count"].to_numpy(np.float32)
        rec = df["reconstruction_error"].to_numpy(np.float32)

        def q(x):
            lo, hi = float(x.min()), float(x.max())
            return np.clip(np.round((x - lo) / (hi - lo + 1e-12) * 255), 0, 255).astype(np.uint8)
        attr = np.empty(m * 4, np.uint8)
        attr[0::4] = dom
        attr[1::4] = np.clip(act, 0, 255).astype(np.uint8)
        attr[2::4] = q(ent)
        attr[3::4] = q(rec)

        pos.tofile(os.path.join(OUT, f"{city}_pos.bin"))
        rgb.astype(np.uint8).tofile(os.path.join(OUT, f"{city}_rgb.bin"))
        attr.tofile(os.path.join(OUT, f"{city}_attr.bin"))
        emb.tofile(os.path.join(OUT, f"{city}_embed.bin"))

        # ---- road edges as undirected node-index pairs ----
        nid2idx = {n: i for i, n in enumerate(df["road_node_id"].tolist())}
        ed = pd.read_parquet(os.path.join(ROOT, CITIES[city], "road_graph_edges.parquet"),
                             columns=["src_node_id", "dst_node_id"])
        si = ed["src_node_id"].map(nid2idx).to_numpy()
        di = ed["dst_node_id"].map(nid2idx).to_numpy()
        ok = ~(pd.isna(si) | pd.isna(di))
        si = si[ok].astype(np.int64); di = di[ok].astype(np.int64)
        lo = np.minimum(si, di); hi = np.maximum(si, di)
        key = lo * np.int64(len(df)) + hi
        _, uniq = np.unique(key, return_index=True)
        edges = np.empty(len(uniq) * 2, np.uint32)
        edges[0::2] = lo[uniq]; edges[1::2] = hi[uniq]
        edges.tofile(os.path.join(OUT, f"{city}_edges.bin"))
        n_edges = len(uniq)

        usage = np.bincount(A.argmax(1), minlength=K).tolist()
        meta["cities"][city] = {
            "n_nodes": int(m), "n_edges": int(n_edges),
            "center": [float(lon.mean()), float(lat.mean())],
            "bounds": [float(lon.min()), float(lat.min()), float(lon.max()), float(lat.max())],
            "basis_usage": usage,
            "entropy_range": [float(ent.min()), float(ent.max())],
            "active_range": [float(act.min()), float(act.max())],
            "recon_range": [float(rec.min()), float(rec.max())],
        }
        print(f"[nodes] {city}: pos/rgb/attr/embed ({m:,} nodes), {n_edges:,} edges")

        # ---- units geojson ----
        up = os.path.join(ROOT, CITIES[city], "minimum_road_landscape_units.geojson")
        ug = json.load(open(up))
        usp = os.path.join(ROOT, CITIES[city], "unit_statistics.csv")
        us = pd.read_csv(usp).set_index("unit_id") if os.path.exists(usp) else None
        keep = ["unit_id", "dominant_basis_id", "road_length_m", "n_road_nodes",
                "n_panos", "activation_entropy", "unit_confidence", "mean_boundary_contrast"]
        feats = []
        for f in ug["features"]:
            p = f["properties"]; uid = p.get("unit_id")
            np_ = {k: p.get(k) for k in keep if k in p}
            if us is not None and uid in us.index:
                if "unit_confidence" in us.columns:
                    np_["unit_confidence"] = float(us.loc[uid, "unit_confidence"])
                if "stability_score" in us.columns:
                    sv = us.loc[uid, "stability_score"]
                    if pd.notna(sv):
                        np_["stability_score"] = float(sv)
            f["properties"] = np_
            f["geometry"]["coordinates"] = coord_round(f["geometry"]["coordinates"])
            feats.append(f)
        ug["features"] = feats
        json.dump(clean_nan(ug), open(os.path.join(OUT, f"{city}_units.geojson"), "w"),
                  separators=(",", ":"), allow_nan=False)
        meta["cities"][city]["n_units"] = len(feats)
        print(f"[units] {city}: {len(feats):,} units")

        # ---- boundaries geojson ----
        bp = os.path.join(ROOT, CITIES[city], "road_activation_boundaries.geojson")
        if os.path.exists(bp):
            bg = json.load(open(bp))
            for f in bg["features"]:
                f["properties"] = {k: f["properties"].get(k)
                                   for k in ["boundary_score", "activation_distance"]}
                f["geometry"]["coordinates"] = coord_round(f["geometry"]["coordinates"])
            json.dump(clean_nan(bg), open(os.path.join(OUT, f"{city}_boundaries.geojson"), "w"),
                      separators=(",", ":"), allow_nan=False)
            meta["cities"][city]["n_boundaries"] = len(bg["features"])
            print(f"[bound] {city}: {len(bg['features']):,} boundaries")
        else:
            meta["cities"][city]["n_boundaries"] = 0

        # ---- sampled pano dots (coords + ids) for the map's pano layer ----
        # Full DINOv3 runs have 100k-600k panos/city; cap the on-map dots so the
        # browser layer stays light (the user asked to subsample for rendering).
        pf = os.path.join(ROOT, CITIES[city], "pano_features.parquet")
        if os.path.exists(pf):
            pdf = pd.read_parquet(pf, columns=["pano_id", "lon", "lat"])
            if len(pdf) > PANO_DOT_CAP:
                pdf = pdf.sample(PANO_DOT_CAP, random_state=42).reset_index(drop=True)
            xy = np.empty(len(pdf) * 2, np.float32)
            xy[0::2] = pdf["lon"].to_numpy(np.float32)
            xy[1::2] = pdf["lat"].to_numpy(np.float32)
            xy.tofile(os.path.join(OUT, f"panos_{city}_xy.bin"))
            with open(os.path.join(OUT, f"panos_{city}_ids.txt"), "w") as fh:
                fh.write("\n".join(pdf["pano_id"].astype(str).tolist()))
            meta["cities"][city]["n_panos_sampled"] = int(len(pdf))
            print(f"[panos] {city}: {len(pdf):,} pano dots")

    json.dump(clean_nan(meta), open(os.path.join(OUT, "meta.json"), "w"), indent=1, allow_nan=False)

    # ---- analysis.json: per-city dominant-basis transition matrices ----
    analysis = {"transition": {}}
    for city, run in CITIES.items():
        act = pd.read_parquet(os.path.join(ROOT, run, "road_basis_activation.parquet"),
                              columns=["road_node_id"] + ACT_COLS)
        dom = dict(zip(act["road_node_id"], act[ACT_COLS].to_numpy().argmax(1)))
        ed = pd.read_parquet(os.path.join(ROOT, run, "road_graph_edges.parquet"),
                             columns=["src_node_id", "dst_node_id"])
        T = np.zeros((K, K), np.float64)
        s = ed["src_node_id"].map(dom).to_numpy()
        d = ed["dst_node_id"].map(dom).to_numpy()
        ok = ~(pd.isna(s) | pd.isna(d))
        for a, b in zip(s[ok].astype(int), d[ok].astype(int)):
            T[a, b] += 1
        row = T.sum(1, keepdims=True); row[row == 0] = 1
        analysis["transition"][city] = (T / row).round(5).tolist()
        print(f"[trans] {city} transition matrix done")
    json.dump(clean_nan(analysis), open(os.path.join(OUT, "analysis.json"), "w"),
              separators=(",", ":"), allow_nan=False)

    tot = sum(os.path.getsize(os.path.join(OUT, f)) for f in os.listdir(OUT))
    print(f"\n[done] dashboard/data total = {tot/1e6:.1f} MB")


if __name__ == "__main__":
    main()
