#!/usr/bin/env python3
"""Reassemble dashboard/data/analysis.json from the (now joint-basis) sweep CSVs +
joint transition/similarity matrices. Skips UMAP (keeps existing *_embed.bin)."""
import os
for _v in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ.setdefault(_v, "1")
import json, math
import numpy as np
import pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
OUT = os.path.join(ROOT, "dashboard", "data")
SWEEP = os.path.join(ROOT, "outputs", "sweep")
CITIES = {"Vienna": "outputs/sweep/Vienna_N5000", "HongKong": "outputs/sweep/HongKong_N2000"}
ACT = [f"a_{i:03d}" for i in range(32)]


def clean(o):
    if isinstance(o, dict):  return {k: clean(v) for k, v in o.items()}
    if isinstance(o, list):  return [clean(v) for v in o]
    if isinstance(o, float) and (math.isnan(o) or math.isinf(o)): return None
    return o


def csv_records(name):
    p = os.path.join(SWEEP, name)
    return clean(pd.read_csv(p).to_dict(orient="records")) if os.path.exists(p) else []


def transition_matrix(run):
    act = pd.read_parquet(os.path.join(ROOT, run, "road_basis_activation_joint.parquet"),
                          columns=["road_node_id"] + ACT)
    dom = dict(zip(act["road_node_id"], act[ACT].to_numpy().argmax(1)))
    ed = pd.read_parquet(os.path.join(ROOT, run, "road_graph_edges.parquet"),
                         columns=["src_node_id", "dst_node_id"])
    T = np.zeros((32, 32), np.float64)
    s = ed["src_node_id"].map(dom).to_numpy(); d = ed["dst_node_id"].map(dom).to_numpy()
    ok = ~(pd.isna(s) | pd.isna(d))
    for a, b in zip(s[ok].astype(int), d[ok].astype(int)):
        T[a, b] += 1
    row = T.sum(1, keepdims=True); row[row == 0] = 1
    return (T / row)


def main():
    A = {k: csv_records(f"{k}.csv") for k in
         ["summary", "visual_syntax", "basis_roles", "k_sweep", "sampling_bench", "interp",
          "spatial", "coherence", "sae_metrics"]}
    A["baseline"] = csv_records("baseline_consistency.csv")
    A["basis_alignment"] = csv_records("basis_alignment.csv")
    A["basis_similarity"] = csv_records("basis_similarity.csv")
    A["spatial"] = csv_records("spatial_organization.csv")
    A["coherence"] = csv_records("unit_coherence.csv")
    A["transition"] = {c: transition_matrix(r).round(5).tolist() for c, r in CITIES.items()}
    X = np.load(os.path.join(ROOT, "outputs", "transfer", "joint", "road_landscape_basis.npy"))
    Xn = X / (np.linalg.norm(X, axis=1, keepdims=True) + 1e-9)
    A["basis_sim_matrix"] = np.abs(Xn @ Xn.T).round(4).tolist()

    # activation co-occurrence among the 32 bases (clustered order) — "grammar"
    from scipy.cluster.hierarchy import linkage, leaves_list, dendrogram
    from scipy.spatial.distance import squareform
    act = np.vstack([pd.read_parquet(os.path.join(ROOT, r, "road_basis_activation_joint.parquet"),
                                     columns=ACT).to_numpy(np.float32) for r in CITIES.values()])
    dead = np.where(act.var(0) < 1e-8)[0].tolist()
    C = np.nan_to_num(np.corrcoef(act.T))
    D = 1.0 - C; np.fill_diagonal(D, 0.0); D = np.clip((D + D.T) / 2, 0, None)
    Z = linkage(squareform(D, checks=False), method="average")
    thr = 0.7 * float(Z[:, 2].max())            # cluster cut -> colour families
    R = dendrogram(Z, labels=list(range(32)), no_plot=True,
                   color_threshold=thr, above_threshold_color="grey")
    PAL = ["#39d6ff", "#22e0a1", "#ffb454", "#ff7eb6", "#7c5cff", "#f0f921", "#9fc6ff", "#e060d0", "#ff9d5c"]
    codes = sorted({c for c in R["color_list"] if c.startswith("C")})
    cmap = {c: PAL[i % len(PAL)] for i, c in enumerate(codes)}
    hx = lambda c: cmap.get(c, "#5f7596")
    A["cooccur"] = {"matrix": np.round(C, 3).tolist(), "order": leaves_list(Z).tolist(), "dead": dead,
                    "max_basis_cos": round(float(np.abs(Xn @ Xn.T)[~np.eye(32, dtype=bool)].max()), 3),
                    "n_families": len(codes),
                    "dendro": {"icoord": R["icoord"], "dcoord": [[round(y, 4) for y in d] for d in R["dcoord"]],
                               "ivl": [int(x) for x in R["ivl"]],
                               "link_colors": [hx(c) for c in R["color_list"]],
                               "leaf_colors": [hx(c) for c in R["leaves_color_list"]]}}

    json.dump(clean(A), open(os.path.join(OUT, "analysis.json"), "w"),
              separators=(",", ":"), allow_nan=False)
    print(f"[done] analysis.json = {os.path.getsize(os.path.join(OUT,'analysis.json'))/1024:.0f} KB (joint)")


if __name__ == "__main__":
    main()
