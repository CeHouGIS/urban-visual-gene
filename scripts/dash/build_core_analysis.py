#!/usr/bin/env python3
"""Build the INTRINSIC analytics for the dashboard 分析 page (4 cities).

Computes, per city, everything derivable from a single pipeline run (no sweep /
baseline / multi-K re-runs):
  - transition          32x32 row-normalised dominant-basis transition on edges
  - visual_syntax       self_transition, transition_entropy (from the matrix)
  - basis_roles         dominant / dead / eff_vocab_90 (90% argmax-mass)
  - cooccur             corr(A) 32x32 + hierarchical-clustering leaf order + n_families
  - basis_similarity    |XX^T| off-diag mean + effective_n_bases (participation ratio)
  - sae_metrics         mean_recon_error, median_active, mean_entropy

Writes outputs/<dashboard>/data/analysis.json  (consumed by analysis.html).

  PROJ-free; run with the conda python:
    OMP_NUM_THREADS=1 python -m scripts.dash.build_core_analysis
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

CITIES = {
    "HongKong":  "outputs/China/HongKong",
    "Singapore": "outputs/Singapore/Singapore",
    "Amsterdam": "outputs/Netherlands/Amsterdam",
    "CapeTown":  "outputs/SouthAfrica/CapeTown",
}
K = 32
ACT = [f"a_{i:03d}" for i in range(K)]


def clean(o):
    if isinstance(o, dict):  return {k: clean(v) for k, v in o.items()}
    if isinstance(o, list):  return [clean(v) for v in o]
    if isinstance(o, float) and (math.isnan(o) or math.isinf(o)): return None
    return o


def transition_and_syntax(run):
    act = pd.read_parquet(os.path.join(ROOT, run, "road_basis_activation.parquet"),
                          columns=["road_node_id"] + ACT)
    A = act[ACT].to_numpy(np.float32)
    dom = dict(zip(act["road_node_id"], A.argmax(1)))
    ed = pd.read_parquet(os.path.join(ROOT, run, "road_graph_edges.parquet"),
                         columns=["src_node_id", "dst_node_id"])
    T = np.zeros((K, K), np.float64)
    s = ed["src_node_id"].map(dom).to_numpy()
    d = ed["dst_node_id"].map(dom).to_numpy()
    ok = ~(pd.isna(s) | pd.isna(d))
    for a, b in zip(s[ok].astype(int), d[ok].astype(int)):
        T[a, b] += 1
    row = T.sum(1, keepdims=True); rown = row.copy(); rown[rown == 0] = 1
    P = T / rown
    self_t = float(np.average(np.diag(P), weights=row[:, 0]) if row.sum() else 0.0)
    # row-wise transition entropy (avg over rows that have outgoing mass), excl self
    ent = []
    for i in range(K):
        p = P[i].copy(); p[i] = 0
        z = p.sum()
        if z > 0:
            p = p / z
            ent.append(-np.sum([q * math.log(q) for q in p if q > 0]))
    trans_ent = float(np.mean(ent)) if ent else 0.0
    return P.round(5).tolist(), {"self_transition": round(self_t, 4),
                                 "transition_entropy": round(trans_ent, 4)}, A


def basis_roles(A):
    usage = np.bincount(A.argmax(1), minlength=K).astype(float)
    dead = int((usage == 0).sum())
    order = np.sort(usage)[::-1]
    csum = np.cumsum(order) / max(usage.sum(), 1)
    eff90 = int(np.searchsorted(csum, 0.90) + 1)
    return {"dominant": int(K - dead), "dead": dead, "eff_dominant_vocab_90": eff90}


def cooccur(A):
    """corr(A) over nodes + average-linkage leaf order (basis families)."""
    from scipy.cluster.hierarchy import linkage, leaves_list, fcluster
    from scipy.spatial.distance import squareform
    std = A.std(0)
    live = std > 1e-8
    C = np.zeros((K, K), np.float64)
    if live.sum() >= 2:
        Cl = np.corrcoef(A[:, live].T)
        idx = np.where(live)[0]
        for a, ia in enumerate(idx):
            for b, ib in enumerate(idx):
                C[ia, ib] = Cl[a, b]
    np.fill_diagonal(C, 1.0)
    C = np.nan_to_num(C, nan=0.0)
    d = np.clip(1.0 - C, 0, 2); d = (d + d.T) / 2; np.fill_diagonal(d, 0.0)
    try:
        Z = linkage(squareform(d, checks=False), method="average")
        order = [int(x) for x in leaves_list(Z)]
        fam = fcluster(Z, t=0.7 * d.max(), criterion="distance")
        n_fam = int(len(set(fam.tolist())))
    except Exception:
        order = list(range(K)); n_fam = 1
    dead = [int(i) for i in range(K) if not live[i]]
    return {"matrix": C.round(3).tolist(), "order": order, "n_families": n_fam, "dead": dead}


def basis_similarity(run):
    p = os.path.join(ROOT, run, "road_landscape_basis.npy")
    if not os.path.exists(p):
        return {"effective_n_bases": float(K), "mean_abs_offdiag": 0.0}
    X = np.load(p).astype(np.float64)                 # (K, D)
    Xn = X / (np.linalg.norm(X, axis=1, keepdims=True) + 1e-9)
    S = np.abs(Xn @ Xn.T)
    off = S[~np.eye(K, dtype=bool)]
    G = Xn @ Xn.T
    w = np.linalg.eigvalsh(G); w = np.clip(w, 0, None)
    eff = float((w.sum() ** 2) / (np.sum(w ** 2) + 1e-12))
    return {"effective_n_bases": round(eff, 2),
            "mean_abs_offdiag": round(float(off.mean()), 4),
            "matrix": S.round(3).tolist()}


def sae_metrics(run):
    rep = {}
    for st in ("stage4_report.json", "stage5_report.json"):
        fp = os.path.join(ROOT, run, "stage_reports", st)
        if os.path.exists(fp):
            rep.update(json.load(open(fp)))
    act = pd.read_parquet(os.path.join(ROOT, run, "road_basis_activation.parquet"),
                          columns=["activation_entropy"])
    return {
        "mean_recon_error": round(float(rep.get("mean_recon_error", rep.get("final_train_loss", 0))), 4),
        "median_active": float(rep.get("median_active_basis", 0)),
        "mean_entropy": round(float(act["activation_entropy"].mean()), 4),
    }


def main():
    out = {"cities": list(CITIES), "K": K, "transition": {}, "visual_syntax": {},
           "basis_roles": {}, "cooccur": {}, "basis_similarity": {}, "sae_metrics": {}}
    for city, run in CITIES.items():
        P, syn, A = transition_and_syntax(run)
        out["transition"][city] = P
        out["visual_syntax"][city] = syn
        out["basis_roles"][city] = basis_roles(A)
        out["cooccur"][city] = cooccur(A)
        out["basis_similarity"][city] = basis_similarity(run)
        out["sae_metrics"][city] = sae_metrics(run)
        print(f"[analysis] {city}: self_t={syn['self_transition']} ent={syn['transition_entropy']} "
              f"roles={out['basis_roles'][city]} eff_bases={out['basis_similarity'][city]['effective_n_bases']} "
              f"families={out['cooccur'][city]['n_families']}", flush=True)
    json.dump(clean(out), open(os.path.join(OUT, "analysis.json"), "w"),
              separators=(",", ":"), allow_nan=False)
    sz = os.path.getsize(os.path.join(OUT, "analysis.json"))
    print(f"[done] analysis.json = {sz/1024:.0f} KB", flush=True)


if __name__ == "__main__":
    main()
