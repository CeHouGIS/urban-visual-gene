"""Build city diversity ablation data for the citygenome web page.

The web figure compares Shannon diversity under several SAE parameter
settings. Diversity is computed from each city's dominant gene profile:
for every image patch, use the strongest latent dimension idx[..., 0].
"""
from __future__ import annotations

import json
import math
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[2]
SITE = ROOT / "formal" / "site" / "citygenome"
BASELINE = ROOT / "formal" / "site" / "citygenome" / "data.json"
K_SWEEP = ROOT / "formal" / "k_sweep_shared" / "comparison.json"
TOPK_WIDTH = ROOT / "formal" / "ablation_topk_width"
TOPK_WIDTH_COMPARISON = TOPK_WIDTH / "comparison.json"
OUT = SITE / "diversity_ablation.json"

CITIES = [
    "HongKong", "Singapore", "Amsterdam", "CapeTown", "Paris", "SaoPaulo",
    "MexicoCity", "Sydney", "Jakarta", "Dhaka", "NewDelhi", "Manila",
]
ZH = {
    "HongKong": "香港",
    "Singapore": "新加坡",
    "Amsterdam": "阿姆斯特丹",
    "CapeTown": "开普敦",
    "Paris": "巴黎",
    "SaoPaulo": "圣保罗",
    "MexicoCity": "墨西哥城",
    "Sydney": "悉尼",
    "Jakarta": "雅加达",
    "Dhaka": "达卡",
    "NewDelhi": "新德里",
    "Manila": "马尼拉",
}


def entropy_eff(profile: np.ndarray) -> tuple[float, float, int]:
    nz = profile[profile > 0]
    if len(nz) == 0:
        return 0.0, 0.0, 0
    h = float(-(nz * np.log2(nz)).sum())
    return h, float(2 ** h), int((profile > 0).sum())


def diversity_from_sparse(path: Path, width: int) -> list[dict]:
    z = np.load(path, allow_pickle=True)
    idx = z["idx"][:, :, 0].astype(np.int64)
    city = np.array([str(c) for c in z["city"]])
    rows = []
    for c in CITIES:
        g = idx[city == c].ravel()
        counts = np.bincount(g, minlength=width).astype(np.float64)
        profile = counts / max(float(counts.sum()), 1.0)
        h, eff, richness = entropy_eff(profile)
        rows.append({
            "city": c,
            "zh": ZH[c],
            "H": round(h, 4),
            "eff": round(eff, 2),
            "richness": richness,
        })
    return rows


def diversity_from_existing(rows: list[dict]) -> list[dict]:
    by_city = {r["city"]: r for r in rows}
    out = []
    for c in CITIES:
        r = by_city[c]
        out.append({
            "city": c,
            "zh": r.get("zh", ZH[c]),
            "H": round(float(r["H"]), 4),
            "eff": round(float(r["eff"]), 2),
            "richness": int(r.get("richness", 0)),
        })
    return out


def add_summary(row: dict) -> dict:
    values = [float(d["eff"]) for d in row["diversity"]]
    row["summary"] = {
        "mean_eff": round(float(np.mean(values)), 2),
        "min_eff": round(float(np.min(values)), 2),
        "max_eff": round(float(np.max(values)), 2),
        "spread_eff": round(float(np.max(values) - np.min(values)), 2),
    }
    return row


def core_count(summary: dict) -> int:
    return int(summary["class_counts"]["core_universal_12cities"])


def main() -> None:
    configs = []

    base = json.loads(BASELINE.read_text())
    configs.append(add_summary({
        "id": "k512_topk32",
        "label": "K512 · topk32",
        "group": "current",
        "width": 512,
        "topk": 32,
        "note": "当前网页主结果",
        "core_12": int(base["pangenome"]["n_core"]),
        "diversity": diversity_from_existing(base["diversity"]),
    }))

    if K_SWEEP.exists():
        sweep = json.loads(K_SWEEP.read_text())
        for k in ["128", "256"]:
            r = sweep["results"][k]
            configs.append(add_summary({
                "id": f"k{k}_topk32",
                "label": f"K{k} · topk32",
                "group": "width sweep",
                "width": int(k),
                "topk": 32,
                "note": "较小字典宽度",
                "core_12": core_count(r),
                "diversity": diversity_from_existing(r["diversity"]),
            }))

    tw_summary = json.loads(TOPK_WIDTH_COMPARISON.read_text()) if TOPK_WIDTH_COMPARISON.exists() else {"results": {}}
    for width in [1024, 2048]:
        for topk in [4, 8, 16]:
            sp = TOPK_WIDTH / f"width{width}_topk{topk}" / "sparse_acts.npz"
            if not sp.exists():
                continue
            cfg_id = f"width{width}_topk{topk}"
            summary = tw_summary["results"].get(cfg_id, {})
            configs.append(add_summary({
                "id": cfg_id,
                "label": f"W{width} · topk{topk}",
                "group": "topk/width",
                "width": width,
                "topk": topk,
                "note": "新消融",
                "core_12": core_count(summary) if summary else None,
                "diversity": diversity_from_sparse(sp, width),
            }))

    values = [d["eff"] for cfg in configs for d in cfg["diversity"]]
    out = {
        "metric": "effective_genes",
        "method": "For each city, count dominant gene idx[...,0] over all patches, compute Shannon H and effective genes 2^H.",
        "cities": [{"city": c, "zh": ZH[c]} for c in CITIES],
        "scale": {
            "min_eff": math.floor(min(values)),
            "max_eff": math.ceil(max(values)),
        },
        "configs": configs,
    }
    OUT.write_text(json.dumps(out, ensure_ascii=False, indent=2))
    print(f"wrote {OUT} ({len(configs)} configs)")


if __name__ == "__main__":
    main()
