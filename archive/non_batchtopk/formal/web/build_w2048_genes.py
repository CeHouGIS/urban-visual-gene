"""Build a lightweight gene-browser dataset for W2048 topk4.

The original genes page is built around the W512 dictionary with hand-labelled
semantic taxonomy. The ablation dictionaries do not have that taxonomy, so this
builder groups W2048 topk4 genes by cross-city prevalence:
12-city core, near-core, accessory, regional, pair-specific, city-unique.
"""
from __future__ import annotations

import json
import time
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.cm as cm
import numpy as np
from PIL import Image

from formal.gpu_run import imgpath


ROOT = Path(__file__).resolve().parents[2]
SPARSE = ROOT / "formal" / "ablation_topk_width" / "width2048_topk4" / "sparse_acts.npz"
OUT = ROOT / "formal" / "site" / "genes_w2048_topk4"
EX = OUT / "exem"

K = 2048
GRID = 28
NEX = 8
THRESHOLD = 5e-4
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
GROUPS = [
    (0, "12城共有基因", "#22e0a1", lambda n: n == 12),
    (1, "近核心 9-11城", "#39d6ff", lambda n: 9 <= n <= 11),
    (2, "辅助型 6-8城", "#7c5cff", lambda n: 6 <= n <= 8),
    (3, "区域型 3-5城", "#f6c945", lambda n: 3 <= n <= 5),
    (4, "双城特异", "#ff9e64", lambda n: n == 2),
    (5, "单城独有", "#ff6b8a", lambda n: n == 1),
]


def log(*args):
    print(f"[{time.strftime('%H:%M:%S')}]", *args, flush=True)


def overlay(img_file: Path, gene_map: np.ndarray) -> np.ndarray:
    m = gene_map.astype(np.float32)
    m = (m - m.min()) / (m.max() - m.min() + 1e-8)
    up = np.array(
        Image.fromarray((m * 255).astype(np.uint8)).resize((224, 224), Image.BILINEAR),
        np.float32,
    ) / 255
    heat = (cm.jet(up)[..., :3] * 255).astype(np.float32)
    img = np.array(Image.open(img_file).convert("RGB").resize((224, 224)), np.float32)
    alpha = (0.3 + 0.55 * up)[..., None]
    return (img * (1 - alpha) + heat * alpha).astype(np.uint8)


def group_id(prevalence: int) -> int:
    for gid, _name, _color, pred in GROUPS:
        if pred(prevalence):
            return gid
    return -1


def chunked(seq: list[int], size: int) -> list[list[int]]:
    return [seq[i:i + size] for i in range(0, len(seq), size)]


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    EX.mkdir(parents=True, exist_ok=True)
    log(f"load {SPARSE}")
    z = np.load(SPARSE, allow_pickle=True)
    idx = z["idx"].astype(np.int64)
    val = z["val"].astype(np.float32)
    city = np.array([str(c) for c in z["city"]])
    pano = np.array([str(p) for p in z["pano"]])
    heading = z["heading"].astype(int)
    n_img, n_patch, topk = idx.shape

    log(f"{n_img:,} images, {n_img * n_patch:,} patches, K={K}, topk={topk}")
    top = idx[:, :, 0]
    profiles = np.zeros((len(CITIES), K), dtype=np.float64)
    for ci, c in enumerate(CITIES):
        g = top[city == c].ravel()
        counts = np.bincount(g, minlength=K).astype(np.float64)
        profiles[ci] = counts / max(float(counts.sum()), 1.0)
    prevalence = (profiles >= THRESHOLD).sum(0)

    log("compute per-image peak activations")
    maxmat = np.zeros((n_img, K), dtype=np.float32)
    for i in range(n_img):
        np.maximum.at(maxmat[i], idx[i].ravel(), val[i].ravel())
        if i and i % 2000 == 0:
            log(f"  maxmat {i:,}/{n_img:,}")
    peak = maxmat.max(0)
    nimg = (maxmat > 0.05 * (peak + 1e-9)).sum(0)
    used = [int(g) for g in np.where(prevalence >= 1)[0]]
    log(f"used genes: {len(used):,}")

    genes = {}
    missing = 0
    for rank, g in enumerate(sorted(used, key=lambda x: (-prevalence[x], -peak[x], x))):
        gd = EX / f"g{g}"
        gd.mkdir(parents=True, exist_ok=True)
        top_rows = [int(i) for i in np.argsort(-maxmat[:, g])[:NEX * 3] if maxmat[i, g] > 1e-6]
        ex = []
        for r, i in enumerate(top_rows[:NEX]):
            try:
                p = imgpath(city[i], pano[i], int(heading[i]))
                gene_map = np.where(idx[i] == g, val[i], 0).max(1).reshape(GRID, GRID)
                Image.fromarray(overlay(p, gene_map)).resize((128, 128)).save(gd / f"{r}.jpg", quality=74)
                Image.open(p).convert("RGB").resize((128, 128)).save(gd / f"{r}_o.jpg", quality=74)
                ex.append(f"genes_w2048_topk4/exem/g{g}/{r}.jpg")
            except Exception:
                missing += 1
        if not ex:
            continue
        gid = group_id(int(prevalence[g]))
        present = [CITIES[i] for i in np.where(profiles[:, g] >= THRESHOLD)[0]]
        genes[str(g)] = {
            "id": g,
            "cat": gid,
            "peak": round(float(peak[g]), 3),
            "nimg": int(nimg[g]),
            "ex": ex,
            "child": GROUPS[gid][1] if gid >= 0 else "未使用",
            "prevalence": int(prevalence[g]),
            "present": "|".join(present),
        }
        if rank % 100 == 0:
            log(f"  rendered {rank:,}/{len(used):,}")

    categories = [{"id": gid, "name": name, "color": color} for gid, name, color, _ in GROUPS]
    taxonomy = []
    tree = []
    for gid, name, color, _pred in GROUPS:
        gids = [int(k) for k, g in genes.items() if g["cat"] == gid]
        gids.sort(key=lambda g: (-genes[str(g)]["peak"], g))
        if not gids:
            continue
        children = []
        for ci, part in enumerate(chunked(gids, 60), start=1):
            lo = (ci - 1) * 60 + 1
            hi = lo + len(part) - 1
            children.append({
                "cat": gid,
                "color": color,
                "name": f"{name} · {lo}-{hi}",
                "n": len(part),
                "genes": part,
            })
        tree.append({
            "cat": gid,
            "color": color,
            "name": name,
            "n": len(gids),
            "children": children,
        })
        taxonomy.append({
            "id": gid,
            "name": name,
            "color": color,
            "n": len(gids),
            "children": [{"id": gid * 100, "name": name, "n": len(gids)}],
        })

    manifest = {
        "dataset": "W2048 · topk4",
        "grouping": "cross-city prevalence",
        "threshold": THRESHOLD,
        "categories": categories,
        "taxonomy": taxonomy,
        "positional": [],
        "n_genes": K,
        "n_active": len(genes),
        "n_imgs": int(n_img),
        "n_examples_per_gene": NEX,
        "genes": genes,
        "tree": tree,
    }
    (OUT / "manifest.json").write_text(json.dumps(manifest, ensure_ascii=False))
    log(f"[done] manifest genes={len(genes):,}, groups={len(tree)}, missing_examples={missing}")


if __name__ == "__main__":
    main()
