"""Build lightweight gene-browser datasets for SAE ablation dictionaries.

Ablation dictionaries do not have the hand-labelled W512 semantic taxonomy, so
they are grouped by cross-city prevalence: 12-city core, near-core, accessory,
regional, pair-specific, and city-unique.
"""
from __future__ import annotations

import argparse
import json
import time
from dataclasses import dataclass
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.cm as cm
import numpy as np
from PIL import Image

from formal.gpu_run import imgpath


ROOT = Path(__file__).resolve().parents[2]
BASE = ROOT / "formal" / "ablation_topk_width"
SITE = ROOT / "formal" / "site"

GRID = 28
NEX = 8
THRESHOLD = 5e-4
CITIES = [
    "HongKong", "Singapore", "Amsterdam", "CapeTown", "Paris", "SaoPaulo",
    "MexicoCity", "Sydney", "Jakarta", "Dhaka", "NewDelhi", "Manila",
]
GROUPS = [
    (0, "12城共有基因", "#22e0a1", lambda n: n == 12),
    (1, "近核心 9-11城", "#39d6ff", lambda n: 9 <= n <= 11),
    (2, "辅助型 6-8城", "#7c5cff", lambda n: 6 <= n <= 8),
    (3, "区域型 3-5城", "#f6c945", lambda n: 3 <= n <= 5),
    (4, "双城特异", "#ff9e64", lambda n: n == 2),
    (5, "单城独有", "#ff6b8a", lambda n: n == 1),
]


@dataclass(frozen=True)
class Dataset:
    width: int
    topk: int
    key_override: str | None = None
    label_override: str | None = None
    sparse_override: Path | None = None
    out_dir_override: str | None = None

    @property
    def key(self) -> str:
        if self.key_override:
            return self.key_override
        return f"w{self.width}_topk{self.topk}"

    @property
    def label(self) -> str:
        if self.label_override:
            return self.label_override
        return f"W{self.width} · topk{self.topk}"

    @property
    def sparse(self) -> Path:
        if self.sparse_override:
            return self.sparse_override
        return BASE / f"width{self.width}_topk{self.topk}" / "sparse_acts.npz"

    @property
    def out(self) -> Path:
        if self.out_dir_override:
            return SITE / self.out_dir_override
        return SITE / f"genes_w{self.width}_topk{self.topk}"

    @property
    def web_dir(self) -> str:
        return self.out.name


DATASETS = {
    d.key: d for d in [
        Dataset(1024, 8),
        Dataset(1024, 16),
        Dataset(2048, 8),
        Dataset(2048, 16),
        Dataset(
            1024,
            8,
            key_override="batchtopk_w1024_k8",
            label_override="BatchTopK W1024 · K8",
            sparse_override=ROOT / "formal" / "batchtopk_w1024_k8" / "sparse_acts.npz",
            out_dir_override="genes_batchtopk_w1024_k8",
        ),
    ]
}


def log(*args: object) -> None:
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


def build_dataset(ds: Dataset, force: bool = False) -> None:
    out = ds.out
    manifest_file = out / "manifest.json"
    if manifest_file.exists() and not force:
        log(f"skip {ds.label}: {manifest_file} exists")
        return
    if not ds.sparse.exists():
        raise FileNotFoundError(ds.sparse)

    ex_root = out / "exem"
    out.mkdir(parents=True, exist_ok=True)
    ex_root.mkdir(parents=True, exist_ok=True)
    log(f"load {ds.sparse}")
    z = np.load(ds.sparse, allow_pickle=True)
    idx = z["idx"].astype(np.int64)
    val = z["val"].astype(np.float32)
    city = np.array([str(c) for c in z["city"]])
    pano = np.array([str(p) for p in z["pano"]])
    heading = z["heading"].astype(int)
    n_img, n_patch, actual_topk = idx.shape

    log(f"{ds.label}: {n_img:,} images, {n_img * n_patch:,} patches, W={ds.width}, topk={actual_topk}")
    top = idx[:, :, 0]
    profiles = np.zeros((len(CITIES), ds.width), dtype=np.float64)
    for ci, c in enumerate(CITIES):
        g = top[city == c].ravel()
        counts = np.bincount(g, minlength=ds.width).astype(np.float64)
        profiles[ci] = counts / max(float(counts.sum()), 1.0)
    prevalence = (profiles >= THRESHOLD).sum(0)

    log("compute per-image peak activations")
    maxmat = np.zeros((n_img, ds.width), dtype=np.float32)
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
        gd = ex_root / f"g{g}"
        gd.mkdir(parents=True, exist_ok=True)
        top_rows = [int(i) for i in np.argsort(-maxmat[:, g])[:NEX * 3] if maxmat[i, g] > 1e-6]
        ex = []
        for r, i in enumerate(top_rows[:NEX]):
            try:
                p = imgpath(city[i], pano[i], int(heading[i]))
                gene_map = np.where(idx[i] == g, val[i], 0).max(1).reshape(GRID, GRID)
                Image.fromarray(overlay(p, gene_map)).resize((128, 128)).save(gd / f"{r}.jpg", quality=74)
                Image.open(p).convert("RGB").resize((128, 128)).save(gd / f"{r}_o.jpg", quality=74)
                ex.append(f"{ds.web_dir}/exem/g{g}/{r}.jpg")
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
        "dataset": ds.label,
        "grouping": "cross-city prevalence",
        "threshold": THRESHOLD,
        "categories": categories,
        "taxonomy": taxonomy,
        "positional": [],
        "n_genes": ds.width,
        "n_active": len(genes),
        "n_imgs": int(n_img),
        "n_examples_per_gene": NEX,
        "genes": genes,
        "tree": tree,
    }
    manifest_file.write_text(json.dumps(manifest, ensure_ascii=False))
    log(f"[done] {ds.label}: manifest genes={len(genes):,}, groups={len(tree)}, missing_examples={missing}")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("datasets", nargs="*", choices=sorted(DATASETS), help="dataset keys to build")
    p.add_argument("--force", action="store_true", help="overwrite an existing manifest")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    keys = args.datasets or sorted(DATASETS)
    for key in keys:
        build_dataset(DATASETS[key], force=args.force)


if __name__ == "__main__":
    main()
