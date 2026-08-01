"""Build city-genome specificity browser data for an SAE sparse-activation set."""

import argparse
import csv
import json
import math
import time
from pathlib import Path
from typing import Dict, List, Tuple

import matplotlib

matplotlib.use("Agg")
import matplotlib.cm as cm
import numpy as np
from PIL import Image


ROOT = Path(__file__).resolve().parents[2]
SITE = ROOT / "formal" / "site"
IMROOT = Path("/global/scratch/users/cehou/data/SVIs/GSV/images")
GRID = 28
THRESHOLD = 5e-4

CITIES = [
    "HongKong",
    "Singapore",
    "Amsterdam",
    "CapeTown",
    "Paris",
    "SaoPaulo",
    "MexicoCity",
    "Sydney",
    "Jakarta",
    "Dhaka",
    "NewDelhi",
    "Manila",
]
CITY_ZH = {
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
CITY_DIR = {
    "HongKong": "China/HongKong",
    "Singapore": "Singapore/Singapore",
    "Amsterdam": "Netherlands/Amsterdam",
    "CapeTown": "SouthAfrica/CapeTown",
    "Paris": "France/Paris",
    "SaoPaulo": "Brazil/SaoPaulo",
    "MexicoCity": "Mexico/MexicoCity",
    "Sydney": "Australia/Sydney",
    "Jakarta": "Indonesia/Jakarta",
    "Dhaka": "Bangladesh/Dhaka",
    "NewDelhi": "India/NewDelhi",
    "Manila": "Philippines/Manila",
}
GROUPS = [
    ("core_universal_12cities", "12城共有基因", lambda n: n == 12),
    ("near_core_9to11cities", "近核心 9-11城", lambda n: 9 <= n <= 11),
    ("accessory_6to8cities", "辅助型 6-8城", lambda n: 6 <= n <= 8),
    ("regional_3to5cities", "区域型 3-5城", lambda n: 3 <= n <= 5),
    ("pair_specific_2cities", "双城特异", lambda n: n == 2),
    ("city_unique_1city", "单城独有", lambda n: n == 1),
]


def log(*args: object) -> None:
    print(f"[{time.strftime('%H:%M:%S')}]", *args, flush=True)


def imgpath(city: str, pano_id: str, heading: int) -> Path:
    return (
        IMROOT
        / CITY_DIR[city]
        / pano_id[0].lower()
        / pano_id[1].lower()
        / pano_id[2].lower()
        / f"{pano_id}_{heading}.jpg"
    )


def class_for(prevalence: int) -> str:
    for key, _label, pred in GROUPS:
        if pred(prevalence):
            return key
    return "unused"


def label_for(cls: str) -> str:
    for key, label, _pred in GROUPS:
        if key == cls:
            return label
    return "未使用"


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


def city_entropy(shares: np.ndarray) -> Tuple[float, float]:
    total = float(shares.sum())
    if total <= 0:
        return 0.0, 0.0
    p = shares / total
    p = p[p > 0]
    h = float(-(p * np.log2(p)).sum())
    specificity = 1.0 - h / math.log2(len(CITIES))
    return h, max(0.0, min(1.0, specificity))


def load_manifest(path: Path) -> Tuple[Dict[int, dict], Dict[int, str]]:
    manifest = json.loads(path.read_text())
    genes = {int(k): v for k, v in manifest.get("genes", {}).items()}
    cats = {int(c["id"]): c["name"] for c in manifest.get("categories", [])}
    return genes, cats


def row_for_gene(
    gene: int,
    profiles: np.ndarray,
    prevalence: np.ndarray,
    global_mean: np.ndarray,
    manifest_genes: Dict[int, dict],
    cat_names: Dict[int, str],
) -> Dict[str, object]:
    shares = profiles[:, gene]
    present_idx = np.where(shares >= THRESHOLD)[0]
    top_city_idx = int(np.argmax(shares))
    cls = class_for(int(prevalence[gene]))
    mg = manifest_genes.get(gene, {})
    category = cat_names.get(int(mg.get("cat", -99)), label_for(cls))
    h, spec = city_entropy(shares)
    row = {
        "gene": gene,
        "class": cls,
        "prevalence_n_cities": int(prevalence[gene]),
        "category": category,
        "label": mg.get("child") or category or f"Gene #{gene}",
        "top_city": CITIES[top_city_idx],
        "top_city_zh": CITY_ZH[CITIES[top_city_idx]],
        "top_share": float(shares[top_city_idx]),
        "top_lift_vs_global_mean": float(shares[top_city_idx] / max(global_mean[gene], 1e-12)),
        "min_share_across_12": float(shares.min()),
        "max_share": float(shares.max()),
        "city_specificity_0to1": spec,
        "city_entropy_bits": h,
        "present_cities_by_share": "|".join(CITIES[i] for i in present_idx),
    }
    for ci, city in enumerate(CITIES):
        row[f"share_{city}"] = float(shares[ci])
    for ci, city in enumerate(CITIES):
        row[f"lift_{city}"] = float(shares[ci] / max(global_mean[gene], 1e-12))
    return row


def write_csv(path: Path, rows: List[Dict[str, object]]) -> None:
    fields = [
        "gene",
        "class",
        "prevalence_n_cities",
        "category",
        "label",
        "top_city",
        "top_city_zh",
        "top_share",
        "top_lift_vs_global_mean",
        "min_share_across_12",
        "max_share",
        "city_specificity_0to1",
        "city_entropy_bits",
        "present_cities_by_share",
    ]
    fields += [f"share_{c}" for c in CITIES] + [f"lift_{c}" for c in CITIES]
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        for r in rows:
            writer.writerow(r)


def render_examples(
    out_dir: Path,
    web_dir: str,
    rows: list[dict[str, object]],
    idx: np.ndarray,
    val: np.ndarray,
    city: np.ndarray,
    pano: np.ndarray,
    heading: np.ndarray,
    maxmat: np.ndarray,
    force: bool,
) -> Dict[str, List[Dict[str, object]]]:
    ex_root = out_dir / web_dir
    ex_root.mkdir(parents=True, exist_ok=True)
    examples = {}
    city_indices = {c: np.where(city == c)[0] for c in CITIES}
    missing = 0
    for n, row in enumerate(rows, start=1):
        gene = int(row["gene"])
        if row["class"] == "unused":
            continue
        gd = ex_root / f"g{gene}"
        gd.mkdir(parents=True, exist_ok=True)
        items = []
        for c in CITIES:
            ci = city_indices[c]
            best_local = int(ci[np.argmax(maxmat[ci, gene])])
            orig = gd / f"{c}_orig.jpg"
            act = gd / f"{c}_act.jpg"
            score = float(maxmat[best_local, gene])
            if force or not (orig.exists() and act.exists()):
                try:
                    p = imgpath(str(city[best_local]), str(pano[best_local]), int(heading[best_local]))
                    gene_map = np.where(idx[best_local] == gene, val[best_local], 0).max(1).reshape(GRID, GRID)
                    Image.fromarray(overlay(p, gene_map)).resize((128, 128)).save(act, quality=74)
                    Image.open(p).convert("RGB").resize((128, 128)).save(orig, quality=74)
                except Exception:
                    missing += 1
                    continue
            items.append(
                {
                    "city": c,
                    "zh": CITY_ZH[c],
                    "orig": f"citygenome/{web_dir}/g{gene}/{c}_orig.jpg",
                    "act": f"citygenome/{web_dir}/g{gene}/{c}_act.jpg",
                    "score": round(score, 4),
                }
            )
        if items:
            examples[str(gene)] = items
        if n % 50 == 0:
            log(f"rendered city examples {n:,}/{len(rows):,}")
    log(f"city examples missing={missing}")
    return examples


def build(args: argparse.Namespace) -> None:
    sparse = Path(args.sparse)
    manifest = Path(args.manifest)
    out_dir = SITE / "citygenome"
    out_dir.mkdir(parents=True, exist_ok=True)

    z = np.load(sparse, mmap_mode="r")
    idx = z["idx"].astype(np.int64)
    val = z["val"].astype(np.float32)
    city = np.array([str(c) for c in z["city"]])
    pano = np.array([str(p) for p in z["pano"]])
    heading = z["heading"].astype(int)
    width = int(args.width or idx.max() + 1)
    n_img, n_patch, topk = idx.shape
    log(f"loaded {sparse}: {n_img:,} images, {n_patch:,} patches/image, W={width}, topk={topk}")

    profiles = np.zeros((len(CITIES), width), dtype=np.float64)
    top = idx[:, :, 0]
    for ci, c in enumerate(CITIES):
        g = top[city == c].ravel()
        counts = np.bincount(g, minlength=width).astype(np.float64)
        profiles[ci] = counts / max(float(counts.sum()), 1.0)
    prevalence = (profiles >= THRESHOLD).sum(0)
    global_mean = profiles.mean(0)

    log("computing per-image peak activation matrix")
    maxmat = np.zeros((n_img, width), dtype=np.float32)
    for i in range(n_img):
        np.maximum.at(maxmat[i], idx[i].ravel(), val[i].ravel())
        if i and i % 2000 == 0:
            log(f"  maxmat {i:,}/{n_img:,}")

    manifest_genes, cat_names = load_manifest(manifest)
    rows = [
        row_for_gene(g, profiles, prevalence, global_mean, manifest_genes, cat_names)
        for g in range(width)
    ]
    rows.sort(key=lambda r: int(r["gene"]))
    prefix = args.prefix
    csv_path = out_dir / f"gene_city_specificity_{prefix}.csv"
    summary_path = out_dir / f"gene_city_specificity_summary_{prefix}.json"
    examples_path = out_dir / f"city_gene_examples_{prefix}.json"
    examples_dir = f"city_gene_examples_{prefix}"

    write_csv(csv_path, rows)
    used_rows = [r for r in rows if r["class"] != "unused"]
    examples = render_examples(
        out_dir,
        examples_dir,
        used_rows,
        idx,
        val,
        city,
        pano,
        heading,
        maxmat,
        force=args.force_examples,
    )

    def by_min_share(r):
        return (-float(r["min_share_across_12"]), int(r["gene"]))

    def by_top_share(r):
        return (-float(r["top_share"]), int(r["gene"]))

    class_counts = {cls: 0 for cls, _label, _pred in GROUPS}
    class_counts["unused"] = 0
    for r in rows:
        class_counts[str(r["class"])] += 1

    city_markers = {}
    for c in CITIES:
        marker_rows = [
            r
            for r in used_rows
            if float(r[f"share_{c}"]) >= THRESHOLD
        ]
        marker_rows.sort(key=lambda r: (-float(r[f"lift_{c}"]), -float(r[f"share_{c}"]), int(r["gene"])))
        city_markers[c] = marker_rows[:8]

    summary = {
        "dataset": args.dataset_label,
        "threshold_present_share": THRESHOLD,
        "n_imgs": int(n_img),
        "n_patches": int(n_img * n_patch),
        "width": width,
        "topk": int(topk),
        "cities": CITIES,
        "city_zh": CITY_ZH,
        "class_counts": class_counts,
        "core_universal_top_by_min_share": sorted(
            [r for r in rows if r["class"] == "core_universal_12cities"], key=by_min_share
        )[:60],
        "city_unique_all": sorted(
            [r for r in rows if r["class"] == "city_unique_1city"], key=by_top_share
        ),
        "pair_specific_top_by_share": sorted(
            [r for r in rows if r["class"] == "pair_specific_2cities"], key=by_top_share
        )[:80],
        "regional_3to5_top_by_share": sorted(
            [r for r in rows if r["class"] == "regional_3to5cities"], key=by_top_share
        )[:100],
        "city_markers_top8": city_markers,
    }
    summary_path.write_text(json.dumps(summary, ensure_ascii=False))
    examples_path.write_text(
        json.dumps({"cities": CITIES, "city_zh": CITY_ZH, "examples": examples}, ensure_ascii=False)
    )
    log(f"wrote {csv_path}")
    log(f"wrote {summary_path}")
    log(f"wrote {examples_path} with {len(examples):,} genes")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--sparse", required=True)
    p.add_argument("--manifest", required=True)
    p.add_argument("--prefix", required=True)
    p.add_argument("--dataset-label", required=True)
    p.add_argument("--width", type=int, default=0)
    p.add_argument("--force-examples", action="store_true")
    return p.parse_args()


if __name__ == "__main__":
    build(parse_args())
