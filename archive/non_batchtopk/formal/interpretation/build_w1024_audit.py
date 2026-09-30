"""Build a semantic-audit dataset for BatchTopK W1024/K8.

The output separates model diagnostics, provisional labels transferred from the
older W512 dictionary, and human annotations. Diagnostics cover all 1024 genes;
image evidence is rendered for a stratified 128-gene pilot. Model artifacts are
read-only.
"""
from __future__ import annotations

import argparse
import csv
import json
import math
import time
from collections import defaultdict
from pathlib import Path
from typing import Any, Iterable

import matplotlib

matplotlib.use("Agg")
import matplotlib.cm as cm
import numpy as np
import torch
from PIL import Image

from formal.gpu_run import imgpath


ROOT = Path(__file__).resolve().parents[2]
RUN = ROOT / "formal" / "batchtopk_w1024_k8"
OLD = ROOT / "formal" / "formal_out_global3"
DEFAULT_OUT = ROOT / "formal" / "site" / "w1024_audit"
CITY_ORDER = [
    "HongKong", "Singapore", "Amsterdam", "CapeTown", "Paris", "SaoPaulo",
    "MexicoCity", "Sydney", "Jakarta", "Dhaka", "NewDelhi", "Manila",
]
PREVALENCE_THRESHOLD = 5e-4
ANNOTATION_FIELDS = [
    "gene_id", "status", "urban_relevance", "primary_concept", "concept_type",
    "scope", "is_a", "part_of", "attributes", "alternative_hypotheses",
    "confidence", "reviewer", "reviewer_notes",
]


def log(*args: object) -> None:
    print(f"[{time.strftime('%H:%M:%S')}]", *args, flush=True)


def prevalence_group(n: int) -> str:
    if n == 0:
        return "below_threshold"
    if n == 1:
        return "city_unique"
    if n == 2:
        return "pair_specific"
    if n <= 5:
        return "regional_3to5"
    if n <= 8:
        return "accessory_6to8"
    if n <= 11:
        return "near_core_9to11"
    return "core_12"


def old_match_group(similarity: float) -> str:
    if similarity >= 0.75:
        return "strong"
    if similarity >= 0.65:
        return "medium"
    return "weak"


def decoder_atoms(checkpoint: Path) -> np.ndarray:
    meta = torch.load(checkpoint, map_location="cpu")
    state = meta["state"]
    for key in ("decoder.weight", "dec.weight"):
        if key in state:
            atoms = state[key].detach().float().cpu().numpy().T
            return atoms / (np.linalg.norm(atoms, axis=1, keepdims=True) + 1e-9)
    raise KeyError(f"no decoder weight in {checkpoint}")


def quartile_codes(values: Iterable[float]) -> np.ndarray:
    values = np.asarray(list(values), dtype=np.float64)
    cuts = np.quantile(values, [0.25, 0.50, 0.75])
    return np.searchsorted(cuts, values, side="right").astype(np.int8)


def select_stratified_pilot(rows: list[dict[str, Any]], target: int, seed: int) -> list[int]:
    """Round-robin strata so rare and low-confidence genes remain visible."""
    rng = np.random.default_rng(seed)
    support_q = quartile_codes(np.log1p([r["support_all_slots"] for r in rows]))
    strata: dict[tuple[Any, ...], list[int]] = defaultdict(list)
    for row, quartile in zip(rows, support_q):
        key = (
            row["prevalence_group"], row["old_match_group"], int(quartile),
            bool(row["position_r2_top1"] >= 0.05),
        )
        strata[key].append(int(row["gene_id"]))
    for ids in strata.values():
        rng.shuffle(ids)

    selected = [
        int(row["gene_id"])
        for row in sorted(rows, key=lambda x: (-x["position_r2_top1"], x["gene_id"]))[:8]
    ]
    selected = selected[:target]
    seen = set(selected)
    keys = sorted(strata, key=lambda key: tuple(str(v) for v in key))
    while len(selected) < min(target, len(rows)):
        progressed = False
        for key in keys:
            while strata[key] and strata[key][-1] in seen:
                strata[key].pop()
            if not strata[key]:
                continue
            gene = strata[key].pop()
            selected.append(gene)
            seen.add(gene)
            progressed = True
            if len(selected) >= target:
                break
        if not progressed:
            break
    return selected


def compute_maxmat(idx: np.ndarray, val: np.ndarray, width: int) -> np.ndarray:
    maxmat = np.zeros((idx.shape[0], width), dtype=np.float32)
    for row in range(idx.shape[0]):
        np.maximum.at(maxmat[row], idx[row].ravel(), val[row].ravel())
        if row and row % 2000 == 0:
            log(f"  per-image maxima {row:,}/{idx.shape[0]:,}")
    return maxmat


def position_diagnostics(top: np.ndarray, width: int) -> dict[str, np.ndarray]:
    """Position-only R2 on binary top-1 assignments plus spatial summaries."""
    n_img, n_patch = top.shape
    grid = int(round(math.sqrt(n_patch)))
    if grid * grid != n_patch:
        raise ValueError(f"patch count {n_patch} is not square")
    counts = np.zeros((width, n_patch), dtype=np.float64)
    patch_ids = np.arange(n_patch, dtype=np.int64)
    for start in range(0, n_img, 500):
        part = top[start:start + 500]
        keys = part.ravel() * n_patch + np.tile(patch_ids, len(part))
        counts += np.bincount(keys, minlength=width * n_patch).reshape(width, n_patch)

    total = counts.sum(1)
    ss_res = total - (counts * counts).sum(1) / n_img
    ss_tot = total - total * total / (n_img * n_patch)
    r2 = np.where(ss_tot > 0, 1 - ss_res / np.maximum(ss_tot, 1e-12), 0.0)
    r2 = np.clip(r2, 0, 1)
    maps = counts.reshape(width, grid, grid)
    denom = maps.sum((1, 2)) + 1e-9
    border = np.zeros((grid, grid), dtype=bool)
    border[0] = border[-1] = True
    border[:, 0] = border[:, -1] = True
    corner = np.zeros((grid, grid), dtype=bool)
    corner[:3, :3] = corner[:3, -3:] = True
    corner[-3:, :3] = corner[-3:, -3:] = True
    row_energy = maps.sum(2) / denom[:, None]
    col_energy = maps.sum(1) / denom[:, None]
    return {
        "r2": r2,
        "maps": maps,
        "border_fraction": maps[:, border].sum(1) / denom,
        "corner_fraction": maps[:, corner].sum(1) / denom,
        "top_quarter_fraction": maps[:, :grid // 4].sum((1, 2)) / denom,
        "bottom_quarter_fraction": maps[:, -grid // 4:].sum((1, 2)) / denom,
        "row_concentration": np.sort(row_energy, axis=1)[:, -3:].sum(1),
        "column_concentration": np.sort(col_energy, axis=1)[:, -3:].sum(1),
    }


def normalized_entropy(values: np.ndarray) -> float:
    values = np.asarray(values, dtype=np.float64)
    probability = values / max(float(values.sum()), 1e-12)
    probability = probability[probability > 0]
    if len(probability) <= 1:
        return 0.0
    return float(-(probability * np.log(probability)).sum() / np.log(len(values)))


def load_old_taxonomy() -> tuple[np.ndarray, np.ndarray, dict[int, str], dict[int, str]]:
    gene_parent = np.load(OLD / "gene2cat.npy").astype(int)
    gene_child = np.load(OLD / "gene2child.npy").astype(int)
    taxonomy = json.loads((OLD / "taxonomy.json").read_text())
    parents = {int(p["id"]): str(p["name"]) for p in taxonomy["parents"]}
    children = {int(c["id"]): str(c["name"]) for c in taxonomy["children"]}
    return gene_parent, gene_child, parents, children


def automatic_review_state(row: dict[str, Any]) -> str:
    if row["support_all_slots"] == 0:
        return "inactive"
    if row["position_r2_top1"] >= 0.35:
        return "likely_positional_artifact"
    if row["position_r2_top1"] >= 0.05 or row["border_fraction"] >= 0.40:
        return "review_artifact_candidate"
    if row["prevalence"] == 0:
        return "low_prevalence_review"
    if row["old_match_cosine"] >= 0.75:
        return "strong_old_label_candidate"
    return "semantic_review"


def build_diagnostics(
    idx: np.ndarray,
    val: np.ndarray,
    city: np.ndarray,
    checkpoint: Path,
    old_checkpoint: Path,
) -> tuple[list[dict[str, Any]], np.ndarray, dict[str, np.ndarray]]:
    n_img, n_patch, _ = idx.shape
    atoms = decoder_atoms(checkpoint)
    old_atoms = decoder_atoms(old_checkpoint)
    width = atoms.shape[0]
    if idx.max() >= width:
        raise ValueError("sparse activation ids exceed checkpoint width")

    flat_val = val.ravel()
    flat_idx = idx.ravel()
    support_all = np.bincount(flat_idx[flat_val > 0], minlength=width)
    top = idx[:, :, 0]
    support_top1 = np.bincount(top.ravel(), minlength=width)
    profiles = np.zeros((len(CITY_ORDER), width), dtype=np.float64)
    for city_id, name in enumerate(CITY_ORDER):
        ids = top[city == name].ravel()
        profiles[city_id] = np.bincount(ids, minlength=width) / max(float(len(ids)), 1.0)
    prevalence = (profiles >= PREVALENCE_THRESHOLD).sum(0)

    log("compute per-image activation maxima")
    maxmat = compute_maxmat(idx, val, width)
    peak = maxmat.max(0)
    n_images_positive = (maxmat > 0).sum(0)
    n_images_strong = (maxmat > 0.05 * (peak + 1e-9)).sum(0)
    log("compute top-1 position diagnostics")
    pos = position_diagnostics(top, width)

    old_sim = np.clip(atoms @ old_atoms.T, -1, 1)
    old_id = old_sim.argmax(1)
    old_score = old_sim.max(1)
    old_parent, old_child, parent_names, child_names = load_old_taxonomy()
    nearest_id = np.zeros(width, dtype=int)
    nearest_score = np.zeros(width, dtype=np.float32)
    for start in range(0, width, 128):
        similarity = np.clip(atoms[start:start + 128] @ atoms.T, -1, 1)
        local = np.arange(len(similarity))
        similarity[local, start + local] = -2
        nearest_id[start:start + len(similarity)] = similarity.argmax(1)
        nearest_score[start:start + len(similarity)] = similarity.max(1)

    rows: list[dict[str, Any]] = []
    for gene in range(width):
        city_values = profiles[:, gene]
        present = [CITY_ORDER[i] for i in np.where(city_values >= PREVALENCE_THRESHOLD)[0]]
        top_city = int(city_values.argmax())
        old_gene = int(old_id[gene])
        row = {
            "gene_id": gene,
            "support_all_slots": int(support_all[gene]),
            "support_top1_patches": int(support_top1[gene]),
            "support_top1_fraction": round(float(support_top1[gene] / (n_img * n_patch)), 8),
            "n_images_positive": int(n_images_positive[gene]),
            "n_images_strong": int(n_images_strong[gene]),
            "peak_activation": round(float(peak[gene]), 6),
            "prevalence": int(prevalence[gene]),
            "prevalence_group": prevalence_group(int(prevalence[gene])),
            "present_cities": present,
            "top_city": CITY_ORDER[top_city],
            "top_city_fraction": round(float(city_values[top_city]), 8),
            "city_entropy": round(normalized_entropy(city_values), 6),
            "city_profile": {CITY_ORDER[i]: round(float(city_values[i]), 8) for i in range(len(CITY_ORDER))},
            "position_r2_top1": round(float(pos["r2"][gene]), 6),
            "border_fraction": round(float(pos["border_fraction"][gene]), 6),
            "corner_fraction": round(float(pos["corner_fraction"][gene]), 6),
            "top_quarter_fraction": round(float(pos["top_quarter_fraction"][gene]), 6),
            "bottom_quarter_fraction": round(float(pos["bottom_quarter_fraction"][gene]), 6),
            "row_concentration": round(float(pos["row_concentration"][gene]), 6),
            "column_concentration": round(float(pos["column_concentration"][gene]), 6),
            "old_gene_id": old_gene,
            "old_match_cosine": round(float(old_score[gene]), 6),
            "old_match_group": old_match_group(float(old_score[gene])),
            "old_parent_candidate": parent_names[int(old_parent[old_gene])],
            "old_child_candidate": child_names[int(old_child[old_gene])],
            "nearest_new_gene": int(nearest_id[gene]),
            "nearest_new_cosine": round(float(nearest_score[gene]), 6),
        }
        row["automatic_review_state"] = automatic_review_state(row)
        rows.append(row)
    return rows, maxmat, pos


def choose_examples(maxmat: np.ndarray, city: np.ndarray, gene: int, seed: int) -> list[tuple[int, str]]:
    scores = maxmat[:, gene]
    positive = np.where(scores > 0)[0]
    if len(positive) == 0:
        return []
    chosen: list[tuple[int, str]] = []
    seen: set[int] = set()

    def add(items: Iterable[int], kind: str, limit: int) -> None:
        count = 0
        for item in items:
            row = int(item)
            if row in seen:
                continue
            chosen.append((row, kind))
            seen.add(row)
            count += 1
            if count >= limit:
                break

    ranked = positive[np.argsort(-scores[positive])]
    add(ranked, "top", 8)
    city_rank = sorted(
        CITY_ORDER, key=lambda name: -float(scores[city == name].max(initial=0)),
    )
    balanced = []
    for name in city_rank:
        candidates = np.where((city == name) & (scores > 0))[0]
        if len(candidates):
            balanced.append(int(candidates[np.argmax(scores[candidates])]))
    add(balanced, "cross_city", 6)
    rng = np.random.default_rng(seed + gene * 1009)
    strong_pool = positive[scores[positive] >= 0.10 * scores[positive].max()]
    if len(strong_pool):
        add(rng.permutation(strong_pool), "random_positive", 6)
    ordered = positive[np.argsort(scores[positive])]
    lo = int(0.05 * len(ordered))
    hi = max(lo + 1, int(0.30 * len(ordered)))
    add(rng.permutation(ordered[lo:hi]), "borderline_positive", 4)
    return chosen


def activation_map(idx_row: np.ndarray, val_row: np.ndarray, gene: int) -> np.ndarray:
    patch = np.where(idx_row == gene, val_row, 0).max(1)
    grid = int(round(math.sqrt(len(patch))))
    return patch.reshape(grid, grid)


def overlay(image: Image.Image, gene_map: np.ndarray, size: int = 192) -> Image.Image:
    values = gene_map.astype(np.float32)
    values = (values - values.min()) / (values.max() - values.min() + 1e-8)
    up = np.asarray(
        Image.fromarray((values * 255).astype(np.uint8)).resize((size, size), Image.BILINEAR),
        dtype=np.float32,
    ) / 255
    heat = (cm.jet(up)[..., :3] * 255).astype(np.float32)
    rgb = np.asarray(image.convert("RGB").resize((size, size)), dtype=np.float32)
    alpha = (0.22 + 0.63 * up)[..., None]
    return Image.fromarray((rgb * (1 - alpha) + heat * alpha).astype(np.uint8))


def render_position_map(values: np.ndarray, output: Path) -> None:
    norm = values.astype(np.float32)
    norm = (norm - norm.min()) / (norm.max() - norm.min() + 1e-8)
    rgb = (cm.viridis(norm)[..., :3] * 255).astype(np.uint8)
    Image.fromarray(rgb).resize((192, 192), Image.NEAREST).save(output, quality=85)


def render_pilot(
    pilot: list[int], rows_by_gene: dict[int, dict[str, Any]], maxmat: np.ndarray,
    pos_maps: np.ndarray, idx: np.ndarray, val: np.ndarray, city: np.ndarray,
    pano: np.ndarray, heading: np.ndarray, output: Path, seed: int,
) -> dict[str, list[dict[str, Any]]]:
    exemplar_manifest: dict[str, list[dict[str, Any]]] = {}
    ex_root = output / "exem"
    ex_root.mkdir(parents=True, exist_ok=True)
    for rank, gene in enumerate(pilot):
        gene_dir = ex_root / f"g{gene}"
        gene_dir.mkdir(parents=True, exist_ok=True)
        examples = []
        for ex_rank, (row_id, kind) in enumerate(choose_examples(maxmat, city, gene, seed)):
            try:
                source = imgpath(str(city[row_id]), str(pano[row_id]), int(heading[row_id]))
                overlay_file = gene_dir / f"{ex_rank:02d}.jpg"
                original_file = gene_dir / f"{ex_rank:02d}_o.jpg"
                if not (overlay_file.exists() and original_file.exists()):
                    original = Image.open(source).convert("RGB")
                    gene_map = activation_map(idx[row_id], val[row_id], gene)
                    overlay(original, gene_map).save(overlay_file, quality=78)
                    original.resize((192, 192)).save(original_file, quality=78)
                examples.append({
                    "rank": ex_rank,
                    "kind": kind,
                    "city": str(city[row_id]),
                    "pano": str(pano[row_id]),
                    "heading": int(heading[row_id]),
                    "image_activation": round(float(maxmat[row_id, gene]), 6),
                    "overlay": f"w1024_audit/exem/g{gene}/{overlay_file.name}",
                    "original": f"w1024_audit/exem/g{gene}/{original_file.name}",
                })
            except Exception as exc:
                log(f"  skip image gene={gene} row={row_id}: {exc}")
        position_file = gene_dir / "position.jpg"
        render_position_map(pos_maps[gene], position_file)
        rows_by_gene[gene]["position_map"] = f"w1024_audit/exem/g{gene}/position.jpg"
        exemplar_manifest[str(gene)] = examples
        if rank % 16 == 0:
            log(f"  rendered pilot {rank}/{len(pilot)}")
    return exemplar_manifest


def write_csv(rows: list[dict[str, Any]], output: Path) -> None:
    flat_rows = []
    for row in rows:
        flat = dict(row)
        flat["present_cities"] = "|".join(row["present_cities"])
        flat["city_profile"] = json.dumps(row["city_profile"], ensure_ascii=False)
        flat_rows.append(flat)
    fields = list(flat_rows[0])
    fields.extend(sorted({key for row in flat_rows for key in row} - set(fields)))
    with output.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, lineterminator="\n")
        writer.writeheader()
        writer.writerows(flat_rows)


def write_annotation_template(pilot_rows: list[dict[str, Any]], output: Path) -> None:
    fields = ANNOTATION_FIELDS + [
        "candidate_parent", "candidate_child", "old_match_cosine", "automatic_review_state",
    ]
    with output.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, lineterminator="\n")
        writer.writeheader()
        for row in pilot_rows:
            blank = {field: "" for field in ANNOTATION_FIELDS}
            blank["gene_id"] = row["gene_id"]
            blank.update({
                "candidate_parent": row["old_parent_candidate"],
                "candidate_child": row["old_child_candidate"],
                "old_match_cosine": row["old_match_cosine"],
                "automatic_review_state": row["automatic_review_state"],
            })
            writer.writerow(blank)


def run(args: argparse.Namespace) -> None:
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=True)
    log(f"load {args.sparse}")
    sparse = np.load(args.sparse, allow_pickle=True)
    idx = sparse["idx"].astype(np.int64)
    val = sparse["val"].astype(np.float32)
    city = np.array([str(x) for x in sparse["city"]])
    pano = np.array([str(x) for x in sparse["pano"]])
    heading = sparse["heading"].astype(int)
    log(f"loaded {idx.shape[0]:,} images, {idx.shape[1]:,} patches/image, topk={idx.shape[2]}")

    rows, maxmat, pos = build_diagnostics(
        idx, val, city, Path(args.checkpoint), Path(args.old_checkpoint),
    )
    pilot = select_stratified_pilot(rows, args.pilot_size, args.seed)
    pilot_set = set(pilot)
    rows_by_gene = {int(row["gene_id"]): row for row in rows}
    for rank, gene in enumerate(pilot):
        row = rows_by_gene[gene]
        row["pilot"] = True
        row["pilot_rank"] = rank
        row["pilot_stratum"] = (
            f"{row['prevalence_group']} / {row['old_match_group']} / "
            f"{row['automatic_review_state']}"
        )
    for row in rows:
        if row["gene_id"] not in pilot_set:
            row.update({"pilot": False, "pilot_rank": None, "pilot_stratum": None})

    log(f"render {len(pilot)} pilot genes")
    examples = render_pilot(
        pilot, rows_by_gene, maxmat, pos["maps"], idx, val, city, pano, heading,
        output, args.seed,
    )
    pilot_rows = [rows_by_gene[gene] for gene in pilot]
    summary = {
        "dataset": "BatchTopK W1024/K8",
        "n_genes": len(rows),
        "n_images": int(idx.shape[0]),
        "n_patches": int(idx.shape[0] * idx.shape[1]),
        "pilot_size": len(pilot),
        "prevalence_threshold": PREVALENCE_THRESHOLD,
        "zero_support_all_slots": sum(r["support_all_slots"] == 0 for r in rows),
        "zero_support_top1": sum(r["support_top1_patches"] == 0 for r in rows),
        "below_prevalence_threshold": sum(r["prevalence"] == 0 for r in rows),
        "position_r2_ge_035": sum(r["position_r2_top1"] >= 0.35 for r in rows),
        "old_match_ge_075": sum(r["old_match_cosine"] >= 0.75 for r in rows),
        "old_match_ge_070": sum(r["old_match_cosine"] >= 0.70 for r in rows),
        "decoder_neighbour_ge_080": sum(r["nearest_new_cosine"] >= 0.80 for r in rows),
    }
    manifest = {
        "summary": summary,
        "annotation_fields": ANNOTATION_FIELDS,
        "pilot_gene_ids": pilot,
        "genes": {str(row["gene_id"]): row for row in pilot_rows},
        "examples": examples,
    }
    (output / "manifest.json").write_text(json.dumps(manifest, ensure_ascii=False))
    (output / "diagnostics.json").write_text(json.dumps({"summary": summary, "genes": rows}, ensure_ascii=False))
    write_csv(rows, output / "diagnostics.csv")
    write_annotation_template(pilot_rows, output / "annotation_template.csv")
    log(f"[done] audit -> {output}")
    log(json.dumps(summary, ensure_ascii=False))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sparse", type=Path, default=RUN / "sparse_acts.npz")
    parser.add_argument("--checkpoint", type=Path, default=RUN / "batch_topk_w1024_k8.pt")
    parser.add_argument("--old-checkpoint", type=Path, default=OLD / "sae_448_k512.pt")
    parser.add_argument("--output", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--pilot-size", type=int, default=128)
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


if __name__ == "__main__":
    run(parse_args())
