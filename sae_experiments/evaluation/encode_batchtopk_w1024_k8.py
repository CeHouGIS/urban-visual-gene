from __future__ import annotations

import argparse
import csv
import json
import time
from pathlib import Path

import numpy as np
import torch
from PIL import Image

from formal.gpu_run import Extractor, contextualize_views, imgpath
from sae_experiments.models.panorama_context import PANORAMA_CONTEXT_WEIGHT
from sae_experiments.models.base_sae import build_sae


CITY_ORDER = [
    "HongKong", "Taipei", "Singapore", "Seoul", "Sapporo", "Bengaluru",
    "Kolkata", "Colombo", "Bandung", "Surabaya", "Dubai", "Johannesburg",
    "CapeTown", "Lagos", "Amsterdam", "Paris", "Barcelona", "Vienna",
    "Prague", "Stockholm", "Oslo", "Moscow", "Lisboa", "BuenosAires",
    "SaoPaulo", "RiodeJaneiro", "Bogota", "NewYork", "LosAngeles", "Chicago",
    "SanFrancisco", "Washington", "Sydney", "Melbourne", "Vancouver",
    "Monterrey", "ChiangMai",
]


def log(*args: object) -> None:
    print(f"[{time.strftime('%H:%M:%S')}]", *args, flush=True)


def reference_rows(ref_sparse: Path) -> list[tuple[str, str, int]]:
    z = np.load(ref_sparse, allow_pickle=True)
    city = np.array([str(c) for c in z["city"]])
    pano = np.array([str(p) for p in z["pano"]])
    heading = z["heading"].astype(int)
    rows = list(zip(city, pano, heading))
    log(f"reference rows={len(rows):,} from {ref_sparse}")
    return rows


def panorama_groups(rows: list[tuple[str, str, int]]) -> list[list[tuple[str, str, int]]]:
    """Return complete four-view groups while preserving reference order."""
    groups: dict[tuple[str, str], dict[int, tuple[str, str, int]]] = {}
    order: list[tuple[str, str]] = []
    for row in rows:
        key = (row[0], row[1])
        if key not in groups:
            groups[key] = {}
            order.append(key)
        groups[key][int(row[2])] = row
    complete = []
    for key in order:
        group = groups[key]
        if set(group) == {0, 90, 180, 270}:
            complete.append([group[d] for d in (0, 90, 180, 270)])
    return complete


def load_model(checkpoint: Path, device: str):
    meta = torch.load(checkpoint, map_location="cpu")
    d = int(meta["D"])
    width = int(meta["K"])
    k = int(meta.get("k", meta.get("topk", 8)))
    model_type = meta.get("model_type", "batch_topk")
    model = build_sae(
        input_dim=d,
        config={"type": model_type, "latent_dim": width, "k": k, "decoder_unit_norm": True},
    ).to(device)
    model.load_state_dict(meta["state"])
    model.eval()
    log(
        f"loaded {checkpoint.name}: type={model_type} D={d} W={width} K={k} "
        f"loss={meta.get('final_train_loss')}"
    )
    return model, meta


@torch.no_grad()
def encode(args: argparse.Namespace) -> None:
    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    sparse_out = outdir / "sparse_acts.npz"
    summary_out = outdir / "summary.json"
    csv_out = outdir / "summary.csv"
    if sparse_out.exists() and not args.force:
        log(f"reuse existing {sparse_out}")
        summarize(sparse_out, Path(args.checkpoint), args.threshold, summary_out, csv_out)
        return

    rows = reference_rows(Path(args.ref_sparse))
    if args.max_images:
        rows = rows[: args.max_images]
    groups = panorama_groups(rows)
    log(f"complete panorama groups={len(groups):,} from {len(rows):,} reference rows")
    device = "cuda" if torch.cuda.is_available() and not args.cpu else "cpu"
    model, meta = load_model(Path(args.checkpoint), device)
    context_weight = float(meta.get("context_weight", args.context_weight))
    log(f"panorama context weight={context_weight}")
    width = int(meta["K"])
    k = int(meta.get("k", meta.get("topk", 8)))
    ext = Extractor(args.res, proj_path=args.project_dirs or None)
    grid = args.res // 16
    n_patch = grid * grid
    idx = np.empty((len(rows), n_patch, k), dtype=np.int16)
    val = np.empty((len(rows), n_patch, k), dtype=np.float16)
    keep_city: list[str] = []
    keep_pano: list[str] = []
    keep_heading: list[int] = []
    kept = 0

    group_batch_size = max(1, args.batch // 4)
    for st in range(0, len(groups), group_batch_size):
        group_batch = groups[st:st + group_batch_size]
        pils = []
        valid_groups = []
        for group in group_batch:
            try:
                views = [
                    Image.open(imgpath(city, pano, int(heading))).convert("RGB").resize(
                        (args.res, args.res), Image.BILINEAR
                    )
                    for city, pano, heading in group
                ]
            except Exception as e:
                log(f"skip incomplete panorama {group[0][0]}/{group[0][1]}: {e}")
                continue
            pils.extend(views)
            valid_groups.append(group)
        if not pils:
            continue
        feats, artifacts = ext.batch(pils)
        contextual, valid_mask = contextualize_views(
            feats, artifacts, grid, context_weight
        )
        flat = contextual.reshape(-1, contextual.shape[-1]).to(device)
        acts = model.encode(flat)
        acts[~valid_mask.reshape(-1).to(device)] = 0
        acts = acts.reshape(len(valid_groups), 4, n_patch, width)
        for local_i, group in enumerate(valid_groups):
            for direction_i, (city, pano, heading) in enumerate(group):
                row_acts = acts[local_i, direction_i]
                v, ix = row_acts.topk(k, dim=1)
                idx[kept] = ix.cpu().numpy().astype(np.int16)
                val[kept] = v.cpu().numpy().astype(np.float16)
                keep_city.append(city)
                keep_pano.append(pano)
                keep_heading.append(heading)
                kept += 1
        if kept % 512 < args.batch:
            log(f"encoded {kept:,}/{len(rows):,} images")

    np.savez_compressed(
        sparse_out,
        idx=idx[:kept],
        val=val[:kept],
        city=np.array(keep_city),
        pano=np.array(keep_pano),
        heading=np.array(keep_heading, dtype=np.int16),
    )
    log(f"saved {sparse_out} shape={idx[:kept].shape}")
    summarize(sparse_out, Path(args.checkpoint), args.threshold, summary_out, csv_out)


def summarize(sparse_path: Path, checkpoint: Path, threshold: float, summary_out: Path, csv_out: Path) -> None:
    z = np.load(sparse_path, allow_pickle=True)
    idx = z["idx"].astype(np.int64)
    val = z["val"].astype(np.float32)
    city = np.array([str(c) for c in z["city"]])
    pano = np.array([str(p) for p in z["pano"]])
    meta = torch.load(checkpoint, map_location="cpu")
    width = int(meta["K"])
    top = idx[:, :, 0]
    prof = np.zeros((len(CITY_ORDER), width), dtype=np.float64)
    images_per_city = {}
    for ci, c in enumerate(CITY_ORDER):
        mask = city == c
        images_per_city[c] = int(mask.sum())
        counts = np.zeros(width, dtype=np.float64)
        city_rows = np.flatnonzero(mask)
        city_panos = np.unique(pano[city_rows])
        for pano_id in city_panos:
            genes = np.unique(top[city_rows[pano[city_rows] == pano_id]].ravel())
            counts[genes] += 1.0
        prof[ci] = counts / max(float(len(city_panos)), 1.0)
    prevalence = (prof >= threshold).sum(0)
    n_cities = len(CITY_ORDER)
    classes = {
        "unused": int((prevalence == 0).sum()),
        "city_unique_1city": int((prevalence == 1).sum()),
        "pair_specific_2cities": int((prevalence == 2).sum()),
        "regional_3to5cities": int(((prevalence >= 3) & (prevalence <= 5)).sum()),
        "accessory_6to8cities": int(((prevalence >= 6) & (prevalence <= 8)).sum()),
        "near_core_9to11cities": int(((prevalence >= 9) & (prevalence <= 11)).sum()),
        f"core_universal_{n_cities}cities": int((prevalence == n_cities).sum()),
    }
    uniq_per_img = np.array([
        len(np.unique(top[i].ravel())) for i in range(top.shape[0])
    ])
    active_per_patch = (val > 0).sum(2).astype(np.float32)
    summary = {
        "model_type": meta.get("model_type", "batch_topk"),
        "width": width,
        "k": int(meta.get("k", meta.get("topk", 8))),
        "threshold": threshold,
        "n_images": int(top.shape[0]),
        "n_patches": int(top.size),
        "n_panoramas": int(np.unique(pano).size),
        "images_per_city": images_per_city,
        "n_used": int((prevalence >= 1).sum()),
        "class_counts": classes,
        f"prevalence_spectrum_1to{n_cities}": [int((prevalence == i).sum()) for i in range(1, n_cities + 1)],
        "core_fraction": round(classes[f"core_universal_{n_cities}cities"] / width, 4),
        "city_unique_fraction": round(classes["city_unique_1city"] / width, 4),
        "unique_or_pair_fraction": round((classes["city_unique_1city"] + classes["pair_specific_2cities"]) / width, 4),
        "unique_genes_per_image_mean": round(float(uniq_per_img.mean()), 3),
        "active_dims_patch_mean": round(float(active_per_patch.mean()), 3),
        "final_train_loss": meta.get("final_train_loss"),
        "approx_train_recon_cos": round(1 - float(meta.get("final_train_loss", 0)), 6)
        if meta.get("final_train_loss") is not None else None,
    }
    summary_out.write_text(json.dumps(summary, ensure_ascii=False, indent=2))
    with csv_out.open("w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow([
            "model_type", "width", "k", "n_used", f"core_{n_cities}", "city_unique_1",
            "pair_specific_2", "regional_3to5", "accessory_6to8",
            "near_core_9to11", "unused", "core_fraction",
            "city_unique_fraction", "unique_or_pair_fraction",
            "final_train_loss", "approx_train_recon_cos",
            "active_dims_patch_mean", "unique_genes_per_image_mean",
        ])
        c = summary["class_counts"]
        writer.writerow([
            summary["model_type"], width, summary["k"], summary["n_used"],
            c[f"core_universal_{n_cities}cities"], c["city_unique_1city"],
            c["pair_specific_2cities"], c["regional_3to5cities"],
            c["accessory_6to8cities"], c["near_core_9to11cities"], c["unused"],
            summary["core_fraction"], summary["city_unique_fraction"],
            summary["unique_or_pair_fraction"], summary["final_train_loss"],
            summary["approx_train_recon_cos"], summary["active_dims_patch_mean"],
            summary["unique_genes_per_image_mean"],
        ])
    log(
        f"summary core{n_cities}={classes[f'core_universal_{n_cities}cities']} unique1={classes['city_unique_1city']} "
        f"pair2={classes['pair_specific_2cities']} n_used={summary['n_used']}"
    )


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Encode complete four-view panoramas with contextual BatchTopK W1024 K8.")
    # Keep the defaults aligned with the resumable formal runner.  The old
    # ``panorama_context_batchtopk_w1024_k8`` and ``formal_out_global`` paths
    # belong to pilot/ablation runs and may not exist in a fresh checkout.
    p.add_argument("--checkpoint", default="formal/formal_out_panorama_context/sae_448_k1024.pt")
    p.add_argument("--outdir", default="formal/formal_out_panorama_context/encoded")
    p.add_argument("--ref-sparse", default="formal/formal_out_panorama_context/genes/sparse_acts.npz")
    p.add_argument("--project-dirs", default="")
    p.add_argument("--res", type=int, default=448)
    p.add_argument("--batch", type=int, default=32)
    p.add_argument("--threshold", type=float, default=5e-4)
    p.add_argument("--max-images", type=int, default=0)
    p.add_argument("--force", action="store_true")
    p.add_argument("--cpu", action="store_true")
    p.add_argument("--context-weight", type=float, default=PANORAMA_CONTEXT_WEIGHT)
    return p.parse_args()


def main() -> None:
    encode(parse_args())


if __name__ == "__main__":
    main()
