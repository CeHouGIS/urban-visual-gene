"""Discover rotation-invariant panorama motifs and compare dictionary scales.

Run on complete, QC-verified four-view feature blocks. The learned clusters
are candidate scene templates; semantic names require image-level validation.
"""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
from sklearn.cluster import AgglomerativeClustering
from sklearn.metrics import adjusted_rand_score

from formal.analysis.analyze_dictionary_scaling import DEFAULT_CHECKPOINTS, load_model


ROWS_PER_PANO = 4 * 28 * 28
DEFAULT_MANIFEST = "/host/root/mnt/nas/huangyj/GoogleSV/urban_visual_gene/formal_batchtopk_307city_qc_filtered/feature_qc_training_manifest.json"
DEFAULT_QC = "/host/root/mnt/nas/huangyj/GoogleSV/urban_visual_gene/formal_city_qc/feature_qc_map.json"
DEFAULT_SCALING = "/host/root/mnt/nas/huangyj/GoogleSV/urban_visual_gene/formal_batchtopk_307city_scaling_analysis_100k"


def filtered_to_source(pano: int, bad_positions: list[int] | tuple[int, ...]) -> int:
    """Invert the filtered panorama index, including consecutive QC removals."""
    source = pano
    for bad in bad_positions:
        if bad <= source:
            source += 1
        else:
            break
    return source


def sample_panoramas(manifest: dict, qc: dict, per_city: int, seed: int, training_rows: int):
    cities = list(manifest["cities"])
    base, extra = divmod(training_rows, len(cities))
    rng = np.random.default_rng(seed)
    selected = []
    for i, city in enumerate(cities):
        meta = manifest["cities"][city]
        q = qc["cities"].get(city, {})
        if not meta.get("verified") or not q.get("verified"):
            continue
        n = int(meta["filtered_rows"]) // ROWS_PER_PANO
        take = min(int(meta["filtered_rows"]), base + (i < extra))
        start = (i * 104729) % (int(meta["filtered_rows"]) - take + 1)
        # Training selected a contiguous row interval. Exclude every panorama
        # touched by that interval, including partially selected endpoints.
        first_train = start // ROWS_PER_PANO
        last_train = (start + take - 1) // ROWS_PER_PANO
        eligible = np.r_[0:first_train, last_train + 1:n]
        if len(eligible) < per_city:
            continue
        for pano in sorted(rng.choice(eligible, size=per_city, replace=False)):
            source_pano = filtered_to_source(int(pano), q.get("bad_kept_positions", []))
            selected.append({"city": city, "filtered_pano": int(pano), "source_pano": source_pano})
    return selected


def baseline_groups(decoder: np.ndarray, n_groups: int) -> np.ndarray:
    normalized = decoder / np.maximum(np.linalg.norm(decoder, axis=1, keepdims=True), 1e-8)
    distance = np.maximum(1 - normalized @ normalized.T, 0)
    np.fill_diagonal(distance, 0)
    return AgglomerativeClustering(n_clusters=n_groups, metric="precomputed", linkage="average").fit_predict(distance).astype(np.int64)


def mapped_groups(groups: np.ndarray, metrics_dir: Path, width: int) -> np.ndarray:
    if width == 1024:
        return groups
    left, right = (512, 1024) if width == 512 else (1024, 2048)
    with np.load(metrics_dir / f"pair_metrics_w{left}_to_w{right}.npz") as data:
        score = .7 * data["decoder_cosine"] + .3 * data["top_row_overlap"]
    match = score.argmax(axis=1 if width == 512 else 0)
    if len(match) != width:
        raise ValueError(f"wrong match dimensions for W={width}: {len(match)}")
    return groups[match]


def activation_map(model, features: np.ndarray, groups: np.ndarray, n_groups: int, device: torch.device) -> np.ndarray:
    x = torch.from_numpy(np.asarray(features, dtype=np.float32)).to(device)
    with torch.inference_mode():
        acts = model.encode(x)
        group_index = torch.as_tensor(groups, device=device, dtype=torch.long)
        grouped = torch.zeros((len(x), n_groups), device=device, dtype=acts.dtype)
        grouped.scatter_add_(1, group_index[None].expand(len(x), -1), acts)
        # Source order is heading, row, column; join headings horizontally.
        panorama = grouped.reshape(4, 28, 28, n_groups).permute(3, 1, 0, 2)
        panorama = panorama.reshape(1, n_groups, 28, 112)
        grid = torch.nn.functional.adaptive_avg_pool2d(panorama, (4, 8))[0]
    return grid.cpu().numpy().astype(np.float32)


def normalize_maps(maps: np.ndarray) -> np.ndarray:
    # Hellinger-style transform balances common and rare group responses.
    mass = maps / np.maximum(maps.sum(axis=(1, 2, 3), keepdims=True), 1e-12)
    root = np.sqrt(mass)
    return root / np.maximum(np.linalg.norm(root.reshape(len(root), -1), axis=1)[:, None, None, None], 1e-12)


def aligned_similarity(maps: np.ndarray, centers: np.ndarray):
    n, g, h, sectors = maps.shape
    shifted = np.stack([np.roll(maps, shift, axis=-1) for shift in range(sectors)], axis=1)
    score = shifted.reshape(n * sectors, -1) @ centers.reshape(len(centers), -1).T
    score = score.reshape(n, sectors, len(centers))
    best_by_template = score.max(axis=1)
    assignment = best_by_template.argmax(axis=1)
    best_shift = score[np.arange(n), :, assignment].argmax(axis=1)
    similarity = score[np.arange(n), best_shift, assignment]
    return assignment, best_shift, similarity


def fit_templates(maps: np.ndarray, n_templates: int, seed: int, iterations: int = 25):
    rng = np.random.default_rng(seed)
    centers = maps[rng.choice(len(maps), size=n_templates, replace=False)].copy()
    previous = None
    for _ in range(iterations):
        labels, shifts, _ = aligned_similarity(maps, centers)
        aligned = np.stack([np.roll(maps[i], int(shifts[i]), axis=-1) for i in range(len(maps))])
        new = centers.copy()
        for k in range(n_templates):
            members = aligned[labels == k]
            if len(members):
                new[k] = members.mean(axis=0)
                new[k] /= max(np.linalg.norm(new[k]), 1e-12)
        centers = new
        if previous is not None and np.array_equal(previous, labels):
            break
        previous = labels.copy()
    return centers, labels


def spatial_null(maps: np.ndarray, seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    result = maps.copy()
    # Same permutation for all groups within each panorama/height band:
    # composition is fixed but angular adjacency is broken.
    for i in range(len(result)):
        for height in range(result.shape[2]):
            result[i, :, height] = result[i, :, height][:, rng.permutation(result.shape[-1])]
    return result


def plot_templates(centers: np.ndarray, maps: np.ndarray, labels: np.ndarray, records: list[dict], output: Path):
    n = len(centers)
    fig, axes = plt.subplots(n, 2, figsize=(13, 2.4 * n), squeeze=False)
    for k in range(n):
        members = np.flatnonzero(labels == k)
        if not len(members):
            continue
        strength = centers[k].sum(axis=(1, 2))
        top = np.argsort(strength)[-6:][::-1]
        view = centers[k, top].sum(axis=0)
        axes[k, 0].imshow(view, aspect="auto", cmap="magma")
        axes[k, 0].set_title(f"Template {k} | {len(members)} discovery panoramas | strongest groups {top.tolist()}", fontsize=9)
        axes[k, 0].set_ylabel("vertical band")
        axes[k, 0].set_xticks(range(8), [str(x * 45) for x in range(8)])
        axes[k, 0].set_xlabel("relative yaw sector (degrees)")
        axes[k, 1].bar(range(len(top)), strength[top], color="#358a88")
        axes[k, 1].set_xticks(range(len(top)), [str(x) for x in top])
        axes[k, 1].set_ylabel("relative group response")
        axes[k, 1].set_xlabel("candidate primitive group")
    fig.tight_layout()
    fig.savefig(output, dpi=160)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", default=DEFAULT_MANIFEST)
    parser.add_argument("--qc-map", default=DEFAULT_QC)
    parser.add_argument("--scaling-dir", default=DEFAULT_SCALING)
    parser.add_argument("--checkpoint", action="append", help="WIDTH=PATH; repeat as needed")
    parser.add_argument("--outdir", required=True)
    parser.add_argument("--per-city", type=int, default=2)
    parser.add_argument("--groups", type=int, default=64)
    parser.add_argument("--templates", type=int, default=8)
    parser.add_argument("--seed", type=int, default=20261009)
    parser.add_argument("--training-rows", type=int, default=30_000_000)
    parser.add_argument("--device", default="cuda:1")
    args = parser.parse_args()
    if args.per_city < 1 or args.groups < 2 or args.templates < 2:
        parser.error("per-city, groups, and templates must be positive")
    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    manifest = json.loads(Path(args.manifest).read_text())
    qc = json.loads(Path(args.qc_map).read_text())
    selected = sample_panoramas(manifest, qc, args.per_city, args.seed, args.training_rows)
    if len(selected) < 2 * args.templates:
        raise ValueError(f"only {len(selected)} QC-verified held-out panoramas")
    checkpoints = dict(DEFAULT_CHECKPOINTS)
    for entry in args.checkpoint or []:
        width, path = entry.split("=", 1)
        checkpoints[int(width)] = path
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    print(f"{len(selected)} panoramas from {len({x['city'] for x in selected})} cities on {device}", flush=True)
    scaling_dir = Path(args.scaling_dir)
    base_model, decoder, _ = load_model(Path(checkpoints[1024]), device)
    groups = baseline_groups(decoder, args.groups)
    del base_model
    np.save(outdir / "baseline_group_ids.npy", groups)
    maps = {}
    source_root = Path(manifest["source_out"])
    for width in (512, 1024, 2048):
        model, _, _ = load_model(Path(checkpoints[width]), device)
        mapping = mapped_groups(groups, scaling_dir, width)
        np.save(outdir / f"group_ids_w{width}.npy", mapping)
        outputs = np.empty((len(selected), args.groups, 4, 8), dtype=np.float32)
        old_city = None
        for i, item in enumerate(selected):
            if item["city"] != old_city:
                old_city = item["city"]
                source = np.load(source_root / f"dict_{old_city}_p784.f16.npy", mmap_mode="r")
            start = item["source_pano"] * ROWS_PER_PANO
            outputs[i] = activation_map(model, source[start:start + ROWS_PER_PANO], mapping, args.groups, device)
            if (i + 1) % 100 == 0 or i + 1 == len(selected):
                print(f"W={width}: {i + 1}/{len(selected)}", flush=True)
        maps[width] = normalize_maps(outputs)
        np.save(outdir / f"spatial_maps_w{width}.npy", maps[width].astype(np.float16))
        del model, outputs, source
        if device.type == "cuda":
            torch.cuda.empty_cache()

    # Hold out cities, not individual panoramas from the same city.
    cities = sorted({item["city"] for item in selected})
    rng = np.random.default_rng(args.seed)
    rng.shuffle(cities)
    train_cities = set(cities[:round(.7 * len(cities))])
    train = np.array([i for i, item in enumerate(selected) if item["city"] in train_cities])
    test = np.array([i for i in range(len(selected)) if i not in set(train)])
    centers, labels = fit_templates(maps[1024][train], args.templates, args.seed)
    np.save(outdir / "template_centers.npy", centers)
    plot_templates(centers, maps[1024][train], labels, [selected[i] for i in train], outdir / "template_spatial_profiles.png")
    baseline_labels, baseline_shifts, baseline_scores = aligned_similarity(maps[1024][test], centers)
    null = spatial_null(maps[1024][test], args.seed + 1)
    _, _, null_scores = aligned_similarity(null, centers)
    scale_results = {}
    for width in (512, 1024, 2048):
        assigned, shifts, scores = aligned_similarity(maps[width][test], centers)
        scale_results[str(width)] = {
            "agreement_with_w1024_ari": float(adjusted_rand_score(baseline_labels, assigned)),
            "assignment_agreement_fraction": float(np.mean(assigned == baseline_labels)),
            "mean_centroid_cosine": float(scores.mean()),
            "mean_cosine_to_w1024_maps": float(np.mean(np.sum(maps[width][test] * maps[1024][test], axis=(1, 2, 3)))),
        }
        for local, global_index in enumerate(test):
            selected[global_index][f"template_w{width}"] = int(assigned[local])
            selected[global_index][f"shift_w{width}"] = int(shifts[local])
            selected[global_index][f"similarity_w{width}"] = float(scores[local])
    for local, global_index in enumerate(train):
        selected[global_index]["template_w1024"] = int(labels[local])
    (outdir / "panorama_assignments.json").write_text(json.dumps(selected, indent=2))
    with (outdir / "panorama_assignments.csv").open("w", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=sorted({key for row in selected for key in row}))
        writer.writeheader()
        writer.writerows(selected)
    scaling_summary = json.loads((scaling_dir / "scaling_summary.json").read_text())
    summary = {
        "status": "candidate spatial templates; semantic interpretation pending image inspection",
        "sample": {"panoramas": len(selected), "cities": len(cities), "training_cities": len(train_cities), "heldout_cities": len(cities) - len(train_cities), "per_city": args.per_city, "excluded_training_cache_rows": args.training_rows},
        "method": {"groups": args.groups, "grouping": "W1024 decoder cosine agglomerative clustering; other widths mapped by decoder cosine plus top-row overlap", "templates": args.templates, "grid": "4 vertical by 8 circular yaw sectors", "rotation": "best of 8 circular yaw shifts", "sparse_encoding_batch": "one complete panorama (3136 patches) per BatchTopK call", "split": "70% discovery cities, 30% held-out cities"},
        "spatial_test": {"heldout_mean_centroid_cosine": float(baseline_scores.mean()), "angular_permutation_null_mean": float(null_scores.mean()), "paired_difference": float(np.mean(baseline_scores - null_scores)), "positive_fraction": float(np.mean(baseline_scores > null_scores))},
        "dictionary_scale": scale_results,
        "feature_scale": {name: pair["summary"] for name, pair in scaling_summary["pairs"].items()},
        "limitations": ["Grouping uses decoder directions rather than validated semantic units", "Scale mappings are based on training-cache activation examples and require held-out confirmation", "Template spatial test permutes angular sectors and is not a full physical-scene null", "Original image montage and semantic review remain necessary"],
    }
    (outdir / "template_summary.json").write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=2), flush=True)


if __name__ == "__main__":
    main()
