"""Compare BatchTopK dictionary widths on identical held-out street-view patches.

The sample excludes every panorama touched by the 30M-row SAE training cache.
Outputs distinguish stable matches, complementary split candidates, and
larger-dictionary features without a convincing smaller-width counterpart.
"""
from __future__ import annotations

import argparse
import csv
import json
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
import torch
from scipy import sparse

from formal.analysis.analyze_dictionary_scaling import DEFAULT_CHECKPOINTS, load_model
from formal.analysis.discover_scene_templates import (
    DEFAULT_MANIFEST,
    DEFAULT_QC,
    ROWS_PER_PANO,
    sample_panoramas,
)


def select_anchors(panoramas: list[dict], patches_per_pano: int, seed: int):
    if patches_per_pano % 4:
        raise ValueError("patches_per_pano must be divisible by four")
    if patches_per_pano > ROWS_PER_PANO:
        raise ValueError("patches_per_pano exceeds one panorama")
    rng = np.random.default_rng(seed)
    anchors = []
    per_view = patches_per_pano // 4
    for record in panoramas:
        for heading in range(4):
            for patch in np.sort(rng.choice(784, per_view, replace=False)):
                anchors.append({**record, "heading": heading * 90, "patch": int(patch)})
    return anchors


def expanded_anchors(manifest: dict, qc: dict, baseline: list[dict], per_city: int,
                     patches_per_pano: int, seed: int, training_rows: int):
    """Keep all baseline patches while adding training-excluded panoramas."""
    existing = defaultdict(list)
    for row in baseline:
        existing[(row["city"], row["source_pano"])].append(row)
    counts = Counter(city for city, _ in existing)
    if any(n > per_city for n in counts.values()):
        raise ValueError("expanded per-city quota is smaller than the baseline")
    # Oversample by the baseline count to leave enough room after deduplication.
    candidates = sample_panoramas(manifest, qc, per_city + max(counts.values(), default=0), seed, training_rows)
    chosen = {key: {k: rows[0][k] for k in ("city", "source_pano", "filtered_pano")} for key, rows in existing.items()}
    for row in candidates:
        key = (row["city"], row["source_pano"])
        if key not in chosen and counts[row["city"]] < per_city:
            chosen[key] = row
            counts[row["city"]] += 1
    panos = sorted(chosen.values(), key=lambda row: (row["city"], row["source_pano"]))
    generated = select_anchors(panos, patches_per_pano, seed)
    grouped = defaultdict(list)
    for row in generated:
        grouped[(row["city"], row["source_pano"])].append(row)
    anchors = [anchor for row in panos for anchor in existing.get((row["city"], row["source_pano"]), grouped[(row["city"], row["source_pano"])])]
    return panos, anchors


def read_anchor_features(anchors: list[dict], root: Path, workers: int = 4):
    values = np.empty((len(anchors), 1024), dtype=np.float16)
    tasks = []
    i = 0
    while i < len(anchors):
        record = anchors[i]
        end = i + 1
        while end < len(anchors) and anchors[end]["city"] == record["city"] and anchors[end]["source_pano"] == record["source_pano"]:
            end += 1
        tasks.append((i, end, record))
        i = end

    def read_task(task):
        i, end, record = task
        source = np.load(root / f"dict_{record['city']}_p784.f16.npy", mmap_mode="r")
        start_row = record["source_pano"] * ROWS_PER_PANO
        pano = np.asarray(source[start_row:start_row + ROWS_PER_PANO]).copy()
        offsets = [row["heading"] // 90 * 784 + row["patch"] for row in anchors[i:end]]
        return i, end, pano[offsets]

    with ThreadPoolExecutor(max_workers=workers) as pool:
        for done, (i, end, chunk) in enumerate(pool.map(read_task, tasks), 1):
            values[i:end] = chunk
            if done % 100 == 0 or done == len(tasks):
                print(f"NAS panorama blocks: {done}/{len(tasks)}", flush=True)
    return values


def encode_anchors(model, features: np.ndarray, batch_size: int, device: torch.device):
    n = len(features)
    acts = np.empty((n, model.latent_dim), dtype=np.float16)
    losses = []
    with torch.inference_mode():
        for start in range(0, n, batch_size):
            x = torch.from_numpy(features[start:start + batch_size].astype(np.float32)).to(device)
            z, reconstruction = model(x)
            acts[start:start + len(x)] = z.cpu().numpy().astype(np.float16)
            loss = 1 - torch.nn.functional.cosine_similarity(x, reconstruction).mean()
            losses.append((len(x), float(loss)))
    mean_loss = sum(count * loss for count, loss in losses) / n
    return acts, mean_loss


def pair_matrices(left: np.ndarray, right: np.ndarray, decoder_left: np.ndarray, decoder_right: np.ndarray):
    la = sparse.csr_matrix(left.astype(np.float32))
    ra = sparse.csr_matrix(right.astype(np.float32))
    dot = (la.T @ ra).toarray()
    norms_left = np.sqrt(np.asarray(la.power(2).sum(axis=0)).ravel())
    norms_right = np.sqrt(np.asarray(ra.power(2).sum(axis=0)).ravel())
    activation_cosine = dot / np.maximum(norms_left[:, None] * norms_right[None], 1e-12)
    lb = la.copy()
    rb = ra.copy()
    lb.data[:] = 1
    rb.data[:] = 1
    intersection = (lb.T @ rb).toarray().astype(np.int32)
    left_frequency = np.asarray(lb.sum(axis=0)).ravel().astype(np.int32)
    right_frequency = np.asarray(rb.sum(axis=0)).ravel().astype(np.int32)
    recall = intersection / np.maximum(left_frequency[:, None], 1)
    precision = intersection / np.maximum(right_frequency[None], 1)
    jaccard = intersection / np.maximum(left_frequency[:, None] + right_frequency[None] - intersection, 1)
    decoder_cosine = decoder_left @ decoder_right.T
    return {
        "decoder_cosine": decoder_cosine.astype(np.float32),
        "activation_cosine": activation_cosine.astype(np.float32),
        "support_jaccard": jaccard.astype(np.float32),
        "parent_recall": recall.astype(np.float32),
        "child_precision": precision.astype(np.float32),
        "left_frequency": left_frequency,
        "right_frequency": right_frequency,
    }


def top_positive_rows(acts: np.ndarray, feature: int, limit: int = 64):
    values = acts[:, feature]
    positive = np.flatnonzero(values > 0)
    if not len(positive):
        return []
    top = positive[np.argsort(values[positive])[-limit:][::-1]]
    return top.astype(int).tolist()


def packed_support(left: np.ndarray, right: np.ndarray):
    """Compact support sets for exact repeated intersection counts."""
    return {"left": np.packbits(np.ascontiguousarray((left > 0).T), axis=1),
            "right": np.packbits(np.ascontiguousarray((right > 0).T), axis=1),
            "triples": {}, "popcount": np.array([i.bit_count() for i in range(256)], dtype=np.uint8)}


def classify(left: np.ndarray, right: np.ndarray, metrics: dict, min_support: int = 25, stable_cosine: float = .70, split_coverage: float = .50, supports: dict | None = None):
    dec = metrics["decoder_cosine"]
    act = metrics["activation_cosine"]
    jac = metrics["support_jaccard"]
    recall = metrics["parent_recall"]
    precision = metrics["child_precision"]
    lf = metrics["left_frequency"]
    rf = metrics["right_frequency"]
    eligible = (lf[:, None] >= min_support) & (rf[None] >= min_support)
    # Mutual best matching prevents several unrelated broad features from
    # claiming the same larger-dictionary feature as a stable continuation.
    score = .5 * dec + .5 * act
    left_best = score.argmax(axis=1)
    right_best = score.argmax(axis=0)
    stable = []
    for i, j in enumerate(left_best):
        if right_best[j] == i and eligible[i, j] and dec[i, j] >= stable_cosine and act[i, j] >= .35 and jac[i, j] >= .15:
            stable.append({"parent": i, "child": int(j), "decoder_cosine": float(dec[i, j]), "activation_cosine": float(act[i, j]), "support_jaccard": float(jac[i, j])})

    splits = []
    supports = packed_support(left, right) if supports is None else supports
    # Two children must each explain nontrivial and partly distinct sets of
    # held-out parent activations. This excludes duplicate children.
    for i in range(len(left_best)):
        if lf[i] < min_support:
            continue
        candidates = np.flatnonzero(eligible[i] & (dec[i] >= .45) & (recall[i] >= .15) & (precision[i] >= .20))
        if len(candidates) < 2:
            continue
        rank = sorted(candidates, key=lambda j: float(recall[i, j] * precision[i, j] * max(dec[i, j], 0)), reverse=True)[:4]
        chosen = None
        for a in range(len(rank)):
            for b in range(a + 1, len(rank)):
                key = (i, *sorted((int(rank[a]), int(rank[b]))))
                if key not in supports["triples"]:
                    intersection = supports["left"][i] & supports["right"][rank[a]] & supports["right"][rank[b]]
                    supports["triples"][key] = int(supports["popcount"][intersection].sum())
                shared = supports["triples"][key]
                first_count = int(round(float(recall[i, rank[a]]) * lf[i]))
                second_count = int(round(float(recall[i, rank[b]]) * lf[i]))
                unique_a = (first_count - shared) / lf[i]
                unique_b = (second_count - shared) / lf[i]
                coverage = (first_count + second_count - shared) / lf[i]
                if unique_a >= .10 and unique_b >= .10 and coverage >= split_coverage:
                    candidate = (coverage, min(unique_a, unique_b), [int(rank[a]), int(rank[b])])
                    if chosen is None or candidate[:2] > chosen[:2]:
                        chosen = candidate
        if chosen:
            coverage, balance, children = chosen
            splits.append({"parent": i, "children": children, "parent_coverage": float(coverage), "minimum_unique_parent_fraction": float(balance), "decoder_cosines": [float(dec[i, j]) for j in children], "parent_recalls": [float(recall[i, j]) for j in children]})

    stable_parents = {row["parent"] for row in stable}
    split_parents = {row["parent"] for row in splits}
    stable_children = {row["child"] for row in stable}
    split_children = {j for row in splits for j in row["children"]}
    # 'No close match' is an observational category, not a semantic novelty
    # claim. Other unmatched features may be reorganizations or rare.
    no_close = []
    reconfigured = []
    insufficient = []
    for j in range(right.shape[1]):
        if j in stable_children or j in split_children:
            continue
        if rf[j] < min_support:
            insufficient.append(j)
        elif float(dec[:, j].max()) < .45 and float(act[:, j].max()) < .25:
            no_close.append(j)
        else:
            reconfigured.append(j)
    return {
        "stable": stable,
        "split_candidates": splits,
        "no_close_match": no_close,
        "related_or_reconfigured": reconfigured,
        "insufficient_support": insufficient,
        "parent_counts": {"stable_only": len(stable_parents - split_parents), "split_including_stable": len(split_parents), "neither": left.shape[1] - len(stable_parents | split_parents)},
        "child_counts": {"stable": len(stable_children), "split_child_not_stable": len(split_children - stable_children), "no_close_match": len(no_close), "related_or_reconfigured": len(reconfigured), "insufficient_support": len(insufficient)},
    }


def add_representatives(results: dict, left: np.ndarray, right: np.ndarray, metrics: dict):
    for row in results["stable"]:
        i, j = row["parent"], row["child"]
        row["parent_top_anchor_rows"] = top_positive_rows(left, i, 16)
        row["child_top_anchor_rows"] = top_positive_rows(right, j, 16)
    for row in results["split_candidates"]:
        i = row["parent"]
        row["parent_top_anchor_rows"] = top_positive_rows(left, i, 16)
        row["child_top_anchor_rows"] = [top_positive_rows(right, j, 16) for j in row["children"]]
        row["exclusive_shared_anchor_rows"] = []
        first, second = row["children"]
        for child, other in ((first, second), (second, first)):
            indices = np.flatnonzero((left[:, i] > 0) & (right[:, child] > 0) & (right[:, other] == 0))
            ranked = indices[np.argsort(right[indices, child])[-16:][::-1]]
            row["exclusive_shared_anchor_rows"].append(ranked.astype(int).tolist())
    right_freq = metrics["right_frequency"]
    no_close_rank = sorted(results["no_close_match"], key=lambda j: int(right_freq[j]), reverse=True)
    results["no_close_examples"] = [{"feature": int(j), "heldout_frequency": int(right_freq[j]), "max_decoder_cosine": float(metrics["decoder_cosine"][:, j].max()), "max_activation_cosine": float(metrics["activation_cosine"][:, j].max()), "top_anchor_rows": top_positive_rows(right, j, 16)} for j in no_close_rank[:20]]


def export_feature_relations(path: Path, results: dict, metrics: dict):
    categories = {}
    parents = {}
    for row in results["stable"]:
        categories[row["child"]] = "stable"
        parents.setdefault(row["child"], set()).add(row["parent"])
    for row in results["split_candidates"]:
        for child in row["children"]:
            categories.setdefault(child, "split_child")
            parents.setdefault(child, set()).add(row["parent"])
    for key in ("no_close_match", "related_or_reconfigured", "insufficient_support"):
        for child in results[key]:
            categories[child] = key
    with path.open("w", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=["feature", "category", "linked_parents", "heldout_support", "nearest_parent", "max_decoder_cosine", "max_activation_cosine", "nearest_support_jaccard"])
        writer.writeheader()
        for j in range(len(metrics["right_frequency"])):
            nearest = int((.5 * metrics["decoder_cosine"][:, j] + .5 * metrics["activation_cosine"][:, j]).argmax())
            writer.writerow({"feature": j, "category": categories[j], "linked_parents": ";".join(map(str, sorted(parents.get(j, set())))), "heldout_support": int(metrics["right_frequency"][j]), "nearest_parent": nearest, "max_decoder_cosine": float(metrics["decoder_cosine"][:, j].max()), "max_activation_cosine": float(metrics["activation_cosine"][:, j].max()), "nearest_support_jaccard": float(metrics["support_jaccard"][nearest, j])})


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--outdir", required=True)
    parser.add_argument("--manifest", default=DEFAULT_MANIFEST)
    parser.add_argument("--qc-map", default=DEFAULT_QC)
    parser.add_argument("--checkpoint", action="append", help="WIDTH=PATH")
    parser.add_argument("--per-city", type=int, default=1)
    parser.add_argument("--patches-per-pano", type=int, default=128)
    parser.add_argument("--training-rows", type=int, default=30_000_000)
    parser.add_argument("--batch-size", type=int, default=4096)
    parser.add_argument("--seed", type=int, default=20261009)
    parser.add_argument("--device", default="cuda:1")
    parser.add_argument("--io-workers", type=int, default=4)
    parser.add_argument("--local-cache", default="/tmp/uvg_scaling_heldout")
    parser.add_argument("--baseline-dir", type=Path, help="Preserve the previous evaluation panoramas and patch coordinates")
    args = parser.parse_args()
    if args.per_city < 1 or args.batch_size < 1 or args.io_workers < 1 or args.training_rows < 1:
        parser.error("per-city, batch-size, io-workers and training-rows must be positive")
    if args.patches_per_pano < 4:
        parser.error("patches-per-pano must be at least four")
    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    manifest = json.loads(Path(args.manifest).read_text())
    qc = json.loads(Path(args.qc_map).read_text())
    if args.baseline_dir:
        baseline_summary = json.loads((args.baseline_dir / "summary.json").read_text())
        baseline = json.loads((args.baseline_dir / "anchor_manifest.json").read_text())
        for key, value in (("patches_per_pano", args.patches_per_pano), ("excluded_training_cache_rows", args.training_rows), ("seed", args.seed), ("batch_size", args.batch_size)):
            if baseline_summary["sample"][key] != value:
                raise ValueError(f"baseline and expansion differ in {key}")
        old_panos = {(row["city"], row["source_pano"]) for row in baseline}
        old_per_city = max(Counter(city for city, _ in old_panos).values())
        expected = sample_panoramas(manifest, qc, old_per_city, args.seed, args.training_rows)
        if old_panos != {(row["city"], row["source_pano"]) for row in expected}:
            raise ValueError("baseline panoramas do not match the current QC and training exclusion")
        panos, anchors = expanded_anchors(manifest, qc, baseline, args.per_city, args.patches_per_pano, args.seed, args.training_rows)
    else:
        panos = sample_panoramas(manifest, qc, args.per_city, args.seed, args.training_rows)
        anchors = select_anchors(panos, args.patches_per_pano, args.seed)
    if not panos:
        raise ValueError("no verified held-out panoramas available")
    (outdir / "anchor_manifest.json").write_text(json.dumps(anchors))
    import hashlib
    digest = hashlib.sha256(json.dumps([str(manifest["source_out"]), anchors]).encode()).hexdigest()[:16]
    cache = Path(args.local_cache)
    cache.mkdir(parents=True, exist_ok=True)
    cache_path = cache / f"anchors_{digest}.npy"
    if cache_path.exists():
        features = np.load(cache_path, mmap_mode="r")
    else:
        features = read_anchor_features(anchors, Path(manifest["source_out"]), args.io_workers)
        tmp = cache / f"anchors_{digest}.tmp.npy"
        np.save(tmp, features)
        tmp.replace(cache_path)
    print(f"loaded {len(features):,} held-out anchors from {len(panos)} panos / {len({x['city'] for x in panos})} cities", flush=True)
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    checkpoints = dict(DEFAULT_CHECKPOINTS)
    for entry in args.checkpoint or []:
        width, path = entry.split("=", 1)
        checkpoints[int(width)] = path
    activations = {}
    decoders = {}
    losses = {}
    dictionaries = {}
    for width in (512, 1024, 2048):
        model, decoder, _ = load_model(Path(checkpoints[width]), device)
        acts, loss = encode_anchors(model, features, args.batch_size, device)
        activations[width] = acts
        decoders[width] = decoder
        losses[str(width)] = loss
        np.save(outdir / f"activations_w{width}.f16.npy", acts)
        counts = np.count_nonzero(acts > 0, axis=0)
        dictionaries[str(width)] = {"zero_support_features": int(np.count_nonzero(counts == 0)), "below_25_support_features": int(np.count_nonzero(counts < 25)), "supported_features": int(np.count_nonzero(counts >= 25)), "mean_active_features_per_patch": float(counts.sum() / len(features))}
        del model
        if device.type == "cuda":
            torch.cuda.empty_cache()
        print(f"W={width}: held-out cosine loss={loss:.5f}", flush=True)
    pair_summaries = {}
    pair_relations = {}
    sensitivities = []
    for left_width, right_width in ((512, 1024), (1024, 2048)):
        name = f"w{left_width}_to_w{right_width}"
        metrics = pair_matrices(activations[left_width], activations[right_width], decoders[left_width], decoders[right_width])
        supports = packed_support(activations[left_width], activations[right_width])
        np.savez_compressed(outdir / f"pair_metrics_{name}.npz", **metrics)
        results = classify(activations[left_width], activations[right_width], metrics, supports=supports)
        pair_relations[name] = results
        add_representatives(results, activations[left_width], activations[right_width], metrics)
        export_feature_relations(outdir / f"feature_relations_{name}.csv", results, metrics)
        (outdir / f"relations_{name}.json").write_text(json.dumps(results, indent=2))
        pair_summaries[name] = {"parent_counts": results["parent_counts"], "child_counts": results["child_counts"]}
        print(f"{name}: {pair_summaries[name]}", flush=True)
        for minimum in (10, 25, 50):
            for cosine in (.65, .70, .75):
                for coverage in (.40, .50, .60):
                    alternative = classify(activations[left_width], activations[right_width], metrics, minimum, cosine, coverage, supports)
                    sensitivities.append({"pair": name, "minimum_support": minimum, "stable_cosine": cosine, "split_coverage": coverage, "stable_pairs": len(alternative["stable"]), "split_parents": len(alternative["split_candidates"]), "no_close_match": len(alternative["no_close_match"]), "insufficient_support": len(alternative["insufficient_support"])})
    with (outdir / "threshold_sensitivity.csv").open("w", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=list(sensitivities[0]))
        writer.writeheader()
        writer.writerows(sensitivities)
    next_matches = {row["parent"]: row for row in pair_relations["w1024_to_w2048"]["stable"]}
    chains = []
    for row in pair_relations["w512_to_w1024"]["stable"]:
        if row["child"] in next_matches:
            next_row = next_matches[row["child"]]
            chains.append({"w512": row["parent"], "w1024": row["child"], "w2048": next_row["child"], "minimum_decoder_cosine": min(row["decoder_cosine"], next_row["decoder_cosine"]), "minimum_activation_cosine": min(row["activation_cosine"], next_row["activation_cosine"])})
    (outdir / "stable_three_width_chains.json").write_text(json.dumps(chains, indent=2))
    summary = {
        "sample": {"verified_cities": len({x["city"] for x in panos}), "panoramas": len(panos), "anchors": len(anchors), "panoramas_per_city_target": args.per_city, "patches_per_pano": args.patches_per_pano, "seed": args.seed, "excluded_training_cache_rows": args.training_rows, "batch_size": args.batch_size, "baseline_dir": str(args.baseline_dir) if args.baseline_dir else None},
        "heldout_cosine_loss": losses,
        "dictionary_activity": dictionaries,
        "stable_three_width_chains": len(chains),
        "pairs": pair_summaries,
        "thresholds": {"minimum_feature_support": 25, "stable_decoder_cosine": .70, "stable_activation_cosine": .35, "stable_support_jaccard": .15, "split_decoder_cosine": .45, "split_parent_recall_each": .15, "split_child_precision_each": .20, "split_parent_union_coverage": .50, "split_unique_parent_fraction_each": .10, "no_close_max_decoder_cosine": .45, "no_close_max_activation_cosine": .25},
        "interpretation": "Matching on held-out SAE-training-excluded patches; relation labels are operational candidates, not validated semantics. Stable and split are separate parent properties and may overlap; child categories are exclusive.",
        "local_feature_cache": str(cache_path),
        "checkpoints": {str(w): str(checkpoints[w]) for w in (512, 1024, 2048)},
        "limitations": ["Thresholds are exploratory; sensitivity table accompanies counts", "One initialization per width cannot establish seed stability", "Positive/negative image inspection is needed for semantic names", "Held-out refers to SAE training rows, not independent DINOv3 pretraining data"],
    }
    (outdir / "summary.json").write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=2), flush=True)


if __name__ == "__main__":
    main()
