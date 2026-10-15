"""Cluster, infer, and interpret the new panorama-aware ViT-L SAE only."""
from __future__ import annotations

import argparse
import hashlib
import heapq
import json
import os
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from PIL import Image, ImageDraw
from sklearn.cluster import AgglomerativeClustering

from sae_experiments.io_utils import digest, write_json
from sae_experiments.models.base_sae import BatchTopKSAE
from sae_experiments.panorama_vitl30 import (
    DIM, K, OUTPUT, PATCHES, SOURCE, TOKEN_ROOT, WIDTH, cohort, filtered_city,
)
from sae_experiments.panorama_vitl30_cache import cache_city, names as cache_names
from sae_experiments.activation_visualization import activation_overlay, add_legend, mark_peak

GROUPS = 64
TOP_PATCHES_PER_PANO = 128
TOP_PANOS_PER_GROUP_CITY = 12


def analysis_cities(args):
    if (args.output / "cached_feature_map.json").is_file():
        return cache_names(args.output)
    return cohort(args.source)


def analysis_city_data(args, city):
    if (args.output / "cached_feature_map.json").is_file():
        return cache_city(args.output, city)
    frame, identity = filtered_city(args.source, args.output, city)
    token_path = args.token_root / city.replace("/", "__") / "context.f16.npy"
    report = json.loads((token_path.parent / "report.json").read_text())
    tokens = np.load(token_path, mmap_mode="r")
    if report["identity"] != identity or tokens.shape != (len(frame) // 4, 4, PATCHES, DIM):
        raise ValueError(f"Context tokens do not match filtered panoramas: {city}")
    return frame, tokens, identity


def model_and_identity(output: Path, device: str):
    path = output / "model_best.pt"
    if not path.is_file():
        raise FileNotFoundError(f"New ViT-L panorama SAE not trained yet: {path}")
    checkpoint = torch.load(path, map_location="cpu", weights_only=False)
    if (checkpoint.get("D"), checkpoint.get("K"), checkpoint.get("k")) != (DIM, WIDTH, K):
        raise ValueError("Checkpoint is not the 1024/4096/k32 main model")
    model = BatchTopKSAE(DIM, WIDTH, K)
    model.load_state_dict(checkpoint["state"])
    model = model.eval().requires_grad_(False).to(device)
    return model, digest(path)


def cluster(args):
    model, model_hash = model_and_identity(args.output, "cpu")
    decoder = model.decoder.weight.detach().cpu().numpy().T
    labels = AgglomerativeClustering(n_clusters=GROUPS, metric="cosine", linkage="average").fit_predict(decoder)
    folder = args.output / "analysis"
    folder.mkdir(parents=True, exist_ok=True)
    path = folder / "decoder_groups.npy"
    np.save(path, labels.astype(np.int16))
    write_json(folder / "decoder_groups.json", {"checkpoint_sha256": model_hash,
        "groups": GROUPS, "method": "decoder cosine distance, average linkage",
        "group_sizes": np.bincount(labels, minlength=GROUPS).tolist(),
        "labels_sha256": digest(path)})
    print("clustered", WIDTH, "dimensions into", GROUPS, "groups", flush=True)


def checked_groups(output: Path, model_hash: str) -> np.ndarray:
    folder = output / "analysis"
    path = folder / "decoder_groups.npy"
    report = json.loads((folder / "decoder_groups.json").read_text())
    labels = np.load(path)
    if (report["checkpoint_sha256"] != model_hash or report["labels_sha256"] != digest(path)
            or labels.shape != (WIDTH,) or labels.min() < 0 or labels.max() >= GROUPS):
        raise ValueError("Decoder groups do not match the selected SAE checkpoint")
    return labels


def top_candidates(values: np.ndarray, latents: np.ndarray, groups: np.ndarray, city: str,
                   pid: int, panoid: str) -> list[dict]:
    if len(values) <= TOP_PATCHES_PER_PANO:
        patch_ids = np.arange(len(values))
    else:
        patch_ids = np.argpartition(values, -TOP_PATCHES_PER_PANO)[-TOP_PATCHES_PER_PANO:]
    selected = {}
    for patch in patch_ids:
        score = float(values[patch])
        if score <= 0:
            continue
        latent = int(latents[patch])
        group = int(groups[latent])
        if group not in selected or score > selected[group]["score"]:
            selected[group] = {"city": city, "pano_index": pid, "panoid": panoid,
                               "heading": int((patch // PATCHES) * 90),
                               "patch": int(patch % PATCHES), "latent": latent,
                               "group": group, "score": score}
    return list(selected.values())


def push_example(heaps: list[list], row: dict, limit: int):
    group = row["group"]
    key = (row["score"], row["pano_index"], row)
    if len(heaps[group]) < limit:
        heapq.heappush(heaps[group], key)
    elif key[:2] > heaps[group][0][:2]:
        heapq.heapreplace(heaps[group], key)


def saved_heaps(path: Path):
    heaps = [[] for _ in range(GROUPS)]
    if path.is_file():
        for row in json.loads(path.read_text()):
            push_example(heaps, row, TOP_PANOS_PER_GROUP_CITY)
    return heaps


def save_heaps(path: Path, heaps):
    rows = [item[2] for heap in heaps for item in heap]
    write_json(path, rows)


@torch.inference_mode()
def infer_city(args, city, model, model_hash, group_ids):
    frame, tokens, identity = analysis_city_data(args, city)
    slug = city.replace("/", "__")
    folder = args.output / "analysis" / "cities" / slug
    folder.mkdir(parents=True, exist_ok=True)
    final = folder / "panorama_group_share.f32.npy"
    partial = folder / "panorama_group_share.partial.npy"
    image_path = folder / "image_group_share.f32.npy"
    progress = folder / "progress.json"
    exemplars = folder / "top_examples.json"
    city_identity = hashlib.sha256(f"{identity}/{model_hash}/{digest(args.output / 'analysis/decoder_groups.npy')}".encode()).hexdigest()
    if final.is_file():
        state = json.loads(progress.read_text())
        if state["identity"] != city_identity or state["completed"] != len(tokens):
            raise ValueError(f"Completed inference identity mismatch: {city}")
        print("inference complete", city, flush=True)
        return
    if partial.is_file() and progress.is_file() and exemplars.is_file():
        state = json.loads(progress.read_text())
        if state["identity"] != city_identity:
            raise ValueError(f"Unsafe inference resume: {city}")
        shares = np.load(partial, mmap_mode="r+")
        start = int(state["completed"])
        heaps = saved_heaps(exemplars)
        image_shares = np.load(image_path, mmap_mode="r+")
    elif not any(x.exists() for x in (partial, progress, exemplars)):
        shares = np.lib.format.open_memmap(partial, mode="w+", dtype=np.float32, shape=(len(tokens), GROUPS))
        start = 0
        heaps = [[] for _ in range(GROUPS)]
        image_shares = np.lib.format.open_memmap(image_path, mode="w+", dtype=np.float32,
                                                 shape=(len(tokens), 4, GROUPS))
    else:
        raise ValueError(f"Partial inference has mismatched files: {city}")
    panoids = frame.panoid.astype(str).to_numpy()[::4]
    from concurrent.futures import ThreadPoolExecutor
    def read(pid):
        return np.array(tokens[pid], dtype=np.float32).reshape(-1, DIM)
    pool = ThreadPoolExecutor(max_workers=1)
    pending = pool.submit(read, start) if start < len(tokens) else None
    for pid in range(start, len(tokens)):
        host = pending.result()
        pending = pool.submit(read, pid + 1) if pid + 1 < len(tokens) else None
        x = torch.from_numpy(host).to(args.device)
        values_all, latent_all = [], []
        for chunk in x.split(args.patch_batch):
            with torch.autocast("cuda", dtype=torch.bfloat16, enabled=args.device.startswith("cuda")):
                acts = model.encode(chunk)
            values, latents = acts.max(dim=1)
            values_all.append(values.float().cpu().numpy())
            latent_all.append(latents.cpu().numpy())
        values = np.concatenate(values_all)
        latents = np.concatenate(latent_all)
        assigned = group_ids[latents]
        counts = np.bincount(assigned[values > 0], minlength=GROUPS).astype(np.float32)
        shares[pid] = counts / counts.sum() if counts.sum() else counts
        for h in range(4):
            active = values[h * PATCHES:(h + 1) * PATCHES] > 0
            counts = np.bincount(assigned[h * PATCHES:(h + 1) * PATCHES][active],
                                 minlength=GROUPS).astype(np.float32)
            image_shares[pid, h] = counts / counts.sum() if counts.sum() else counts
        if panoids[pid]:
            for row in top_candidates(values, latents, group_ids, city, pid, panoids[pid]):
                push_example(heaps, row, TOP_PANOS_PER_GROUP_CITY)
        if (pid + 1) % args.save_every == 0 or pid + 1 == len(tokens):
            shares.flush()
            image_shares.flush()
            save_heaps(exemplars, heaps)
            write_json(progress, {"city": city, "identity": city_identity, "completed": pid + 1,
                                  "total": len(tokens), "checkpoint_sha256": model_hash})
            print("infer", city, pid + 1, "/", len(tokens), flush=True)
    shares.flush()
    image_shares.flush()
    pool.shutdown()
    os.replace(partial, final)


def infer(args):
    torch.set_num_threads(8)
    model, model_hash = model_and_identity(args.output, args.device)
    groups = checked_groups(args.output, model_hash)
    for city in analysis_cities(args)[args.city_start:args.city_stop]:
        infer_city(args, city, model, model_hash, groups)


def merge_examples(args):
    _, model_hash = model_and_identity(args.output, "cpu")
    checked_groups(args.output, model_hash)
    all_rows = [[] for _ in range(GROUPS)]
    for city in analysis_cities(args):
        slug = city.replace("/", "__")
        folder = args.output / "analysis" / "cities" / slug
        if not (folder / "panorama_group_share.f32.npy").is_file():
            raise FileNotFoundError(f"Inference incomplete: {city}")
        for row in json.loads((folder / "top_examples.json").read_text()):
            all_rows[row["group"]].append(row)
    selected = []
    for group, rows in enumerate(all_rows):
        rows.sort(key=lambda r: r["score"], reverse=True)
        seen_cities = set()
        for row in rows:
            if row["city"] in seen_cities:
                continue
            selected.append(row)
            seen_cities.add(row["city"])
            if len(seen_cities) == args.examples_per_group:
                break
    write_json(args.output / "analysis/top_examples.json", selected)
    print("selected", len(selected), "examples across", GROUPS, "groups", flush=True)


@torch.inference_mode()
def draw_evidence(args, row: dict, sae, group_ids: np.ndarray) -> dict:
    frame, tokens, _ = analysis_city_data(args, row["city"])
    pid = int(row["pano_index"])
    paths = frame.image_path.iloc[pid * 4:pid * 4 + 4].tolist()
    patches = torch.from_numpy(np.asarray(tokens[pid], dtype=np.float32).reshape(-1, DIM)).to(args.device)
    feature_ids = torch.from_numpy(np.flatnonzero(group_ids == row["group"])).to(args.device)
    activation = []
    for chunk in patches.split(args.patch_batch):
        with torch.autocast("cuda", dtype=torch.bfloat16,
                            enabled=args.device.startswith("cuda") and getattr(args, "evidence_autocast", True)):
            acts = sae.encode(chunk)
        activation.append(acts[:, feature_ids].max(dim=1).values.float().cpu().numpy())
    heatmap = np.concatenate(activation).reshape(4, 28, 28)
    scale = max(float(heatmap.max()), 1e-12)
    images = []
    for path in paths:
        with Image.open(path) as source:
            images.append(source.convert("RGB").resize((448, 448), Image.Resampling.BICUBIC))
    heading_i = row["heading"] // 90
    patch = row["patch"]
    col, line = patch % 28, patch // 28
    x0, y0 = col * 16, line * 16
    marked = []
    for i, image in enumerate(images):
        marked.append(activation_overlay(image, heatmap[i], scale))
    mark_peak(marked[heading_i], (x0, y0, x0 + 16, y0 + 16))
    panorama = Image.new("RGB", (1792, 448))
    original = Image.new("RGB", (1792, 448))
    for i, image in enumerate(images):
        panorama.paste(marked[i], (448 * i, 0))
        original.paste(image, (448 * i, 0))
    located = original.copy()
    ImageDraw.Draw(located).rectangle(
        (heading_i * 448 + x0, y0, heading_i * 448 + x0 + 16, y0 + 16),
        outline=(255, 255, 255), width=2)
    cx, cy = x0 + 8, y0 + 8
    crop_box = (max(0, min(cx - 96, 256)), max(0, min(cy - 96, 256)))
    crop = images[heading_i].crop((crop_box[0], crop_box[1], crop_box[0] + 192, crop_box[1] + 192))
    activation_crop = marked[heading_i].crop((crop_box[0], crop_box[1], crop_box[0] + 192, crop_box[1] + 192))
    folder = getattr(args, "evidence_root", args.output / "analysis" / "evidence") / f"group_{row['group']:02d}"
    folder.mkdir(parents=True, exist_ok=True)
    stem = f"{row['city'].replace('/', '__')}_{pid}_{row['heading']}"
    pano_path, crop_path = folder / f"{stem}_panorama.jpg", folder / f"{stem}_crop.jpg"
    original_path = folder / f"{stem}_original.jpg"
    located_path = folder / f"{stem}_located.jpg"
    add_legend(panorama, scale).save(pano_path, quality=95)
    original.save(original_path, quality=88)
    located.save(located_path, quality=88)
    crop.save(crop_path, quality=92)
    activation_crop_path = folder / f"{stem}_activation_crop.jpg"
    activation_crop.save(activation_crop_path, quality=95)
    return {**row, "panorama_image": str(pano_path), "crop_image": str(crop_path),
            "activation_crop": str(activation_crop_path), "activation_scale": scale,
            "original_panorama": str(original_path), "located_panorama": str(located_path),
            "source_image": paths[heading_i]}


def interpret(args):
    from transformers import AutoProcessor, Qwen3VLForConditionalGeneration

    torch.set_num_threads(8)
    source = args.output / "analysis/top_examples.json"
    if not source.exists():
        raise FileNotFoundError("Run merge-examples after complete inference")
    rows = json.loads(source.read_text())
    sae, model_hash = model_and_identity(args.output, args.device)
    groups = checked_groups(args.output, model_hash)
    if not args.vlm_dir.is_dir():
        raise FileNotFoundError(f"VLM weights unavailable: {args.vlm_dir}")
    processor = AutoProcessor.from_pretrained(args.vlm_dir, local_files_only=True)
    vlm = Qwen3VLForConditionalGeneration.from_pretrained(args.vlm_dir, local_files_only=True,
        torch_dtype=torch.bfloat16 if args.device.startswith("cuda") else torch.float32).eval().to(args.device)
    by_group = {g: [] for g in range(GROUPS)}
    for row in rows:
        by_group[row["group"]].append(row)
    folder = args.output / "analysis/vlm"
    folder.mkdir(parents=True, exist_ok=True)
    for group, examples in by_group.items():
        target = folder / f"group_{group:02d}.json"
        if target.exists():
            old = json.loads(target.read_text())
            if old["checkpoint_sha256"] != model_hash:
                raise ValueError(f"Stale VLM interpretation: group {group}")
            continue
        if not examples:
            continue
        evidence = [draw_evidence(args, row, sae, groups) for row in examples[:args.examples_per_group]]
        prompt = ("Each numbered example consists of a four-direction ORIGINAL panorama followed by "
                  "one ORIGINAL crop from the marked view. A small white square locates the peak patch; "
                  "it is an annotation. Compare the depicted objects and surfaces near that square and "
                  "in the corresponding crop. Describe shared visual cues, not annotations. Do not assume "
                  "a single shared object category when examples differ. Return JSON with keys label, "
                  "semantic_type, recurring_cues, ambiguities, camera_artifacts, coherence. Use label=mixed "
                  "if evidence lacks a common referent. Do not infer city, country or unseen content.")
        content = []
        for number, item in enumerate(evidence, 1):
            content.append({"type": "text", "text": f"Example {number}: panorama, then local crop."})
            with Image.open(item["located_panorama"]) as panorama:
                overview = panorama.convert("RGB").resize((896, 224), Image.Resampling.BICUBIC)
            with Image.open(item["crop_image"]) as crop:
                detail = crop.convert("RGB")
            content.extend([{"type": "image", "image": overview},
                            {"type": "image", "image": detail}])
        content.append({"type": "text", "text": prompt})
        messages = [{"role": "user", "content": content}]
        inputs = processor.apply_chat_template(messages, tokenize=True, add_generation_prompt=True,
                                               return_dict=True, return_tensors="pt")
        inputs = {k: v.to(args.device) if hasattr(v, "to") else v for k, v in inputs.items()}
        with torch.inference_mode():
            response_ids = vlm.generate(**inputs, max_new_tokens=args.max_new_tokens, do_sample=False)
        response = processor.batch_decode(response_ids[:, inputs["input_ids"].shape[1]:],
                                          skip_special_tokens=True)[0]
        write_json(target, {"group": group, "checkpoint_sha256": model_hash,
                            "vlm": str(args.vlm_dir), "prompt": prompt,
                            "evidence": evidence, "raw_response": response})
        print("interpreted group", group, response[:100], flush=True)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("command", choices=["cluster", "infer", "merge-examples", "interpret"])
    p.add_argument("--source", type=Path, default=SOURCE)
    p.add_argument("--output", type=Path, default=OUTPUT)
    p.add_argument("--token-root", type=Path, default=TOKEN_ROOT)
    p.add_argument("--vlm-dir", type=Path, default=Path("/root/.cache/modelscope/models/Qwen--Qwen3-VL-2B-Instruct/snapshots/master"))
    p.add_argument("--device", default="cuda:3")
    p.add_argument("--city-start", type=int, default=0)
    p.add_argument("--city-stop", type=int, default=30)
    p.add_argument("--patch-batch", type=int, default=4 * PATCHES)
    p.add_argument("--save-every", type=int, default=100)
    p.add_argument("--examples-per-group", type=int, default=3)
    p.add_argument("--max-new-tokens", type=int, default=384)
    args = p.parse_args()
    if not 0 <= args.city_start < args.city_stop <= 30 or min(args.patch_batch, args.save_every,
            args.examples_per_group, args.max_new_tokens) <= 0:
        p.error("Invalid city slice or positive count")
    if args.command == "cluster":
        cluster(args)
    elif args.command == "infer":
        infer(args)
    elif args.command == "merge-examples":
        merge_examples(args)
    else:
        interpret(args)


if __name__ == "__main__":
    main()
