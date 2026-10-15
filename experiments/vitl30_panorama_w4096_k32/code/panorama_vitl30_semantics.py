"""Recover sparse group evidence and produce auditable post-hoc VLM annotations.

This analysis reuses the frozen epoch-10 SAE and contextualized ViT-L caches.
It does not change the dominant-patch numerical summaries or train a model.
"""
from __future__ import annotations

import argparse
import csv
import fcntl
import hashlib
import html
import json
import shutil
import time
from datetime import datetime, timezone
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch
from PIL import Image, ImageDraw

from sae_experiments.io_utils import digest, write_json
from sae_experiments.panorama_vitl30 import DIM, OUTPUT, PATCHES
from sae_experiments.panorama_vitl30_analysis import (
    GROUPS, checked_groups, draw_evidence, model_and_identity,
)
from sae_experiments.panorama_vitl30_cache import cache_city, names

VERSION = "v2_sparse_group_clean_crops"
PROMPT = """These images are unannotated crops around the strongest positive patch responses of one learned visual feature group. Each image comes from a different panorama. Focus on the object, surface, texture or visual pattern near each crop's center. Describe what actually recurs across examples. A common color alone does not establish a common object. Do not infer geography, brand, text that cannot be read, or hidden content. If examples have different referents, use label=mixed and list their visible categories in ambiguities. If only a texture recurs, name the texture rather than inventing an object.
Return ONLY compact JSON with exactly these keys:
{"label":"short visible category or mixed", "semantic_type":"object|surface_texture|vegetation|built_structure|camera_artifact|mixed", "recurring_cues":["up to 3 short cues"], "ambiguities":["up to 2 short issues"], "confidence":"low|medium|high"}.
An ordinary manufactured object belongs to object or built_structure; camera_artifact means blur, stitching, occlusion, or imaging defects. Confidence concerns this small evidence set only. Keep the entire answer under 130 words."""
TYPES = {"object", "surface_texture", "vegetation", "built_structure", "camera_artifact", "mixed"}


def stamp():
    return datetime.now(timezone.utc).isoformat()


def semantic_root(output):
    return output / "analysis" / "semantics_v2"


def atomic_json(path, data):
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix(path.suffix + ".tmp")
    write_json(temp, data)
    temp.replace(path)


def run_identity(output, model_hash, args):
    files = [output / "analysis/decoder_groups.npy", output / "cached_feature_map.json",
             output / "cache_manifests/report.json"]
    cfg = {"version": VERSION, "checkpoint": model_hash, "inputs": [digest(x) for x in files],
           "candidates_per_city_group": args.candidates_per_city_group,
           "uniform_per_city": args.uniform_per_city, "examples_per_group": args.examples_per_group,
           "patch_batch": 4 * PATCHES, "precision": "float32", "prompt": PROMPT}
    return hashlib.sha256(json.dumps(cfg, sort_keys=True).encode()).hexdigest(), cfg


def sparse_group_peaks(acts, group_ids, city, pid, panoid):
    """Consider every positive sparse latent, including non-dominant responses."""
    peaks, patches = acts.max(dim=0)
    peaks = peaks.float().cpu().numpy()
    patches = patches.cpu().numpy()
    rows = []
    for group in range(int(group_ids.max()) + 1):
        members = np.flatnonzero(group_ids == group)
        latent = int(members[np.argmax(peaks[members])])
        if peaks[latent] <= 0:
            continue
        patch = int(patches[latent])
        rows.append({"city": city, "pano_index": int(pid), "panoid": str(panoid),
                     "heading": (patch // PATCHES) * 90, "patch": patch % PATCHES,
                     "latent": latent, "group": group, "score": float(peaks[latent])})
    return rows


def candidate_ids(shares, previous, city, per_group, uniform):
    selected = set(np.linspace(0, len(shares) - 1, min(uniform, len(shares)), dtype=int).tolist())
    for group in range(GROUPS):
        positive = np.flatnonzero(shares[:, group] > 0)
        order = np.argsort(shares[positive, group], kind="stable")[-per_group:]
        selected.update(positive[order].tolist())
    selected.update(int(row["pano_index"]) for row in previous if row["city"] == city)
    return sorted(selected)


def diverse_examples(rows, limit):
    """Prefer different cities and dimensions, then fill from other panoramas."""
    ordered = sorted(rows, key=lambda x: (-x["score"], x["city"], x["pano_index"]))
    chosen, used_panos, cities, latents = [], set(), set(), set()
    for stage in range(3):
        for row in ordered:
            key = row["city"], row["pano_index"]
            if key in used_panos:
                continue
            if stage == 0 and (row["city"] in cities or row["latent"] in latents):
                continue
            if stage == 1 and row["city"] in cities:
                continue
            chosen.append(row)
            used_panos.add(key)
            cities.add(row["city"])
            latents.add(row["latent"])
            if len(chosen) >= limit:
                return chosen
    return chosen


@torch.inference_mode()
def recover(args):
    torch.set_num_threads(8)
    root = semantic_root(args.output)
    root.mkdir(parents=True, exist_ok=True)
    sae, model_hash = model_and_identity(args.output, args.device)
    group_ids = checked_groups(args.output, model_hash)
    identity, cfg = run_identity(args.output, model_hash, args)
    identity_file = root / "identity.json"
    if identity_file.exists() and json.loads(identity_file.read_text())["identity"] != identity:
        raise ValueError("Semantic retrieval settings changed; use a separate output version")
    atomic_json(identity_file, {"identity": identity, **cfg})
    archive = root / "previous_drafts"
    if not archive.exists():
        shutil.copytree(args.output / "analysis/vlm", archive)
        shutil.copy2(args.output / "analysis/vlm_review_report.json", archive / "review_report.json")
    previous = json.loads((args.output / "analysis/top_examples.json").read_text())
    all_rows, processed, total, eligible = [], 0, 0, []
    for city in names(args.output):
        frame, tokens, cache_identity = cache_city(args.output, city)
        if not frame.image_mapping_verified.all():
            print("EXCLUDE unmapped", city, flush=True)
            continue
        eligible.append(city)
        shares = np.load(args.output / "analysis/cities" / city.replace("/", "__") /
                         "panorama_group_share.f32.npy", mmap_mode="r")
        ids = candidate_ids(shares, previous, city, args.candidates_per_city_group, args.uniform_per_city)
        total += len(ids)
        dest = root / "retrieval" / f"{city.replace('/', '__')}.json"
        if dest.exists():
            saved = json.loads(dest.read_text())
            if (saved["identity"] != identity or saved["cache_identity"] != cache_identity
                    or saved["candidate_ids"] != ids):
                raise ValueError(f"Stale semantic retrieval: {city}")
            all_rows.extend(saved["rows"])
            processed += len(ids)
            print("REUSE", city, len(ids), flush=True)
            continue
        rows = []
        begin = time.monotonic()
        from concurrent.futures import ThreadPoolExecutor
        with ThreadPoolExecutor(max_workers=1) as pool:
            def read(pid):
                return np.array(tokens[pid], dtype=np.float32).reshape(-1, DIM)
            pending = pool.submit(read, ids[0]) if ids else None
            for number, pid in enumerate(ids):
                host = pending.result()
                pending = pool.submit(read, ids[number + 1]) if number + 1 < len(ids) else None
                x = torch.from_numpy(host).to(args.device)
                # Keep the same whole-panorama BatchTopK batch as the numeric analysis.
                # Float32 avoids additional evidence-ranking ties from bf16 rounding.
                acts = sae.encode(x)
                rows.extend(sparse_group_peaks(acts, group_ids, city, pid, frame.panoid.iloc[4 * pid]))
                del acts, x
                if (number + 1) % 25 == 0:
                    print("RETRIEVE", city, number + 1, "/", len(ids),
                          "seconds", round(time.monotonic() - begin, 1), flush=True)
        saved = {"identity": identity, "cache_identity": cache_identity, "city": city,
                 "candidate_ids": ids, "rows": rows, "seconds": time.monotonic() - begin}
        atomic_json(dest, saved)
        all_rows.extend(rows)
        processed += len(ids)
        print("CITY COMPLETE", city, len(ids), "group peaks", len(rows), flush=True)
    groups = {g: [] for g in range(GROUPS)}
    for row in all_rows:
        groups[row["group"]].append(row)
    selected = {str(g): diverse_examples(rows, args.examples_per_group) for g, rows in groups.items()}
    atomic_json(root / "selected_examples.json", selected)
    report = {"identity": identity, "checkpoint_sha256": model_hash, "version": VERSION,
              "candidate_panoramas": total, "completed_panoramas": processed,
              "eligible_cities": eligible, "excluded_cities": [x for x in names(args.output) if x not in eligible],
              "retrieved_groups": sum(bool(v) for v in selected.values()),
              "examples": sum(len(v) for v in selected.values()),
              "missing_groups": [int(g) for g, v in selected.items() if not v],
              "method": f"Top {args.candidates_per_city_group} dominant-share panoramas per group per verified city, {args.uniform_per_city} evenly spaced panorama indices per city, and previous retrieved examples; rank each group by its own maximum positive sparse activation over all 3136 patches and member latents.",
              "scope": "bounded evidence search, not an exhaustive global maximum or a proof of inactivity",
              "updated_utc": stamp()}
    atomic_json(root / "retrieval_report.json", report)
    print("RETRIEVAL COMPLETE", report, flush=True)


def focused_crop(image, cx, cy, side=96):
    # Keep actual pixels at image edges; artificial padding can look like an imaging defect.
    left = max(0, min(cx - side // 2, image.width - side))
    top = max(0, min(cy - side // 2, image.height - side))
    return image.crop((left, top, left + side, top + side))


@torch.inference_mode()
def evidence(args):
    root = semantic_root(args.output)
    identity = json.loads((root / "identity.json").read_text())["identity"]
    selected = json.loads((root / "selected_examples.json").read_text())
    sae, model_hash = model_and_identity(args.output, args.device)
    group_ids = checked_groups(args.output, model_hash)
    evidence_args = SimpleNamespace(output=args.output, device=args.device,
                                    patch_batch=4 * PATCHES, evidence_autocast=False,
                                    evidence_root=root / "evidence")
    dest = root / "evidence_manifest.json"
    old = json.loads(dest.read_text()) if dest.exists() else {"identity": identity, "groups": {}}
    if old["identity"] != identity:
        raise ValueError("Evidence identity mismatch")
    for group, examples in selected.items():
        if group in old["groups"] and all(Path(x["focused_crop"]).is_file() for x in old["groups"][group]):
            continue
        found = []
        for row in examples:
            item = draw_evidence(evidence_args, row, sae, group_ids)
            with Image.open(item["source_image"]) as source:
                view = source.convert("RGB").resize((448, 448), Image.Resampling.BICUBIC)
            cx, cy = row["patch"] % 28 * 16 + 8, row["patch"] // 28 * 16 + 8
            crop = focused_crop(view, cx, cy)
            folder = root / "evidence" / f"group_{int(group):02d}"
            folder.mkdir(parents=True, exist_ok=True)
            target = folder / f"{row['city'].replace('/', '__')}_{row['pano_index']}_{row['heading']}_focus.jpg"
            crop.save(target, quality=95)
            item["focused_crop"] = str(target)
            item["crop_side_at_448"] = 96
            found.append(item)
        old["groups"][group] = found
        atomic_json(dest, old)
        print("EVIDENCE", group, len(found), flush=True)


def parse_response(raw, truncated=False):
    if truncated:
        raise ValueError("Truncated generation")
    start, stop = raw.find("{"), raw.rfind("}")
    if start < 0 or stop < start:
        raise ValueError("Missing JSON object")
    obj = json.loads(raw[start:stop + 1])
    expected = {"label", "semantic_type", "recurring_cues", "ambiguities", "confidence"}
    if not isinstance(obj, dict) or set(obj) != expected:
        raise ValueError("Incorrect JSON keys")
    if not isinstance(obj["label"], str) or not 1 <= len(obj["label"]) <= 100:
        raise ValueError("Invalid label")
    if obj["semantic_type"] not in TYPES or obj["confidence"] not in {"low", "medium", "high"}:
        raise ValueError("Invalid enum")
    for name, limit in [("recurring_cues", 3), ("ambiguities", 2)]:
        values = obj[name]
        if not isinstance(values, list) or len(values) > limit or any(not isinstance(x, str) or len(x) > 240 for x in values):
            raise ValueError(f"Invalid {name}")
        if len({x.casefold().strip() for x in values}) != len(values):
            raise ValueError(f"Repetitive {name}")
    return obj


def complete_response(data, identity):
    if data.get("identity") != identity or data.get("status") != "interpreted":
        return False
    try:
        parsed = parse_response(data["raw_response"], data.get("truncated", False))
        return parsed == data.get("interpretation")
    except (KeyError, ValueError, TypeError):
        return False


def load_vlm(args):
    from accelerate import infer_auto_device_map, init_empty_weights
    from transformers import AutoConfig, AutoProcessor, Qwen3VLForConditionalGeneration
    config = AutoConfig.from_pretrained(args.vlm_dir, local_files_only=True)
    with init_empty_weights():
        skeleton = Qwen3VLForConditionalGeneration(config)
    skeleton.tie_weights()
    memories = {"cpu": "64GiB"}
    if torch.cuda.is_available():
        # Reserve CUDA context, activations, KV cache, and headroom for other jobs.
        for index in range(torch.cuda.device_count()):
            free, _ = torch.cuda.mem_get_info(index)
            available = min(1100, int(free / 2**20) - 650)
            if available >= 350:
                memories[index] = f"{available}MiB"
    mapping = infer_auto_device_map(skeleton, max_memory=memories, dtype=torch.bfloat16,
        no_split_module_classes=["Qwen3VLTextDecoderLayer", "Qwen3VLVisionBlock"])
    del skeleton
    all_cpu = not any(isinstance(v, int) for v in mapping.values())
    model = Qwen3VLForConditionalGeneration.from_pretrained(args.vlm_dir, local_files_only=True,
        dtype=torch.float32 if all_cpu else torch.bfloat16, device_map=mapping,
        attn_implementation="sdpa").eval()
    processor = AutoProcessor.from_pretrained(args.vlm_dir, local_files_only=True)
    input_device = next(model.parameters()).device
    return model, processor, mapping, input_device


@torch.inference_mode()
def interpret(args):
    root = semantic_root(args.output)
    meta = json.loads((root / "identity.json").read_text())
    manifest = json.loads((root / "evidence_manifest.json").read_text())
    if meta["identity"] != manifest["identity"]:
        raise ValueError("Evidence identity mismatch")
    torch.set_num_threads(8)
    folder = root / "vlm"
    folder.mkdir(parents=True, exist_ok=True)
    model, processor, mapping, input_device = load_vlm(args)
    atomic_json(root / "vlm_runtime.json", {"model": str(args.vlm_dir), "config_sha256": digest(args.vlm_dir / "config.json"),
        "weight_sha256": {p.name: digest(p) for p in sorted(args.vlm_dir.glob("model*.safetensors"))},
        "device_map": mapping, "input_device": str(input_device), "prompt": PROMPT, "updated_utc": stamp()})
    for group in range(GROUPS):
        target = folder / f"group_{group:02d}.json"
        prior = json.loads(target.read_text()) if target.exists() else None
        if prior and complete_response(prior, meta["identity"]):
            print("REUSE VLM", group, flush=True)
            continue
        examples = manifest["groups"].get(str(group), [])
        record = {"group": group, "identity": meta["identity"], "checkpoint_sha256": meta["checkpoint"],
                  "model": str(args.vlm_dir), "prompt": PROMPT, "evidence": examples,
                  "independently_validated": False, "updated_utc": stamp()}
        if not examples:
            atomic_json(target, {**record, "status": "insufficient_evidence", "interpretation": None,
                                 "reason": "No positive sparse activation in the bounded candidate set"})
            print("INSUFFICIENT", group, flush=True)
            continue
        attempts = prior.get("attempts", []) if prior else []
        for attempt in range(3):
            content = []
            for item in examples:
                with Image.open(item["focused_crop"]) as image:
                    clean = image.convert("RGB").resize((192, 192), Image.Resampling.BICUBIC)
                content.append({"type": "image", "image": clean})
            prompt = PROMPT
            if len(examples) == 1:
                prompt += " Only one example was retrieved for this group. Describe that example conservatively; it cannot establish a recurring group-level meaning. Use confidence=low and mention the single-example limitation in ambiguities."
            if attempt:
                prompt += " Previous output did not satisfy the schema. Use exactly the allowed semantic_type values and no more than three cues and two ambiguities. Complete all fields."
            content.append({"type": "text", "text": prompt})
            inputs = processor.apply_chat_template([{"role": "user", "content": content}], tokenize=True,
                add_generation_prompt=True, return_dict=True, return_tensors="pt")
            inputs = {k: v.to(input_device) if hasattr(v, "to") else v for k, v in inputs.items()}
            start = time.monotonic()
            limit = 256 + attempt * 64
            tokens = model.generate(**inputs, max_new_tokens=limit, do_sample=False,
                                    repetition_penalty=1.1, no_repeat_ngram_size=10)
            generated = tokens[:, inputs["input_ids"].shape[1]:]
            raw = processor.batch_decode(generated, skip_special_tokens=True)[0]
            eos = model.generation_config.eos_token_id
            eos = {eos} if isinstance(eos, int) else set(eos)
            truncated = int(generated[0, -1]) not in eos and generated.shape[1] >= limit
            outcome = {"prompt": prompt, "raw_response": raw, "truncated": truncated,
                       "tokens": generated.shape[1], "seconds": time.monotonic() - start}
            try:
                parsed = parse_response(raw, truncated)
                outcome["schema_valid"] = True
            except (ValueError, TypeError) as exc:
                outcome["schema_valid"] = False
                outcome["error"] = str(exc)
            attempts.append(outcome)
            if outcome["schema_valid"]:
                atomic_json(target, {**record, **outcome, "attempts": attempts, "status": "interpreted",
                                     "interpretation": parsed})
                print("VLM", group, parsed["label"], parsed["confidence"],
                      round(outcome["seconds"], 1), "seconds", flush=True)
                break
            atomic_json(target, {**record, **outcome, "attempts": attempts, "status": "response_invalid",
                                 "interpretation": None})
            print("RETRY VLM", group, outcome["error"], flush=True)
    print("VLM STAGE COMPLETE", flush=True)


def checked_visual_review(root, identity, manifest):
    """Bind the separate automated review to the actual inspected clean crops."""
    target = root / "visual_review.json"
    if not target.exists():
        return {}, None
    review = json.loads(target.read_text())
    if review["identity"] != identity or review.get("independent_human_validation") is not False:
        raise ValueError("Visual review identity or provenance mismatch")
    entries = {entry["group"]: entry for entry in review["groups"]}
    if set(entries) != set(range(GROUPS)) or len(review["groups"]) != GROUPS:
        raise ValueError("Visual review must cover exactly 64 groups")
    for group, entry in entries.items():
        actual = manifest["groups"][str(group)]
        inspected = entry["evidence"]
        if len(actual) != len(inspected):
            raise ValueError(f"Changed review evidence count: {group}")
        for image, source in zip(inspected, actual):
            for key in ["city", "pano_index", "latent", "focused_crop"]:
                if image[key] != source[key]:
                    raise ValueError(f"Changed review evidence: {group}, {key}")
            if digest(Path(source["focused_crop"])) != image["sha256"]:
                raise ValueError(f"Changed reviewed crop pixels: {group}")
    return entries, {k: v for k, v in review.items() if k != "groups"}


def export(args):
    root = semantic_root(args.output)
    identity = json.loads((root / "identity.json").read_text())
    retrieval = json.loads((root / "retrieval_report.json").read_text())
    manifest = json.loads((root / "evidence_manifest.json").read_text())
    if manifest["identity"] != identity["identity"]:
        raise ValueError("Export evidence identity mismatch")
    reviews, review_meta = checked_visual_review(root, identity["identity"], manifest)
    groups = np.load(args.output / "analysis/decoder_groups.npy")
    rows, records = [], []
    for group in range(GROUPS):
        data = json.loads((root / "vlm" / f"group_{group:02d}.json").read_text())
        if data["identity"] != identity["identity"]:
            raise ValueError("Export identity mismatch")
        if data["status"] == "interpreted" and not complete_response(data, identity["identity"]):
            raise ValueError(f"Invalid response marked complete: {group}")
        interpretation = data.get("interpretation") or {}
        examples = manifest["groups"].get(str(group), [])
        # Heatmaps may be redrawn while inference uses an already-loaded manifest.
        # Merge display metadata only after verifying that clean VLM inputs agree.
        if [x["focused_crop"] for x in examples] != [x["focused_crop"] for x in data.get("evidence", [])]:
            raise ValueError(f"VLM evidence changed: {group}")
        data["evidence"] = examples
        review = reviews.get(group, {})
        if review:
            data["visual_review"] = review
            data["visual_review_provenance"] = review_meta
        row = {"group": group, "dictionary_dimensions": int((groups == group).sum()),
               "status": data["status"], "label": interpretation.get("label", "insufficient evidence"),
               "semantic_type": interpretation.get("semantic_type", ""),
               "confidence": interpretation.get("confidence", ""), "examples": len(examples),
               "cities": len({x["city"] for x in examples}), "represented_dimensions": len({x["latent"] for x in examples}),
               "evidence_scope": "single_example" if len(examples) == 1 else "multiple_examples" if examples else "no_examples",
               "peak_score_min": min((x["score"] for x in examples), default=0),
               "peak_score_max": max((x["score"] for x in examples), default=0),
               "recurring_cues": " | ".join(interpretation.get("recurring_cues", [])),
               "ambiguities": " | ".join(interpretation.get("ambiguities", [])),
               "reviewed_label": review.get("label", ""),
               "reviewed_semantic_type": review.get("semantic_type", ""),
               "review_evidence_pattern": review.get("evidence_pattern", ""),
               "review_visible_categories": " | ".join(review.get("visible_categories", [])),
               "review_caveat": review.get("caveat", ""),
               "reviewer": (review_meta or {}).get("reviewer", ""),
               "independently_validated": False}
        rows.append(row)
        records.append(data)
    target = root / "group_interpretations.csv"
    with target.open("w", newline="", encoding="utf-8-sig") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    atomic_json(root / "group_interpretations.json", records)
    report = {"status": "complete_provisional" if all(x["status"] in {"interpreted", "insufficient_evidence"} for x in rows) else "needs_retry",
              "version": VERSION, "identity": identity["identity"], "groups": GROUPS,
              "interpreted_groups": sum(x["status"] == "interpreted" for x in rows),
              "mixed_groups": [x["group"] for x in rows if x["label"].lower() == "mixed" or x["semantic_type"] == "mixed"],
              "insufficient_evidence_groups": [x["group"] for x in rows if x["status"] == "insufficient_evidence"],
              "single_example_groups": [x["group"] for x in rows if x["examples"] == 1],
              "response_invalid_groups": [x["group"] for x in rows if x["status"] == "response_invalid"],
              "visual_review": review_meta,
              "reviewed_groups": len(reviews),
              "review_mixed_groups": [g for g, r in reviews.items() if r["evidence_pattern"] == "mixed"],
              "review_recurring_groups": [g for g, r in reviews.items() if r["evidence_pattern"] == "recurring"],
              "review_predominant_groups": [g for g, r in reviews.items() if r["evidence_pattern"] == "predominant_with_outliers"],
              "retrieval": retrieval, "independently_validated": False,
              "caveat": "Generated descriptions and the separate Codex visual review apply to a bounded set of high-response examples, not all member dimensions. Raw Qwen answers are preserved. Neither automated review nor model confidence establishes calibrated semantic accuracy. No independent human semantic annotation has been conducted.",
              "updated_utc": stamp()}
    atomic_json(root / "report.json", report)
    gallery(root, rows, records, report)
    print("EXPORT", report, flush=True)


def gallery(root, rows, records, report):
    def escape(value):
        return html.escape(str(value))
    cards = []
    for row, record in zip(rows, records):
        tiles = []
        for item in record.get("evidence", []):
            def url(key):
                import os
                return escape(Path(os.path.relpath(item[key], root)).as_posix())
            activation = f' · <a href="{url("activation_crop")}" target="_blank">Activation detail</a>' if "activation_crop" in item else ""
            tiles.append(f'<div><a href="{url("panorama_image")}" target="_blank"><img loading="lazy" src="{url("focused_crop")}" alt="clean crop"><span>{escape(item["city"])}<br>dimension {item["latent"]}, heading {item["heading"]}, patch {item["patch"]}<br>response {item["score"]:.3f}</span></a><p><a href="{url("original_panorama")}" target="_blank">Original four views</a> · <a href="{url("crop_image")}" target="_blank">Context crop</a>{activation}</p></div>')
        text = ' '.join(str(v) for v in row.values()).lower()
        reviewed = row["reviewed_label"] or row["label"]
        review_text = f'<p><b>Automated visual review:</b> {escape(row["review_evidence_pattern"])}. Visible categories: {escape(row["review_visible_categories"])}.</p><p>{escape(row["review_caveat"])}</p>' if row["reviewed_label"] else ""
        cards.append(f'<section id="g{row["group"]:02d}" data-search="{escape(text)}"><h2>G{row["group"]:02d} · {escape(reviewed)}</h2><p>{row["dictionary_dimensions"]} dictionary dimensions · {row["examples"]} panoramas · {row["cities"]} cities</p>{review_text}<div class="tiles">{"".join(tiles)}</div><details><summary>Qwen draft: {escape(row["label"])} · self-reported confidence: {escape(row["confidence"] or "n/a")}</summary><p>Cues: {escape(row["recurring_cues"])}<br>Ambiguities: {escape(row["ambiguities"])}</p><pre>{escape(record.get("raw_response", record.get("reason", "")))}</pre></details></section>')
    page = '<!doctype html><meta charset="utf-8"><meta name="viewport" content="width=device-width, initial-scale=1"><title>ViT-L panorama SAE · VLM evidence</title><style>body{font:16px system-ui;background:#f4f5f7;color:#222;margin:24px auto;max-width:1400px;padding:0 20px}section{background:white;border-radius:10px;padding:20px;margin:20px 0}h2{font-size:22px}.tiles{display:grid;grid-template-columns:repeat(auto-fit,minmax(170px,1fr));gap:12px}.tiles a{color:#444;text-decoration:none;font-size:12px}.tiles img{width:100%;aspect-ratio:1;object-fit:contain;background:#ddd}.tiles span{display:block;margin-top:6px}input{width:95%;padding:12px;font-size:17px}pre{white-space:pre-wrap}header{position:sticky;top:0;background:#f4f5f7;padding:10px 0;z-index:1}</style>'
    page += f'<header><h1>Panorama-aware BatchTopK SAE: VLM drafts and visual evidence</h1><p>{report["interpreted_groups"]}/64 Qwen drafts completed; {report["reviewed_groups"]}/64 groups visually reviewed with Codex image tools. Click a clean crop to inspect the four-view activation overlay.</p><input id="search" placeholder="Filter by group, label, cue, or ambiguity"></header><p>Headings show separate conservative visual-review descriptions; raw Qwen drafts remain expandable. Neither is independent human validation. Dictionary clustering and numerical results are fixed independently. Retrieval searches {report["retrieval"]["candidate_panoramas"]:,} candidate panoramas in 29 verified cities. Colors run from purple to orange to yellow on a gray background and are normalized within each panorama; raw peak responses are shown. Model confidence has not been calibrated.</p>'
    page += ''.join(cards) + '<script>document.getElementById("search").addEventListener("input",e=>{let q=e.target.value.toLowerCase();document.querySelectorAll("section").forEach(x=>x.hidden=!x.dataset.search.includes(q))})</script>'
    (root / "index.html").write_text(page)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=["recover", "evidence", "interpret", "export", "all"])
    parser.add_argument("--output", type=Path, default=OUTPUT)
    parser.add_argument("--device", default="cuda:2")
    parser.add_argument("--candidates-per-city-group", type=int, default=2)
    parser.add_argument("--uniform-per-city", type=int, default=16)
    parser.add_argument("--examples-per-group", type=int, default=6)
    parser.add_argument("--vlm-dir", type=Path, default=Path("/root/.cache/modelscope/models/Qwen--Qwen3-VL-2B-Instruct/snapshots/master"))
    args = parser.parse_args()
    root = semantic_root(args.output)
    root.mkdir(parents=True, exist_ok=True)
    with (root / "run.lock").open("w") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        for name, action in [("recover", recover), ("evidence", evidence), ("interpret", interpret), ("export", export)]:
            if args.command in {name, "all"}:
                action(args)
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
