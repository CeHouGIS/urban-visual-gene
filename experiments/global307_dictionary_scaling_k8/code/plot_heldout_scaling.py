"""Plot held-out scale counts and original-image evidence for relation candidates."""
from __future__ import annotations

import argparse
import csv
import json
from collections import defaultdict
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
from PIL import Image

from formal.analysis.analyze_dictionary_scaling import DEFAULT_CHECKPOINTS
from formal.analysis.plot_feature_activation_maps import load_checkpoint, pano_id


def recover_selected_images(qc_root: Path, records: list[dict], wanted: set[int], qc: dict):
    cities = {records[i]["city"] for i in wanted}
    orders = defaultdict(list)
    paths = defaultdict(dict)
    for part in sorted(qc_root.glob("formal_sample_images.part*.tsv")):
        with part.open() as source:
            for line in source:
                city, path = line.rstrip("\n").split("\t", 1)
                if city not in cities:
                    continue
                pid, heading = pano_id(path)
                if not orders[city] or orders[city][-1] != pid:
                    orders[city].append(pid)
                paths[city].setdefault(pid, {})[heading] = path
    flags = defaultdict(lambda: defaultdict(lambda: [False, 0]))
    for part in sorted(qc_root.glob("qc_part*/images.part*.csv")):
        with part.open(newline="") as source:
            for row in csv.DictReader(source):
                city = row["dataset"]
                if city not in cities:
                    continue
                pid, _ = pano_id(row["path"])
                state = flags[city][pid]
                state[0] |= row.get("black", "0") == "1"
                state[1] += int(row.get("tunnel", "0") == "1")
    kept = {city: [pid for pid in ids if not flags[city][pid][0] and flags[city][pid][1] < 2] for city, ids in orders.items()}
    resolved = {}
    for i in wanted:
        row = records[i]
        city = row["city"]
        if len(kept.get(city, [])) != qc["cities"][city]["cached_panos"]:
            raise ValueError(f"image-order count verification failed for {city}")
        pid = kept[city][row["source_pano"]]
        image = paths[city][pid].get(row["heading"])
        if image:
            resolved[i] = {**row, "image": image, "pano_id": pid, "images": paths[city][pid]}
    return resolved


def select_examples(result_dir: Path, left: int, right: int, n: int):
    relations = json.loads((result_dir / f"relations_w{left}_to_w{right}.json").read_text())
    stable = sorted(relations["stable"], key=lambda x: x["activation_cosine"] * x["support_jaccard"], reverse=True)[:n]
    splits = sorted(relations["split_candidates"], key=lambda x: x["minimum_unique_parent_fraction"] * x["parent_coverage"], reverse=True)[:n]
    novel = relations["no_close_examples"][:n]
    return stable, splits, novel


def select_anchor(rows: list[int], anchors: list[dict]):
    # The square image comparison excludes equirectangular Baidu examples.
    baidu = {"Shanghai", "Hangzhou", "Nanjing", "Beijing", "Tianjin", "Guangzhou", "Shenzhen"}
    return next((i for i in rows if anchors[i]["city"] not in baidu), rows[0])


def response_maps(model, source: np.ndarray, feature_ids: list[int], device):
    x = torch.from_numpy(source.astype(np.float32)).to(device)
    with torch.inference_mode():
        dense = torch.relu(model._preactivate(x)).cpu().numpy()
    return [dense[:, feature].reshape(28, 28) for feature in feature_ids]


def draw_examples(result_dir: Path, source_root: Path, left: int, right: int, chosen, resolved, models, device, outdir: Path):
    saved = []
    stable, splits, novel = chosen
    with np.load(result_dir / f"pair_metrics_w{left}_to_w{right}.npz") as data:
        cos = data["decoder_cosine"]
        act_cos = data["activation_cosine"]
    configurations = []
    for relation in stable:
        configurations.append(("stable", relation, [(left, relation["parent"]), (right, relation["child"])], relation["parent_top_anchor_rows"]))
    for relation in splits:
        for child, rows in zip(relation["children"], relation.get("exclusive_shared_anchor_rows", relation["child_top_anchor_rows"])):
            configurations.append(("split", relation, [(left, relation["parent"])] + [(right, j) for j in relation["children"]], rows))
    for relation in novel:
        child = relation["feature"]
        old = int((.5 * cos[:, child] + .5 * act_cos[:, child]).argmax())
        configurations.append(("no_close_match", relation, [(left, old), (right, child)], relation["top_anchor_rows"]))
    for kind in ("stable", "split", "no_close_match"):
        entries = [x for x in configurations if x[0] == kind]
        if not entries:
            continue
        columns = 4 if kind == "split" else 3
        fig, axes = plt.subplots(len(entries), columns, figsize=(columns * 3.3, len(entries) * 3.3), squeeze=False)
        for line, (_, relation, features, rows) in enumerate(entries):
            index = select_anchor(rows, json.loads((result_dir / "anchor_manifest.json").read_text()))
            info = resolved.get(index)
            if info is None:
                for ax in axes[line]:
                    ax.axis("off")
                continue
            image = Image.open(info["image"]).convert("RGB").resize((448, 448))
            array = np.load(source_root / f"dict_{info['city']}_p784.f16.npy", mmap_mode="r")
            start = info["source_pano"] * 3136 + info["heading"] // 90 * 784
            x = np.asarray(array[start:start + 784]).copy()
            maps = [response_maps(models[width], x, [feature], device)[0] for width, feature in features]
            vmax = max(float(m.max()) for m in maps) or 1
            axes[line, 0].imshow(image)
            axes[line, 0].set_title(f"{info['city']} | {info['heading']} deg | anchor {index}", fontsize=9)
            for column, ((width, feature), heatmap) in enumerate(zip(features, maps), 1):
                axes[line, column].imshow(image)
                axes[line, column].imshow(heatmap, cmap="magma", alpha=.6, vmin=0, vmax=vmax, extent=(0, 448, 448, 0), interpolation="bilinear")
                axes[line, column].set_title(f"W={width}, feature {feature}", fontsize=10)
            for ax in axes[line]:
                ax.axis("off")
            saved.append({"pair": f"{left}->{right}", "relation_type": kind, "relation": relation, "anchor": index, "image_mapping": info, "overlay": "dense ReLU preactivation; shared vmax per image; relation statistics use sparse BatchTopK"})
        fig.suptitle(f"W={left} to W={right}: {kind} candidates\nSame image, dense ReLU preactivation (not sparse BatchTopK output)", fontsize=12)
        fig.tight_layout(rect=(0, 0, 1, .97))
        fig.savefig(outdir / f"{kind}_w{left}_w{right}.png", dpi=150, bbox_inches="tight")
        plt.close(fig)
    return saved


def draw_counts(summary: dict, outdir: Path):
    names = list(summary["pairs"])
    keys = ["stable", "split_child_not_stable", "no_close_match", "related_or_reconfigured", "insufficient_support"]
    colors = ["#258977", "#e4a446", "#b84461", "#6b74ad", "#bababa"]
    fig, axes = plt.subplots(1, 2, figsize=(13, 4.6))
    for x, name in enumerate(names):
        bottom = 0
        total = sum(summary["pairs"][name]["child_counts"].values())
        for key, color in zip(keys, colors):
            count = summary["pairs"][name]["child_counts"][key]
            axes[0].bar(x, count / total, bottom=bottom, color=color, label=key.replace("_", " ") if x == 0 else None)
            if count / total > .04:
                axes[0].text(x, bottom + count / total / 2, str(count), ha="center", va="center", fontsize=9)
            bottom += count / total
    axes[0].set_xticks(range(2), [x.replace("w", "W=").replace("_to_", " -> ") for x in names])
    axes[0].set_ylabel("fraction of larger dictionary")
    axes[0].legend(fontsize=7, loc="upper left", bbox_to_anchor=(1.01, 1))
    widths = list(summary["heldout_cosine_loss"])
    axes[1].plot(widths, [summary["heldout_cosine_loss"][w] for w in widths], "o-", color="#258977")
    axes[1].set_xlabel("dictionary width")
    axes[1].set_ylabel("held-out cosine reconstruction loss")
    fig.suptitle("Dictionary scaling on identical held-out street-view patches")
    fig.tight_layout()
    fig.savefig(outdir / "scaling_overview.png", dpi=170)
    plt.close(fig)


def draw_sensitivity(result_dir: Path, outdir: Path):
    rows = list(csv.DictReader((result_dir / "threshold_sensitivity.csv").open()))
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))
    for pair, color in (("w512_to_w1024", "#258977"), ("w1024_to_w2048", "#b84461")):
        selected = [r for r in rows if r["pair"] == pair and int(r["minimum_support"]) == 25]
        stable = [r for r in selected if float(r["split_coverage"]) == .50]
        split = [r for r in selected if float(r["stable_cosine"]) == .70]
        axes[0].plot([float(r["stable_cosine"]) for r in stable], [int(r["stable_pairs"]) for r in stable], "o-", label=pair, color=color)
        axes[1].plot([float(r["split_coverage"]) for r in split], [int(r["split_parents"]) for r in split], "o-", label=pair, color=color)
    axes[0].set_xlabel("stable decoder cosine threshold")
    axes[0].set_ylabel("stable one-to-one pairs")
    axes[1].set_xlabel("minimum parent coverage by child union")
    axes[1].set_ylabel("split candidates")
    for ax in axes:
        ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(outdir / "threshold_sensitivity.png", dpi=170)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--result-dir", required=True)
    parser.add_argument("--outdir", required=True)
    parser.add_argument("--qc-root", default="/host/root/mnt/nas/huangyj/GoogleSV/urban_visual_gene/formal_city_qc")
    parser.add_argument("--manifest", default="/host/root/mnt/nas/huangyj/GoogleSV/urban_visual_gene/formal_batchtopk_307city_qc_filtered/feature_qc_training_manifest.json")
    parser.add_argument("--examples", type=int, default=2)
    parser.add_argument("--device", default="cuda:1")
    args = parser.parse_args()
    result_dir = Path(args.result_dir)
    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    summary = json.loads((result_dir / "summary.json").read_text())
    draw_counts(summary, outdir)
    draw_sensitivity(result_dir, outdir)
    anchors = json.loads((result_dir / "anchor_manifest.json").read_text())
    pairs = {}
    wanted = set()
    for left, right in ((512, 1024), (1024, 2048)):
        chosen = select_examples(result_dir, left, right, args.examples)
        pairs[left, right] = chosen
        for row in chosen[0]:
            wanted.add(select_anchor(row["parent_top_anchor_rows"], anchors))
        for row in chosen[1]:
            for rows in row.get("exclusive_shared_anchor_rows", row["child_top_anchor_rows"]):
                wanted.add(select_anchor(rows, anchors))
        for row in chosen[2]:
            wanted.add(select_anchor(row["top_anchor_rows"], anchors))
    qc_root = Path(args.qc_root)
    qc = json.loads((qc_root / "feature_qc_map.json").read_text())
    print(f"recovering {len(wanted)} image anchors", flush=True)
    resolved = recover_selected_images(qc_root, anchors, wanted, qc)
    manifest = json.loads(Path(args.manifest).read_text())
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    models = {w: load_checkpoint(Path(DEFAULT_CHECKPOINTS[w]), device) for w in (512, 1024, 2048)}
    evidence = []
    for (left, right), chosen in pairs.items():
        evidence.extend(draw_examples(result_dir, Path(manifest["source_out"]), left, right, chosen, resolved, models, device, outdir))
    (outdir / "image_evidence.json").write_text(json.dumps(evidence, indent=2))
    print(f"saved {len(evidence)} evidence panels to {outdir}", flush=True)


if __name__ == "__main__":
    main()
