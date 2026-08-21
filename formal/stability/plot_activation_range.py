"""Summarize cross-seed activation coverage and render representative families."""
from __future__ import annotations

import csv
from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib.cm as cm
import numpy as np
import torch
from PIL import Image

from formal.gpu_run import imgpath
from sae_experiments.models.base_sae import build_sae


ROOT = Path(__file__).resolve().parents[2]
RUN_ROOT = ROOT / "formal" / "seed_stability_w1024_k8"
RESULTS = RUN_ROOT / "results"
FIGURES = ROOT / "formal" / "figures"
SEEDS = (11, 23, 37, 53, 71)
N_ANCHORS = 250_000
COLORS = {11: "#39d6ff", 23: "#a58bff", 37: "#22e0a1", 53: "#f6c945", 71: "#ff7488"}


def load_members() -> dict[str, dict[int, int]]:
    families: dict[str, dict[int, int]] = {}
    with (RESULTS / "stable_gene_members.csv").open(newline="") as handle:
        for row in csv.DictReader(handle):
            families.setdefault(row["family_id"], {})[int(row["seed"])] = int(row["gene_id"])
    return families


def load_seed(seed: int) -> dict[str, np.ndarray]:
    with np.load(RUN_ROOT / f"seed_{seed:03d}" / "anchor_activations.npz") as z:
        return {key: z[key].copy() for key in ("rows", "cols", "vals", "support")}


def activation_values(payload: dict[str, np.ndarray], gene: int) -> np.ndarray:
    return payload["vals"][payload["cols"] == gene].astype(np.float32)


def decoder_atoms(checkpoint: Path) -> np.ndarray:
    payload = torch.load(checkpoint, map_location="cpu")
    weight = payload["state"]["decoder.weight"].detach().float().numpy().T
    return weight / (np.linalg.norm(weight, axis=1, keepdims=True) + 1e-9)


def load_sae(checkpoint: Path) -> torch.nn.Module:
    payload = torch.load(checkpoint, map_location="cpu")
    config = {
        "type": payload["model_type"],
        "latent_dim": int(payload["K"]),
        "k": int(payload.get("k", payload.get("topk"))),
        "decoder_unit_norm": True,
    }
    model = build_sae(int(payload["D"]), config)
    model.load_state_dict(payload["state"])
    return model.eval()


def activation_map(model: torch.nn.Module, features: torch.Tensor, gene: int) -> np.ndarray:
    with torch.inference_mode():
        acts, _ = model(features)
    return acts[:, gene].float().numpy().reshape(28, 28)


def activation_overlay(image: Image.Image, values: np.ndarray, size: int = 224) -> Image.Image:
    scaled = values.astype(np.float32)
    positive = scaled[scaled > 0]
    ceiling = float(np.quantile(positive, 0.98)) if len(positive) else 1.0
    scaled = np.clip(scaled / max(ceiling, 1e-8), 0, 1)
    up = np.asarray(
        Image.fromarray((scaled * 255).astype(np.uint8)).resize((size, size), Image.BILINEAR),
        dtype=np.float32,
    ) / 255
    heat = (cm.inferno(up)[..., :3] * 255).astype(np.float32)
    rgb = np.asarray(image.resize((size, size)), dtype=np.float32)
    alpha = (0.08 + 0.74 * up)[..., None]
    return Image.fromarray((rgb * (1 - alpha) + heat * alpha).astype(np.uint8))


def representative_examples(
    chosen: dict[str, dict], families: dict[str, dict[int, int]]
) -> tuple[dict[str, Image.Image], torch.Tensor]:
    """Pick one common exemplar per family and reconstruct its patch features."""
    checkpoint = ROOT / "formal" / "batchtopk_w1024_k8" / "batch_topk_w1024_k8.pt"
    baseline = decoder_atoms(checkpoint)
    seed11 = decoder_atoms(RUN_ROOT / "seed_011" / "batch_topk_w1024_k8.pt")
    with np.load(ROOT / "formal" / "batchtopk_w1024_k8" / "sparse_acts.npz") as z:
        idx, val = z["idx"], z["val"]
        city, pano, heading = z["city"], z["pano"], z["heading"]
        output: dict[str, Image.Image] = {}
        selected_rows: list[int] = []
        for _, record in chosen.items():
            family = record["family"]
            seed_gene = families[family][11]
            baseline_gene = int(np.argmax(baseline @ seed11[seed_gene]))
            scores = np.where(idx == baseline_gene, val, 0).max(axis=(1, 2))
            for row in np.argsort(scores)[::-1]:
                source = imgpath(str(city[row]), str(pano[row]), int(heading[row]))
                if source.exists():
                    output[family] = Image.open(source).convert("RGB")
                    selected_rows.append(int(row))
                    break
            if family not in output:
                raise FileNotFoundError(f"no source exemplar found for {family}")
        latent = np.zeros((len(selected_rows), idx.shape[1], 1024), dtype=np.float32)
        for output_row, source_row in enumerate(selected_rows):
            np.put_along_axis(latent[output_row], idx[source_row].astype(np.int64), val[source_row].astype(np.float32), axis=1)
    payload = torch.load(checkpoint, map_location="cpu")
    state = payload["state"]
    decoder = state["decoder.weight"].detach().float()
    bias = state["pre_bias"].detach().float()
    with torch.inference_mode():
        reconstructed = torch.from_numpy(latent) @ decoder.T + bias
    return output, reconstructed


def main() -> None:
    FIGURES.mkdir(parents=True, exist_ok=True)
    families = load_members()
    seed_data = {seed: load_seed(seed) for seed in SEEDS}

    records = []
    for family, members in families.items():
        supports = [int(seed_data[s]["support"][members[s]]) if s in members else 0 for s in SEEDS]
        rates = np.asarray(supports, dtype=float) / N_ANCHORS * 100
        records.append(
            {
                "family": family,
                "coverage": len(members),
                "supports": supports,
                "rates": rates,
                "range_pp": float(rates.max() - rates.min()),
                "mean_pct": float(rates.mean()),
                "cv": float(np.std(rates) / np.mean(rates)) if np.mean(rates) else np.nan,
            }
        )
    records.sort(key=lambda r: r["range_pp"])

    stats_path = RESULTS / "activation_range_by_family.csv"
    with stats_path.open("w", newline="") as handle:
        fields = ["family_id", "seed_coverage", *[f"seed_{s}_activation_pct" for s in SEEDS], "mean_activation_pct", "range_pp", "coefficient_of_variation"]
        writer = csv.DictWriter(handle, fieldnames=fields, lineterminator="\n")
        writer.writeheader()
        for r in records:
            writer.writerow(
                {
                    "family_id": r["family"],
                    "seed_coverage": r["coverage"],
                    **{f"seed_{s}_activation_pct": f"{r['rates'][i]:.6f}" for i, s in enumerate(SEEDS)},
                    "mean_activation_pct": f"{r['mean_pct']:.6f}",
                    "range_pp": f"{r['range_pp']:.6f}",
                    "coefficient_of_variation": f"{r['cv']:.6f}",
                }
            )

    five = [r for r in records if r["coverage"] == 5]
    chosen = {
        "Most consistent": five[0],
        "Typical (median range)": five[len(five) // 2],
        "Largest range (5/5)": five[-1],
        "Roadside-greenery example": next(r for r in records if r["family"] == "stable_0122"),
    }
    selected_ids = {r["family"] for r in chosen.values()}

    fig, ax = plt.subplots(figsize=(18, 7))
    fig.subplots_adjust(left=0.07, right=0.985, bottom=0.12, top=0.82)
    x = np.arange(len(records))
    bars = ax.bar(
        x,
        [r["range_pp"] for r in records],
        width=1.0,
        color=["#39d6ff" if r["coverage"] == 5 else "#f6c945" for r in records],
        alpha=0.84,
        linewidth=0,
    )
    for i, r in enumerate(records):
        if r["family"] in selected_ids:
            bars[i].set_color("#ff7488")
            bars[i].set_alpha(1)
            ax.annotate(r["family"], (i, r["range_pp"]), xytext=(0, 7), textcoords="offset points", rotation=65, ha="left", fontsize=8)
    fig.suptitle("Cross-seed activation-coverage range for every stable visual-gene family", x=0.07, y=0.965, ha="left", fontsize=17, weight="bold")
    fig.text(0.07, 0.915, "Range = max(seed activation rate) − min(seed activation rate), evaluated on the same 250,000 anchor patches", color="#58677c", fontsize=10)
    ax.set_xlabel("Stable families sorted by activation-rate range")
    ax.set_ylabel("Activation-rate range (percentage points)")
    ax.set_xticks([])
    ax.spines[["top", "right"]].set_visible(False)
    ax.grid(axis="y", alpha=0.22)
    ax.scatter([], [], color="#39d6ff", label=f"5/5 seeds (n={sum(r['coverage']==5 for r in records)})")
    ax.scatter([], [], color="#f6c945", label=f"4/5 seeds (n={sum(r['coverage']==4 for r in records)})")
    ax.scatter([], [], color="#ff7488", label="Families used in montage")
    ax.legend(frameon=False, ncol=3, loc="upper left")
    fig.savefig(FIGURES / "sae_seed_activation_range_bars.png", dpi=220, facecolor="white")
    plt.close(fig)

    originals, feature_batch = representative_examples(chosen, families)
    models = {
        seed: load_sae(RUN_ROOT / f"seed_{seed:03d}" / "batch_topk_w1024_k8.pt")
        for seed in SEEDS
    }

    fig, axes = plt.subplots(4, 5, figsize=(20, 9.5), constrained_layout=True)
    for row_idx, (role, record) in enumerate(chosen.items()):
        family = record["family"]
        members = families[family]
        for col_idx, seed in enumerate(SEEDS):
            ax = axes[row_idx, col_idx]
            gene = members.get(seed)
            if gene is None:
                ax.text(0.5, 0.52, "No strict match", ha="center", va="center", transform=ax.transAxes, color="#c44e52", weight="bold")
            else:
                gene_map = activation_map(models[seed], feature_batch[row_idx], gene)
                original = originals[family].resize((224, 224))
                overlay = activation_overlay(original, gene_map)
                pair = np.concatenate([np.asarray(original), np.asarray(overlay)], axis=1)
                ax.imshow(pair)
                active = int(np.count_nonzero(gene_map))
                ax.text(0.02, 0.04, "ORIGINAL", transform=ax.transAxes, color="white", fontsize=7, weight="bold", bbox={"facecolor": "#172033", "alpha": 0.78, "pad": 2, "edgecolor": "none"})
                ax.text(0.52, 0.04, f"ACTIVATION · Gene {gene} · {active}/784 patches", transform=ax.transAxes, color="white", fontsize=7, weight="bold", bbox={"facecolor": "#172033", "alpha": 0.78, "pad": 2, "edgecolor": "none"})
            if row_idx == 0:
                ax.set_title(f"Seed {seed}", color=COLORS[seed], weight="bold")
            if col_idx == 0:
                ax.set_ylabel(f"{role}\n{family}\nrange {record['range_pp']:.3f} pp", fontsize=9, weight="bold")
            ax.set_xticks([])
            ax.set_yticks([])
            for spine in ax.spines.values():
                spine.set_visible(False)
    fig.suptitle("Four representative stable families: original image + SAE activation across five seeds", x=0.01, ha="left", fontsize=18, weight="bold")
    fig.savefig(FIGURES / "sae_seed_representative_gene_montage.png", dpi=220, facecolor="white")
    plt.close(fig)

    print(f"wrote {stats_path}")
    print(f"wrote {FIGURES / 'sae_seed_activation_range_bars.png'}")
    print(f"wrote {FIGURES / 'sae_seed_representative_gene_montage.png'}")
    for role, r in chosen.items():
        print(role, r["family"], r["coverage"], f"range={r['range_pp']:.6f} pp")


if __name__ == "__main__":
    main()
