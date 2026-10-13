"""Recompute paper-style figures and analysis tables from the new SAE outputs.

Outputs have their own directory and names; no Feature-MAE picture is reused.
Unsupported legacy figures are listed explicitly in figure_inventory.json.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import pandas as pd
from scipy.cluster.hierarchy import dendrogram
from scipy.spatial.distance import jensenshannon
from sklearn.decomposition import PCA

from .train import CITIES, DEFAULT_OUTPUT, EXPECTED_IMAGES, ROOT, write_json

LEGACY_FIGURES = ROOT / "archive/non_batchtopk/paper/figures"


def inventory(output: Path, generated: list[str]) -> dict:
    figures = sorted({p.stem for folder in ("main", "supplementary")
                      for p in (LEGACY_FIGURES / folder).glob("Fig*.*")
                      if p.suffix.lower() in (".png", ".pdf", ".jpg")})
    new_mapping = {
        "Fig_MAE_Hierarchy_EDP_32_64": "Fig_BatchTopK_Hierarchy_32_64",
        "Fig_All_City_UMAP_512_Dimensions": "Fig_BatchTopK_UMAP_4096_Dimensions",
        "Fig_Image_Dimension_Coexistence": "Fig_BatchTopK_Image_Feature_Coexistence",
        "Fig_Image_Dimension_PPMI_512": "Fig_BatchTopK_Image_Feature_PPMI_64",
    }
    metric_notes = {
        "Fig_MAE_Hierarchy_EDP_32_64": "Decoder cosine hierarchy replaces the old E+D+P relation graph.",
        "Fig_Image_Dimension_Coexistence": "Current figure shows feature prevalence only; old spatial-share and pair panels are not reproduced.",
        "Fig_Image_Dimension_PPMI_512": "PPMI is computed on 64 SAE clusters, not 512 MAE dimensions.",
    }
    records = []
    for original in figures:
        destination = new_mapping.get(original, original)
        available = destination in generated
        records.append({"legacy_figure": original, "new_figure": destination if available else None,
                        "status": "recomputed_from_SAE" if available else "requires_additional_analysis_or_inputs",
                        "reason": metric_notes.get(original) if available else "requires its own semantic, segmentation, graph-archetype or layout analysis on the new cohort"})
    result = {"original_figure_count": len(figures), "recomputed_count": sum(r["status"] == "recomputed_from_SAE" for r in records),
              "figures": records, "note": "Feature-MAE originals are preserved separately and are never relabeled as SAE."}
    write_json(output / "figure_inventory.json", result)
    return result


def save(fig, folder, name):
    folder.mkdir(parents=True, exist_ok=True)
    fig.savefig(folder / (name + ".png"), dpi=220, bbox_inches="tight")
    fig.savefig(folder / (name + ".pdf"), bbox_inches="tight")
    plt.close(fig)


def heatmap(array, rows, cols, folder, name, title, cmap="viridis"):
    fig, ax = plt.subplots(figsize=(11, 6))
    im = ax.imshow(array, aspect="auto", cmap=cmap)
    ax.set_title(title)
    ax.set_xticks(range(len(cols)), cols, rotation=90, fontsize=6)
    ax.set_yticks(range(len(rows)), rows, fontsize=7)
    fig.colorbar(im, ax=ax, fraction=.03)
    save(fig, folder, name)


def run(output: Path, figures: Path, manifest_root: Path) -> dict:
    import scipy.sparse as sp
    path = output / "image_cluster64_patch_winners.u16.npy"
    labels64 = np.load(output / "feature_to_cluster64.npy")
    labels32 = np.load(output / "feature_to_cluster32.npy")
    counts = np.load(path, mmap_mode="r")
    acts = np.load(output / "image_activations.f16.npy", mmap_mode="r")
    tree = np.load(output / "hierarchy_linkage.npy")
    report = json.loads((output / "inference_report.json").read_text())
    clusters = json.loads((output / "cluster_report.json").read_text())
    batch_path = manifest_root / "batch.json"
    batch = json.loads(batch_path.read_text()) if batch_path.is_file() else None
    cities = tuple(batch["cities"]) if batch else CITIES
    expected_images = int(report["images"])
    if (len(cities) != 30 or counts.shape != (expected_images, 64) or acts.shape != (expected_images, 4096)
            or labels32.shape != (4096,) or labels64.shape != (4096,)
            or report["data_identity"] != clusters["dataset_identity"]):
        raise ValueError("inference, hierarchy or cohort mismatch")
    lengths = []
    for city in cities:
        frame = pd.read_parquet(manifest_root / (city.replace("/", "__") + ".parquet"))
        if batch:
            quality = pd.read_parquet(manifest_root.parent / "quality" / (city.replace("/", "__") + ".parquet"),
                                      columns=["panoid", "is_bad"])
            good = set(quality.loc[~quality.is_bad, "panoid"].astype(str))
            frame = frame[frame.panoid.astype(str).isin(good)]
        lengths.append(len(frame))
    if sum(lengths) != expected_images:
        raise ValueError("manifest and activation row counts disagree")
    city_names = [city.split("/")[-1] for city in cities]
    fine = np.zeros((30, 64), dtype=float)
    coarse = np.zeros((30, 32), dtype=float)
    occurrence_by_city = []
    cooc = np.zeros((64, 64), dtype=np.int64)
    cursor = 0
    min_support = max(100, int(np.ceil(.0005 * expected_images)))
    for i, n in enumerate(lengths):
        city_counts = np.asarray(counts[cursor:cursor+n], dtype=np.float32)
        fine[i] = city_counts.sum(0) / city_counts.sum()
        select = np.argpartition(-city_counts, 7, axis=1)[:, :8]
        present = city_counts[np.arange(n)[:, None], select] > 0
        binary = np.zeros((n, 64), dtype=np.uint8)
        binary[np.arange(n)[:, None], select] = present.astype(np.uint8)
        occurrence_by_city.append(binary)
        cooc += (sp.csr_matrix(binary.astype(np.int32)).T @ sp.csr_matrix(binary.astype(np.int32))).toarray()
        cursor += n
    # The fine hierarchy nests within the coarse hierarchy. Aggregate by parent.
    parents = np.array([np.bincount(labels32[labels64 == f], minlength=32).argmax() for f in range(64)])
    for i in range(30):
        np.add.at(coarse[i], parents, fine[i])
    support = np.diag(cooc).copy()
    expected = support[:, None] * support[None, :] / expected_images
    ppmi = np.maximum(np.log((cooc + .5) / (expected + .5)), 0)
    ppmi[cooc < min_support] = 0
    np.fill_diagonal(ppmi, 0)
    cosine = fine / np.linalg.norm(fine, axis=1, keepdims=True).clip(min=1e-12)
    cosine = cosine @ cosine.T
    js = np.array([[1 - jensenshannon(a, b, base=2) for b in fine] for a in fine])
    co_city = np.zeros((30, 64, 64), dtype=np.float32)
    for i, binary in enumerate(occurrence_by_city):
        co_city[i] = (sp.csr_matrix(binary.astype(np.int32)).T @ sp.csr_matrix(binary.astype(np.int32))).toarray()
    co_sim = np.corrcoef(co_city.reshape(30, -1))
    co_sim = np.nan_to_num(co_sim, nan=0.0)
    data = output / "results"; data.mkdir(exist_ok=True)
    pd.DataFrame(fine, index=city_names, columns=[f"F{i:03d}" for i in range(64)]).to_csv(data / "city_visual_composition_fine.csv")
    pd.DataFrame(coarse, index=city_names, columns=[f"C{i:03d}" for i in range(32)]).to_csv(data / "city_visual_composition_coarse.csv")
    for name, value in (("city_similarity_cosine", cosine), ("city_similarity_js", js),
                        ("city_cooccurrence_similarity", co_sim)):
        pd.DataFrame(value, index=city_names, columns=city_names).to_csv(data / (name + ".csv"))
    names = [f"F{i:03d}" for i in range(64)]
    pd.DataFrame(cooc, index=names, columns=names).to_csv(data / "global_element_cooccurrence.csv")
    pd.DataFrame(ppmi, index=names, columns=names).to_csv(data / "global_element_ppmi.csv")
    generated = []
    def matrix(a, rows, cols, name, title, cmap="viridis"):
        heatmap(a, rows, cols, figures, name, title, cmap); generated.append(name)
    matrix(fine, city_names, names, "Fig_City_Visual_Composition_Heatmap", "BatchTopK SAE visual composition by city")
    matrix(cosine, city_names, city_names, "Fig_City_Similarity_Heatmap", "City cosine similarity", "magma")
    matrix(js, city_names, city_names, "Fig_City_Similarity_JS_Heatmap", "City Jensen-Shannon similarity", "magma")
    matrix(cooc, names, names, "Fig_Global_Cooccurrence_Heatmap", "Global image co-occurrence", "magma")
    matrix(co_sim, city_names, city_names, "Fig_City_Cooccurrence_Similarity", "City co-occurrence similarity", "magma")
    fig, ax = plt.subplots(figsize=(9, 3.8))
    ax.bar(range(64), np.bincount(labels64, minlength=64)); ax.set(xlabel="Fine cluster", ylabel="SAE features", title="BatchTopK dictionary cluster sizes")
    save(fig, figures, "Fig_Cluster_Size_Distribution"); generated.append("Fig_Cluster_Size_Distribution")
    fig, ax = plt.subplots(figsize=(10, 5))
    dendrogram(tree, truncate_mode="lastp", p=64, no_labels=True, ax=ax)
    ax.set(title="BatchTopK SAE decoder hierarchy, 32/64 clusters", ylabel="Cosine distance")
    save(fig, figures, "Fig_BatchTopK_Hierarchy_32_64"); generated.append("Fig_BatchTopK_Hierarchy_32_64")
    coords = PCA(n_components=2, random_state=42).fit_transform(fine)
    fig, ax = plt.subplots(figsize=(8, 6)); ax.scatter(coords[:, 0], coords[:, 1])
    for (x, y), name in zip(coords, city_names): ax.annotate(name, (x, y), fontsize=6)
    ax.set(title="City visual composition embedding (PCA)", xlabel="PC1", ylabel="PC2")
    save(fig, figures, "Fig_City_Similarity_Embedding"); generated.append("Fig_City_Similarity_Embedding")
    # Feature-level image statistics, streamed to avoid a 3 GB dense float64 copy.
    feature_support = np.zeros(4096, np.int64)
    for start in range(0, expected_images, 512):
        feature_support += (np.asarray(acts[start:start+512]) > 0).sum(0)
    fig, ax = plt.subplots(figsize=(9, 4)); ax.hist(feature_support / expected_images, bins=60)
    ax.set(title="SAE feature image prevalence", xlabel="Fraction of images active", ylabel="Features")
    save(fig, figures, "Fig_BatchTopK_Image_Feature_Coexistence"); generated.append("Fig_BatchTopK_Image_Feature_Coexistence")
    fig, ax = plt.subplots(figsize=(8, 6))
    graph = nx.Graph()
    graph.add_nodes_from(range(64))
    for i in range(64):
        for j in np.argsort(-ppmi[i])[:10]:
            if i != j and ppmi[i, j] > 0:
                graph.add_edge(i, int(j), weight=float(ppmi[i, j]))
    positions = nx.spring_layout(graph, seed=42, weight="weight")
    nx.draw_networkx(graph, positions, ax=ax, with_labels=True,
                node_size=100, font_size=5, width=.35, edge_color="#999999")
    ax.set_title("BatchTopK visual element network (PPMI)")
    save(fig, figures, "Fig_Global_Visual_Element_Network"); generated.append("Fig_Global_Visual_Element_Network")
    result = inventory(output, generated)
    result["status"] = "partial_figure_reproduction"
    result["generated"] = generated
    write_json(output / "figure_inventory.json", result)
    return result


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    p.add_argument("--figures", type=Path, default=DEFAULT_OUTPUT / "figures")
    p.add_argument("--manifest-root", type=Path, default=DEFAULT_OUTPUT / "recovered_manifests")
    p.add_argument("--inventory-only", action="store_true")
    args = p.parse_args()
    result = inventory(args.output, []) if args.inventory_only else run(args.output, args.figures, args.manifest_root)
    print(json.dumps({k: v for k, v in result.items() if k != "figures"}, indent=2))


if __name__ == "__main__":
    main()
