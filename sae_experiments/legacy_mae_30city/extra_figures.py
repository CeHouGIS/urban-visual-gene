"""Additional SAE analogues of archived Feature-MAE analytical figures."""
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
from scipy.cluster.hierarchy import dendrogram, linkage
from scipy.spatial.distance import pdist

from .figures import save
from .nas_batch import OUTPUT
from .train import write_json


def run(output: Path, figures: Path, seed: int = 42) -> dict:
    rng = np.random.default_rng(seed)
    data = output / "results"
    figures.mkdir(parents=True, exist_ok=True)
    composition = pd.read_csv(data / "city_visual_composition_fine.csv", index_col=0)
    co_city = pd.read_csv(data / "city_cooccurrence_similarity.csv", index_col=0)
    co = pd.read_csv(data / "global_element_cooccurrence.csv", index_col=0).to_numpy()
    ppmi = pd.read_csv(data / "global_element_ppmi.csv", index_col=0).to_numpy()
    counts = np.load(output / "image_cluster64_patch_winners.u16.npy", mmap_mode="r")
    labels64 = np.load(output / "feature_to_cluster64.npy")
    labels32 = np.load(output / "feature_to_cluster32.npy")
    n = len(counts)
    generated = []

    tree = linkage(pdist(composition.to_numpy(), metric="cosine"), method="average")
    fig, ax = plt.subplots(figsize=(9, 8))
    dendrogram(tree, labels=composition.index.tolist(), orientation="right", leaf_font_size=7, ax=ax)
    ax.set(xlabel="Cosine distance", title="City hierarchy from SAE visual composition")
    save(fig, figures, "Fig_City_Hierarchical_Clustering")
    generated.append("Fig_City_Hierarchical_Clustering")

    fig, axes = plt.subplots(1, 2, figsize=(11, 4.2), constrained_layout=True)
    composition_similarity = np.corrcoef(composition.to_numpy())
    tri = np.triu_indices(len(composition), 1)
    axes[0].scatter(composition_similarity[tri], co_city.to_numpy()[tri], s=15, alpha=.6)
    axes[0].set(xlabel="Composition correlation", ylabel="Co-occurrence correlation",
                title="City-pair similarity")
    top = composition.to_numpy().max(axis=1)
    mean_relational_similarity = (co_city.to_numpy().sum(axis=1) - 1) / (len(co_city) - 1)
    axes[1].scatter(top, mean_relational_similarity, s=20)
    axes[1].set(xlabel="Largest category share", ylabel="Mean relational similarity to other cities",
                title="Composition concentration")
    save(fig, figures, "Fig_Composition_vs_Cooccurrence")
    generated.append("Fig_Composition_vs_Cooccurrence")

    thresholds = (1, 2, 4, 8, 16)
    sample_indices = rng.choice(n, min(n, 20_000), replace=False)
    selected = np.asarray(counts[sample_indices], dtype=np.uint16)
    means = [(selected >= threshold).sum(axis=1).mean() for threshold in thresholds]
    pd.DataFrame({"patch_winner_threshold": thresholds, "mean_clusters_per_image": means}).to_csv(
        data / "cluster_threshold_sensitivity.csv", index=False)
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.plot(thresholds, means, marker="o", color="#156f71")
    ax.set(xlabel="Minimum winning patches per image", ylabel="Mean present clusters",
           title="SAE cluster threshold sensitivity")
    save(fig, figures, "Fig_Image_Dimension_Threshold_Sensitivity")
    generated.append("Fig_Image_Dimension_Threshold_Sensitivity")

    fig, ax = plt.subplots(figsize=(8, 7))
    im = ax.imshow(ppmi, cmap="magma", interpolation="nearest")
    ax.set(xlabel="SAE cluster", ylabel="SAE cluster",
           title="Positive PMI of 64 SAE clusters (top-eight presence)")
    fig.colorbar(im, ax=ax, fraction=.046)
    save(fig, figures, "Fig_BatchTopK_Image_Feature_PPMI_64")
    generated.append("Fig_BatchTopK_Image_Feature_PPMI_64")

    support = np.diag(co)
    pairs = []
    for i in range(64):
        for j in range(i + 1, 64):
            if co[i, j] >= max(100, int(.0005 * n)):
                pairs.append((i, j, int(co[i, j]), float(ppmi[i, j])))
    pair_table = pd.DataFrame(pairs, columns=["cluster_a", "cluster_b", "joint_images", "ppmi"])
    pair_table = pair_table.sort_values(["ppmi", "joint_images"], ascending=False)
    pair_table.to_csv(data / "cluster_pair_ppmi.csv", index=False)
    strongest = pair_table.head(20)
    fig, ax = plt.subplots(figsize=(9, 6))
    ax.barh(np.arange(len(strongest)), strongest.ppmi.iloc[::-1], color="#228b75")
    ax.set_yticks(np.arange(len(strongest)),
                  [f"F{a:02d}+F{b:02d}" for a, b in zip(strongest.cluster_a.iloc[::-1],
                                                         strongest.cluster_b.iloc[::-1])], fontsize=7)
    ax.set(xlabel="Positive PMI", title="Most unexpectedly frequent SAE cluster pairs")
    save(fig, figures, "Fig_Unexpected_Visual_Cooccurrence_Global")
    generated.append("Fig_Unexpected_Visual_Cooccurrence_Global")

    common = pair_table.sort_values("joint_images", ascending=False).head(20)
    fig, ax = plt.subplots(figsize=(9, 6))
    ax.barh(np.arange(len(common)), common.joint_images.iloc[::-1], color="#bf6b40")
    ax.set_yticks(np.arange(len(common)),
                  [f"F{a:02d}+F{b:02d}" for a, b in zip(common.cluster_a.iloc[::-1],
                                                         common.cluster_b.iloc[::-1])], fontsize=7)
    ax.set(xlabel="Co-occurring images", title="Most prevalent SAE cluster pairs")
    save(fig, figures, "Fig_Prevalent_Dimension_Combinations")
    generated.append("Fig_Prevalent_Dimension_Combinations")

    checkpoint = __import__("torch").load(output / "model_best.pt", map_location="cpu",
                                          weights_only=False)
    decoder = checkpoint["state"]["decoder.weight"].float().numpy().T
    decoder /= np.linalg.norm(decoder, axis=1, keepdims=True).clip(min=1e-8)
    from sklearn.decomposition import PCA
    coords = PCA(n_components=2, random_state=seed).fit_transform(decoder)
    fig, ax = plt.subplots(figsize=(8, 7))
    dots = ax.scatter(coords[:, 0], coords[:, 1], c=labels64, cmap="turbo",
                      s=4, alpha=.6, linewidths=0)
    fig.colorbar(dots, ax=ax, label="64-cluster assignment", fraction=.046)
    ax.set(xlabel="Decoder PC1", ylabel="Decoder PC2",
           title="All 4,096 SAE dictionary directions (PCA)")
    save(fig, figures, "Fig_BatchTopK_Feature_Embedding_4096")
    generated.append("Fig_BatchTopK_Feature_Embedding_4096")

    graph = nx.Graph()
    graph.add_nodes_from(range(64))
    for a in range(64):
        for b in np.argsort(-ppmi[a])[:3]:
            if a != b and ppmi[a, b] > 0 and co[a, b] >= 100:
                graph.add_edge(a, int(b), weight=float(ppmi[a, b]))
    coarse_labels = labels32[np.array([np.flatnonzero(labels64 == i)[0] for i in range(64)])]
    ordered_nodes = np.lexsort((np.arange(64), coarse_labels))
    angles = np.linspace(0, 2 * np.pi, 64, endpoint=False)
    positions = {int(node): np.array([np.cos(angle), np.sin(angle)])
                 for node, angle in zip(ordered_nodes, angles)}
    fig, ax = plt.subplots(figsize=(12, 12))
    edge_weights = np.array([graph[u][v]["weight"] for u, v in graph.edges])
    nx.draw_networkx_edges(graph, positions, ax=ax, edge_color="#677d80",
                           width=.3 + edge_weights / max(edge_weights.max(), 1) * 1.5, alpha=.23)
    node_sizes = 70 + 260 * np.sqrt(support / max(support.max(), 1))
    nx.draw_networkx_nodes(graph, positions, ax=ax, node_size=node_sizes,
                           node_color=coarse_labels,
                           cmap="tab20", edgecolors="white", linewidths=.5)
    label_positions = {node: point * 1.09 for node, point in positions.items()}
    nx.draw_networkx_labels(graph, label_positions, ax=ax, font_size=7)
    ax.set_title("SAE cluster network: size = image support, edge = positive PMI")
    ax.set_xlim(-1.18, 1.18)
    ax.set_ylim(-1.18, 1.18)
    ax.axis("off")
    save(fig, figures, "Fig_Global_Visual_Element_Network_Detailed")
    generated.append("Fig_Global_Visual_Element_Network_Detailed")

    history = pd.read_csv(output / "history.csv")
    fig, ax = plt.subplots(figsize=(10, 4.5))
    steps = np.arange(1, len(history) + 1)
    ax.plot(steps, history.train_squared_l2.rolling(200, min_periods=1).mean(),
            color="#247d76", linewidth=1.3, label="Training (200-step mean)")
    validated = history.val_squared_l2.notna()
    ax.plot(steps[validated], history.loc[validated, "val_squared_l2"],
            marker="o", color="#bd6948", label="Validation")
    ax.set(xlabel="Optimizer step", ylabel="Squared L2 reconstruction loss",
           title="BatchTopK W4096/K32 training and validation loss")
    ax.legend(frameon=False)
    save(fig, figures, "Fig_BatchTopK_Training_Loss")
    generated.append("Fig_BatchTopK_Training_Loss")

    # The old resolution plot used MAE-specific metrics. Here show the
    # distribution of dictionary members per cluster at both hierarchy cuts.
    fig, axes = plt.subplots(1, 2, figsize=(10, 4), constrained_layout=True)
    for ax, labels, title in zip(axes, (labels32, labels64), ("32 clusters", "64 clusters")):
        sizes = np.bincount(labels)
        ax.hist(sizes, bins=16, color="#4d7584", edgecolor="white")
        ax.set(xlabel="Dictionary features per cluster", ylabel="Clusters", title=title)
    save(fig, figures, "Fig_Cluster_Resolution_Metrics")
    generated.append("Fig_Cluster_Resolution_Metrics")

    payload = json.loads((output / "figure_inventory.json").read_text())
    rename = {"Fig_Image_Dimension_PPMI_512": "Fig_BatchTopK_Image_Feature_PPMI_64"}
    notes = {
        "Fig_Cluster_Resolution_Metrics": "Shows 32/64-cluster dictionary membership, not old MAE resolution/stability metrics.",
        "Fig_Prevalent_Dimension_Combinations": "Shows frequent SAE cluster pairs only; old triples and conditional spatial shares are not reproduced.",
        "Fig_Unexpected_Visual_Cooccurrence_Global": "Uses SAE cluster PPMI; old E+D+P visual-distance weighting is not reproduced.",
        "Fig_Image_Dimension_PPMI_512": "PPMI uses 64 SAE clusters rather than 512 MAE dimensions.",
        "Fig_Global_Visual_Element_Network_Detailed": "Detailed network uses SAE-cluster PPMI and decoder-cluster colors.",
    }
    for row in payload["figures"]:
        target = rename.get(row["legacy_figure"], row["legacy_figure"])
        if target in generated:
            row.update(new_figure=target, status="recomputed_from_SAE",
                       reason=notes.get(row["legacy_figure"]))
    payload["generated"] = sorted(set(payload.get("generated", [])) | set(generated))
    payload["recomputed_count"] = sum(row["status"] == "recomputed_from_SAE" for row in payload["figures"])
    write_json(output / "figure_inventory.json", payload)
    return {"generated": generated, "recomputed": payload["recomputed_count"],
            "legacy": payload["original_figure_count"]}


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--output", type=Path, default=OUTPUT)
    p.add_argument("--figures", type=Path)
    args = p.parse_args()
    print(json.dumps(run(args.output, args.figures or args.output / "figures"), indent=2))


if __name__ == "__main__":
    main()
