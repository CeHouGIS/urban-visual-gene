"""Report raw VLM responses separately from automated visual evidence review."""
from __future__ import annotations

import argparse
import json
import shutil
import textwrap
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from PIL import Image

from sae_experiments.panorama_vitl30 import OUTPUT
from sae_experiments.panorama_vitl30_semantics import semantic_root

REPORT = Path("/root/code/urban-visual-gene/reports/vitl30_vlm_interpretation")


def tex_escape(value):
    mapping = {"\\": r"\textbackslash{}", "&": r"\&", "%": r"\%", "$": r"\$", "#": r"\#",
               "_": r"\_", "{": r"\{", "}": r"\}", "~": r"\textasciitilde{}", "^": r"\textasciicircum{}"}
    return "".join(mapping.get(char, char) for char in str(value))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=OUTPUT)
    parser.add_argument("--report", type=Path, default=REPORT)
    parser.add_argument("--thesis", type=Path, default=Path("/root/code/HKUST-Thesis"))
    args = parser.parse_args()
    root = semantic_root(args.output)
    report = json.loads((root / "report.json").read_text())
    if report["status"] != "complete_provisional":
        raise ValueError("Finish failed VLM responses before publishing the report")
    records = json.loads((root / "group_interpretations.json").read_text())
    rows = pd.read_csv(root / "group_interpretations.csv").fillna("")
    args.report.mkdir(parents=True, exist_ok=True)
    for name in ["report.json", "group_interpretations.csv", "group_interpretations.json", "identity.json", "vlm_runtime.json", "retrieval_report.json", "visual_review.json"]:
        shutil.copy2(root / name, args.report / name)
    for suffix in ["png", "pdf"]:
        activation = args.output / "figures" / f"Fig_ViTL30_Activation_Examples.{suffix}"
        shutil.copy2(activation, args.report / activation.name)
    if report["reviewed_groups"] != 64:
        raise ValueError("Finish the separate visual review before creating thesis annotations")
    counts = rows.review_evidence_pattern.value_counts().to_dict()
    retrieval = report["retrieval"]
    body = ["# 30-city panorama-aware BatchTopK SAE: VLM interpretation", "",
            "本报告使用已完成的 ViT-L/16 panorama-aware SAE（输入 1,024 维，字典 4,096 维，BatchTopK K=32，epoch-10 checkpoint）。复用了特征缓存与模型，没有重新提取 DINO 特征或重新训练。", "",
            f"64 个 decoder group 全部完成 Qwen3-VL 结构化初稿，并通过 Codex 图像工具核对了全部 379 张局部裁剪。原始 Qwen 标签、confidence 和回答独立保留，复核描述在 reviewed_* 字段中；两者不能混为同一个模型的输出。", "",
            f"在这批检索实例中，{len(report['review_recurring_groups'])} 组显示重复出现的物体家族或外观模式，{len(report['review_predominant_groups'])} 组有主要类别但包含异类实例，{len(report['review_mixed_groups'])} 组保留 mixed 描述。这是对小规模证据的定性检查，不能当作全量语义纯度或准确率。", "",
            f"只有一个实例的组为 {report['single_example_groups']}；其描述只针对单张裁剪，不构成组内语义一致性的证据。", "",
            f"证据检索覆盖 {len(retrieval['eligible_cities'])} 个映射已核对的城市、{retrieval['candidate_panoramas']:,} 个候选全景点位，选出 {retrieval['examples']} 个组别实例。候选包含各城市、各组 dominant-patch share 最高的 2 个点位、均匀分布的 16 个点位，以及之前的实例；这不是全量图像检索。Moscow 的图像映射未核对，因此不用于视觉解释。", "",
            "针对每个候选点位，将四方向的 3,136 个 contextualized patch 一起送入冻结 SAE，按组内所有维度的最大正稀疏激活检索实例，保留非 dominant 响应。每组优先选择不同城市和不同维度，最终每个点位只贡献一个实例。", "",
            "Qwen3-VL-2B-Instruct 接收干净的 96×96 局部裁剪（在 448×448 方向图上裁剪，再放大到 192×192 输入）。原始四方向全景、192×192 上下文裁剪和激活叠图另行保存供检查。颜色叠图、位置框、城市名称与地理标签均不输入 VLM。", "",
            "每个回答包含 label、semantic_type、recurring_cues、ambiguities 和 confidence。程序检查 JSON、字段取值、列表长度和生成是否截断；不合格回答自动重试并保留历次原始输出。旧的 35 份草稿单独保留。", "",
            "复核发现局部模型有时会用单个实例来命名整组：例如 G00 的 water cooler 只对应其中一张裁剪，G06 的 trampoline 也只出现一次，G12 的 bench 忽略了另外五张植被裁剪。复核据此保留异质组，避免将一个物体外推成整组的含义。", "",
            "较明确的实例包括 G01 的交通工具家族、G30 的道路标线和边界、G35 的天空与云层。G35 的六张裁剪只覆盖两个成员维度；即使实例看起来一致，也不能证明全组的所有维度语义相同。", "",
            "这些结果只描述检索到的实例，不能证明一个组内的全部 SAE 维度都有相同语义。VLM 的 confidence 是模型自述，不能当作准确率；Codex 复核同样是自动化图像判断，尚未进行独立人工标注验证。decoder 分组、城市统计与共现分析均不使用这些描述。", "",
            "激活图已使用灰度底图和 inferno 色谱（紫→橙→黄），青色框标出检索 patch；每张全景按自己的最大响应归一化，并保留原始 peak 数值。颜色适合检查激活位置，跨全景比较强度要看原始数值。", "",
            "- 完整结果表：`group_interpretations.csv`", "- 原始回答、提示词与图像路径：`group_interpretations.json`", f"- 本地交互证据页面：`{root / 'index.html'}`", "- 语义证据插图：`Fig_ViTL30_VLM_Interpretation.png` / `.pdf`", "- 激活对照插图：`Fig_ViTL30_Activation_Examples.png` / `.pdf`", "",
            "## 组别解释", "", "| Group | 维度数 | Qwen 原始标签 | 图像复核描述 | 实例模式 | 实例/城市 |", "|---|---:|---|---|---|---:|"]
    for _, row in rows.iterrows():
        label = str(row.label).replace("|", "/")
        reviewed = str(row.reviewed_label).replace("|", "/")
        body.append(f"| G{int(row.group):02d} | {int(row.dictionary_dimensions)} | {label} | {reviewed} | {row.review_evidence_pattern} | {int(row.examples)}/{int(row.cities)} |")
    (args.report / "REPORT.md").write_text("\n".join(body) + "\n")
    # Include recurring, heterogeneous, and single-example evidence explicitly.
    # This fixed illustrative selection is not a semantic accuracy benchmark.
    by_group = {r["group"]: r for r in records}
    selected = [by_group[g] for g in [1, 3, 12, 30, 34, 35, 0, 51]]
    fig, axes = plt.subplots(len(selected), 6, figsize=(12, 1.68 * len(selected)), squeeze=False)
    for y, record in enumerate(selected):
        obj = record["visual_review"]
        for x, ax in enumerate(axes[y]):
            ax.set_xticks([]); ax.set_yticks([])
            for spine in ax.spines.values(): spine.set_visible(False)
            examples = record["evidence"]
            if x < len(examples):
                with Image.open(examples[x]["focused_crop"]) as image:
                    ax.imshow(image.convert("RGB"))
                ax.set_xlabel(examples[x]["city"].split("/")[-1], fontsize=7)
            else:
                ax.set_visible(False)
        label = f"G{record['group']:02d}\n" + "\n".join(textwrap.wrap(obj["label"], 21))
        axes[y, 0].set_ylabel(label, fontsize=8, rotation=0, ha="right", va="center", labelpad=9)
    fig.subplots_adjust(left=.16, right=.995, top=.965, bottom=.025, wspace=.06, hspace=.27)
    fig.suptitle("Retrieved crops and conservative visual-review descriptions", fontsize=12)
    png = args.report / "Fig_ViTL30_VLM_Interpretation.png"
    pdf = args.report / "Fig_ViTL30_VLM_Interpretation.pdf"
    fig.savefig(png, dpi=180, bbox_inches="tight")
    fig.savefig(pdf, bbox_inches="tight")
    plt.close(fig)
    thesis_assets = args.thesis / "img_gene/vitl30_w4096"
    for path in [png, pdf]:
        shutil.copy2(path, thesis_assets / path.name)
    tex = [r"\begin{figure}[htbp]", r"    \centering", r"    \includegraphics[width=\textwidth]{img_gene/vitl30_w4096/Fig_ViTL30_VLM_Interpretation.pdf}",
           r"    \caption{Retrieved local crops and conservative descriptions from automated visual review of the Qwen3-VL drafts. Each row shows up to six distinct panorama locations from one decoder group. The selected rows illustrate recurring appearances, outliers, heterogeneous evidence, and the single-example limitation; they are not an accuracy evaluation. The headings describe the displayed evidence rather than every member dimension. Raw VLM answers are retained separately, and independent human validation has not been conducted.}",
           r"    \label{fig:vitl30_vlm_interpretation}", r"\end{figure}", "",
           r"\begingroup", r"\footnotesize", r"\setlength{\tabcolsep}{4pt}",
           r"\begin{longtable}{p{0.95cm}r p{6.0cm}cc}",
           r"\caption{Conservative evidence descriptions for all 64 decoder groups, following automated visual review of raw VLM drafts. P/C gives retrieved panoramas and distinct cities. R denotes a recurring object family or appearance in the displayed crops, O denotes a predominant category with outliers, M denotes heterogeneous evidence, and S denotes a single example. These qualitative evidence codes do not measure semantic accuracy or whole-group purity. Group identifiers are zero-based.}\label{tab:vitl30_vlm_groups}\\", r"\toprule",
           r"Group & Dim. & Evidence description & P/C & Pattern \\ \midrule", r"\endfirsthead",
           r"\multicolumn{5}{l}{\tablename\ \thetable\ (continued)}\\ \toprule",
           r"Group & Dim. & Evidence description & P/C & Pattern \\ \midrule", r"\endhead",
           r"\bottomrule", r"\endfoot"]
    for _, row in rows.iterrows():
        code = {"recurring": "R", "predominant_with_outliers": "O", "mixed": "M", "single_example": "S"}[row.review_evidence_pattern]
        tex.append(f"G{int(row.group):02d} & {int(row.dictionary_dimensions)} & {tex_escape(row.reviewed_label)} & {int(row.examples)}/{int(row.cities)} & {code} " + r"\\")
    tex.extend([r"\end{longtable}", r"\endgroup", ""])
    (args.thesis / "chapter5_vitl30_semantics.tex").write_text("\n".join(tex))
    print(json.dumps({"report": str(args.report), "semantic_type_counts": counts,
                      "figure_groups": [x["group"] for x in selected]}, indent=2))


if __name__ == "__main__":
    main()
