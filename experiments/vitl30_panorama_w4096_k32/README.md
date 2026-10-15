# 当前主实验：30 城市 ViT-L panorama-aware BatchTopK SAE

## 实际设置

| 项目 | 设置 |
| --- | --- |
| 数据 | 复用 ViT-L 缓存，30 城市、89,031 个四方向全景 |
| 划分 | 85,191 训练；3,840 验证，每城固定 128 全景 |
| Backbone | 冻结 DINOv3 ViT-L/16；448×448 → 每方向 28×28×1024 |
| 全景构建 | 四方向网格环形组织后做邻域 context mixing，再恢复逐 patch 输入 |
| SAE | 输入 D=1024；字典 W=4096；BatchTopK K=32；重建 D=1024 |
| 优化 | AdamW，lr=3e-4，squared-L2 reconstruction，seed=42 |
| 训练 | 16 轮，每方向每轮 49 个位置，16 轮覆盖全部 784 个位置；最优第 10 轮 |
| 验证 | best sampled validation squared-L2 = 0.2873438360 |
| 64 组 | decoder cosine distance + average linkage |

K=32 控制批次的总激活预算，并非每个 patch 恰好 32 个非零。各方向单独经 DINOv3，全景邻接处理在 SAE 前完成；没有直接将四个 1024 向量拼成 SAE 输入。

## 文件

- code/：训练、缓存映射、激活推理、论文图、VLM、UMAP、景观组合。
- tests/：本版本输入映射、激活区域、语义与统计检验。
- [results/](results/README.md)：GitHub 发布的训练记录、图、摘要和表格；manifest.json 记录来源与 SHA256。
- outputs/：本机模型、历史、缓存映射、推理和 Top100 裁剪。
- reports/semantic_umap/：一维一个点，Top100 区域 embedding 汇总，64 组唯一颜色。
- reports/instance_umap/：每维 Top10 图像实例分别成点。
- reports/vlm_groups/、reports/vlm_dimensions/：64 组及逐维 VLM 记录。
- reports/landscape_prototypes/：NMF 16/32/64 原型探索。
- reports/unit_motifs/：单维一致性、跨城市组合与空间关系。

## 数据身份与解释范围

实际训练身份以 outputs/experiment.json、cached_feature_map.json、cached_feature_audit.json 为准，快照在 results/training/。source_cohort/ 是早期候选清单与 QC 记录，城市有替换，不能代替实际缓存定义。

Top100 使用可核对原图的 **29 城市、3,712 验证全景、14,848 张方向图**。Moscow 原图映射未核实，未进入这轮原图解释。它不是全 NAS 搜索：282,004 个有效裁剪，3,178 维有证据，918 维在本样本无正响应。至少 20 例的 2,863 维进入逐维 UMAP；Top10 实例图有 29,934 点。

激活峰值连通区域裁剪 → SigLIP encoder。逐维图对各例先归一化、等权平均后再归一化，做 cosine UMAP；实例图逐例投影。二维距离供相似性浏览，不验证语义类别边界。

VLM 为本地 Qwen3-VL-2B-Instruct，名称是待复核的候选，已有明显误标实例。375 个跨城市组合是探索性候选，存在冗余，不能直接视为 375 个独立景观类型。

## 运行

从仓库根目录运行，旧模块名保持兼容：

```bash
python -m sae_experiments.panorama_vitl30 status
python -m sae_experiments.panorama_vitl30 train --cached-feature-map outputs/experiments/nas_30city_vitl_panorama_w4096_k32/cached_feature_map.json
python -m sae_experiments.panorama_vitl30_semantic_umap --help
python -m sae_experiments.panorama_vitl30_instance_umap --help
python -u -m sae_experiments.panorama_vitl30_dimension_vlm_runner --devices 1,3 --batch-size 1
```

规范模块位置为 experiments.vitl30_panorama_w4096_k32.code.<模块名>。旧 outputs/experiments/nas_30city_vitl_panorama_w4096_k32 与 reports/vitl30_* 是本目录相对链接，兼容清单与后台进程。NAS 特征保持原路径；训练 / 推理需已有缓存与权重。
