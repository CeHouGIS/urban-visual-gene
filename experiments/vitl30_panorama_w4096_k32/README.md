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
- reports/vlm_groups/：64 组 VLM 记录。
- reports/vlm_dimensions/：此前 Top100 未遮挡裁剪解释的中间结果，已停止运行并保留。
- reports/vlm_dimensions_top10_masked/：当前逐维 VLM；Top10 高激活区域裁剪、背景 mask、原始回答、CSV 和交互核对页。
- reports/landscape_prototypes/：NMF 16/32/64 原型探索。
- reports/unit_motifs/：单维一致性、跨城市组合与空间关系。

## 数据身份与解释范围

实际训练身份以 outputs/experiment.json、cached_feature_map.json、cached_feature_audit.json 为准，快照在 results/training/。source_cohort/ 是早期候选清单与 QC 记录，城市有替换，不能代替实际缓存定义。

Top100 使用可核对原图的 **29 城市、3,712 验证全景、14,848 张方向图**。Moscow 原图映射未核实，未进入这轮原图解释。它不是全 NAS 搜索：282,004 个有效裁剪，3,178 维有证据，918 维在本样本无正响应。至少 20 例的 2,863 维进入逐维 UMAP；Top10 实例图有 29,934 点。

激活峰值连通区域裁剪 → SigLIP encoder。逐维图对各例先归一化、等权平均后再归一化，做 cosine UMAP；实例图逐例投影。二维距离供相似性浏览，不验证语义类别边界。

VLM 为本地 Qwen3-VL-2B-Instruct，名称是待复核的候选，已有明显误标实例。375 个跨城市组合是探索性候选，存在冗余，不能直接视为 375 个独立景观类型。

### 当前逐维 VLM：Top10 + mask

复用上述固定验证样本的已存排名及28×28激活图。每维按正激活峰值取前10个不同全景，各取最强方向（不足10例用实际例数），共29,934个实例。无需重提DINO/SigLIP特征或重训SAE。

1. 在每张方向图的已存8bit归一化激活图中保留值≥128的patch，近似该图峰值的50%；保存精度不是原始float激活精度。
2. 二值mask最近邻映射回原始图像，裁到所有保留区域的联合包围框。框内未保留像素统一置灰RGB(127,127,127)，空洞保留；不平滑、不膨胀，也不只取最大连通块。
3. 原始裁剪、二值mask及masked裁剪分别保存为PNG；每维最多10例按5列×2行、单格224像素排列。缩放后重新应用二值mask，避免插值带入遮挡区内容。
4. 仅把masked拼图和解释提示词输入Qwen，记录中文候选含义、类型、共同依据、其他解释、一致性、主观信心、局限及原始回答。少于5例要求低信心；无正激活的918维不编造解释。

输出中的index.html可逐维查看Qwen实际输入，并对照原裁剪→masked图→二值mask；records/保留回答和重试；masked_crops/保留像素mask、坐标和SHA256。Top100旧结果与新结果使用独立目录和运行身份，不能混用。此mask表达patch级归因，不等同于像素级物体分割。语义UMAP仍是原来的Top100干净裁剪embedding分析。

### 未解释维度的扩大检索

`panorama_vitl30_expand_interpretation.py` 只补做此前没有VLM解释的918维，保留已解释的3178维。复用景观原型实验的完整稀疏激活缓存，在29城市、86,038全景（344,152方向图）中检索每维Top10不同全景，使用最强方向的高激活原图裁剪和mask。排名按缓存float16正激活峰值计算；相同值依次按全景清单顺序、最早patch排序。

其中456维检索到正激活，共1747个实例；仅68维有满10例，其余按实际数量提供证据，少于5例的VLM解释要求低信心。462维在扩大后的范围仍无正激活，不编造含义。这个检索库包含SAE训练与验证图像，只用于事后解释，不能作为独立验证集；既有UMAP和统计分析的样本范围保持原定义。

输出为 `outputs/analysis/interpretation_expanded_top10/` 和 `reports/vlm_dimensions_top10_expanded/`。新报告中 `not_requested` 表示沿用原报告的解释，不能理解为未解释。`expansion_status.json`、`ranking_summary.json`、`runner_status.json` 分别记录总流程、检索计数和GPU任务状态。

Qwen仅在CUDA GPU上以bf16推理，不使用CPU推理或模型卸载。默认等待GPU 1、3中至少7000 MiB空闲显存，支持断点续跑，不终止其他任务：

```bash
python -u -m sae_experiments.panorama_vitl30_expand_interpretation --stage all --devices 1,3 --min-free-mib 7000 --workers 4
```

## 运行

从仓库根目录运行，旧模块名保持兼容：

```bash
python -m sae_experiments.panorama_vitl30 status
python -m sae_experiments.panorama_vitl30 train --cached-feature-map outputs/experiments/nas_30city_vitl_panorama_w4096_k32/cached_feature_map.json
python -m sae_experiments.panorama_vitl30_semantic_umap --help
python -m sae_experiments.panorama_vitl30_instance_umap --help
python -u -m sae_experiments.panorama_vitl30_dimension_vlm_runner --devices 1,3 --batch-size 1 --input-mode masked-top10 --mask-fraction 0.5
```

规范模块位置为 experiments.vitl30_panorama_w4096_k32.code.<模块名>。旧 outputs/experiments/nas_30city_vitl_panorama_w4096_k32 与 reports/vitl30_* 是本目录相对链接，兼容清单与后台进程。NAS 特征保持原路径；训练 / 推理需已有缓存与权重。
