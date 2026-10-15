# 项目目录清单与 ViT-B 清理记录

进一步扫描见 [清理后剩余文件详解](REMAINING_FILES.md)，逐文件清单见 [代码和关键元数据 CSV](remaining_code_and_metadata.csv)，目录汇总见 [remaining_directories.csv](remaining_directories.csv)。后续又清理了 16 个 ViT-B 遗留日志；当前 ViT-L 数据和模型保持原样。

更新时间：2026-10-15T13:28:11.892967+00:00。扫描范围为本地工程；未遍历 NAS，未跟随原图符号链接。

## 当前主实验

当前主实验为 **30 城市 DINOv3 ViT-L/16 + panorama context + BatchTopK SAE（W4096/K32）**。实际复用缓存包含 89,031 个全景，其中训练 85,191 个、验证 3,840 个。训练已完成 16 轮。原始候选采样清单与最终缓存的城市存在替换；实际训练身份以主实验的 experiment.json、cached_feature_map.json 和 cached_feature_audit.json 为准。

每个方向的 448×448 图像对应 28×28×1024 patch 表征，四向网格通过环形邻接融合上下文，再以 1024 维 patch 向量输入 SAE。字典是 4096 维，重建仍是 1024 维。

## 顶层目录

| 目录 | 本地占用 | 用途 |
| --- | ---: | --- |
| `outputs/` | 5.49 GiB | 当前 ViT-L 检查点、映射、激活、分析证据与论文图件；大规模特征本体仍在 NAS。 |
| `reports/` | 11.31 GiB | 文字报告、交互网页、Top 激活区域裁剪、语义 UMAP、景观组合和 VLM 记录。 |
| `archive/` | 1.22 GiB | 已归档的 Feature-MAE、图视觉词汇、道路 MRLU、旧网站及历史结果。 |
| `.git/` | 4.63 GiB | Git 历史与对象库；本次没有改写提交历史。 |
| `formal/` | 2.96 MiB | 较早的 307 城市 ViT-L 训练、消融、字典尺度分析与共用质量筛选。 |
| `sae_experiments/` | 1.11 MiB | 当前 ViT-L 实验入口、共享模型和分析代码。 |
| `tests/` | 0.32 MiB | 数据映射、全景 context、训练接口与分析校验。 |
| `configs/` | 0.11 MiB | 较早的 300 Google + 7 百度城市列表、采样与训练配置。 |

## 当前代码分别做什么

| 文件或目录 | 用途 |
| --- | --- |
| sae_experiments/panorama_vitl30.py | ViT-L 特征构建、缓存复用与 BatchTopK 训练 |
| sae_experiments/panorama_vitl30_cache.py | 已有特征与全景、原图、元数据之间的映射 |
| sae_experiments/panorama_vitl30_analysis.py | SAE 激活推理、64 组划分及统计 |
| sae_experiments/panorama_vitl30_figures.py、panorama_vitl30_heatmaps.py | 论文结果图及激活叠图 |
| sae_experiments/panorama_vitl30_semantics.py | 组级 VLM 语义解释 |
| sae_experiments/panorama_vitl30_semantic_*.py | 每维 Top100 激活区域、SigLIP 语义向量与逐维 UMAP |
| sae_experiments/panorama_vitl30_instance_umap.py | 每维 Top10 图像实例的 UMAP |
| sae_experiments/panorama_vitl30_landscape_prototypes.py | 探索性的景观组合分解 |
| sae_experiments/panorama_vitl30_unit_motifs.py | 单维一致性、跨城市组合复现与峰值空间关系 |
| sae_experiments/panorama_vitl30_dimension_vlm.py | 对每维已有 Top100 干净裁剪记录 VLM 候选含义 |
| sae_experiments/panorama_vitl30_dimension_vlm_runner.py | VLM 后台执行、显存等待、OOM 重试和断点恢复 |
| sae_experiments/models/ | BatchTopKSAE 和 panorama context mixer |
| sae_experiments/io_utils.py | 公共哈希、原子保存和 metadata 查找，已脱离旧 ViT-B 包 |
| formal/quality/ | 全黑与隧道样本筛选及筛选清单映射 |
| formal/analysis/ | 较早的消融、字典尺度、交互 Sankey、语义图和共现分析 |

## 当前 ViT-L 结果放在哪里

主实验：`outputs/experiments/nas_30city_vitl_panorama_w4096_k32/`。

- `model_best.pt`、`checkpoint_last.pt`：当前模型和恢复检查点。
- `history.csv`、`training_report.json`：训练日志与完成状态。
- `cached_feature_map.json`、`cache_manifests/`：实际缓存及图像映射。
- `analysis/semantic_umap_top100/`：282,004 个 Top100 裁剪、SigLIP embedding、逐维排名和原图链接；图像证据覆盖 29 城市的 3,712 个验证全景，Moscow 因原图映射未核实而排除。
- `figures/`：当前 ViT-L 论文图件。
- `source_cohort/`：删除 ViT-B 前迁出的共用候选清单和质量筛选记录，共 91 个文件，逐文件核对 SHA-256；这些数据溯源不是 ViT-B 模型结果。

| 报告目录 | 内容 |
| --- | --- |
| reports/vitl30_semantic_umap/ | 一个维度一个点的图像语义 UMAP，64 组配色 |
| reports/vitl30_instance_umap/ | 每维 Top10 图像实例的语义 UMAP |
| reports/vitl30_vlm_interpretation/ | 64 组 VLM 候选解释 |
| reports/vitl30_dimension_vlm/ | 逐维 Top100 的 VLM 记录、原始回答、输入拼图和后台状态 |
| reports/vitl30_landscape_prototypes/ | 探索性的景观组合分解 |
| reports/vitl30_unit_motifs/ | 单维一致性审计和统计组合验证 |

VLM 结果为候选语义，需要结合保存的输入图核对。JSON 格式通过不代表语义正确，模型自述 confidence 也不代表准确率。

## 还保留哪些较早的实验

- `formal/`、`reports/scaling/`：307 城市 ViT-L 及字典尺度、稳定/细分/未匹配的交互展示。
- `reports/sae_4096/`：较早 W4096 的 K16/32/64 对照图库。
- `reports/data/semantic_umap_w2048_k64_top100/` 与 `semantic_umap_w4096_k64_top100/`：较早 SAE 的语义裁剪与网页数据，合计约 10 GiB，是 reports 占用的主要来源。
- `archive/non_batchtopk/`：Feature-MAE 等历史线；约 14 万个路径由 Git 跟踪，主要是旧网站与图像资产。
- `archive/road_mrlu/`：旧道路流水线。

这些较早的 ViT-L 实验及非 BatchTopK 归档没有被当作 ViT-B 实验删除。

## 已删除的 ViT-B 实验

本项目释放约 **106.97 GiB**，删除清单如下：

- `outputs/experiments/nas_30city_batchtopk_w4096_k32`
- `outputs/experiments/feature_mae_30city_batchtopk_w4096_k32`
- `sae_experiments/legacy_mae_30city`
- `reports/nas_30city_batchtopk_w4096_k32.md`
- `reports/nas_30city_batchtopk_w4096_k32_experiment_report.md`

其中包括旧 checkpoint、ViT-B token 缓存、激活、统计表、全部结果图和运行脚本。论文仓库中未被 LaTeX 引用的 `img_gene/nas30_w4096/`、两张 `Fig_PatchSAE_Examples_*.png` 及其 provenance CSV 也已删除；论文正文未改动。

当前 ViT-L 检查点、缓存映射及 Top100 裁剪清单在清理前后哈希一致，清理后的公共入口可导入。根目录 README 已更新，formal/README 已标明历史实验身份。`.gitignore` 的 `models/` 改为 `/models/`，以免误忽略 sae_experiments/models/ 内的当前模型代码。

## 清单文件

- [清理后体积与状态](current_summary.json)
- [ViT-B 删除记录](vitb_cleanup.json)
- [ViT-L 关键文件哈希](protected_vitl_integrity.json)
- [清理前完整目录快照](inventory_before_cleanup.json)
