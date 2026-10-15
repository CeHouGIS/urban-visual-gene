# 清理后还剩哪些文件

扫描日期：2026-10-15。范围是本地项目，不遍历 NAS，不沿原图符号链接展开。VLM 后台仍在写文件，因此体积是扫描时的近似值。

## 1. 当前主实验：保留

**30 城市、DINOv3 ViT-L/16、448px、28×28 patch、panorama context、BatchTopK W4096/K32。** 实际缓存是 89,031 个全景；训练已经完成。输入 SAE 的单个向量为 1024 维，4096 是字典宽度。

| 位置 | 里面是什么 | 本地占用 |
| --- | --- | ---: |
| `sae_experiments/panorama_vitl30*.py` | 特征映射、训练、推理、绘图、语义解释、原型和组合分析 | 代码文件 |
| `sae_experiments/models/` | BatchTopK SAE 和四方向的 panorama context mixer | 代码文件 |
| `outputs/experiments/nas_30city_vitl_panorama_w4096_k32/` | 模型、训练记录、实际数据映射、激活统计、论文图件、Top100 裁剪与 embedding | 约 5.5 GiB |

主实验目录内需要区分：

- `model_best.pt`：最佳模型；`checkpoint_last.pt`：恢复训练的状态。
- `experiment.json`、`cached_feature_map.json`、`cached_feature_audit.json`、`cache_manifests/`：实际训练所用缓存与图像的身份和映射。
- `history.csv`、`training_report.json`：loss 和训练完成状态。
- `analysis/`：推理、聚类、城市和空间统计、Top100 证据；占这个目录的大部分空间。
- `analysis/cities/`、`analysis/results/`：逐城市推理数据和合并后的统计表；`analysis/decoder_groups.*`：4096 维到 64 组的对应关系。
- `analysis/evidence/`、`analysis/semantics_v2/`、`analysis/vlm/`：不同阶段的语义证据和组级解释。带 `pilot`、`overlay_confounded` 的文件是试验及问题排查记录。
- `analysis/semantic_umap_top100/unique_embeddings/`：约 1 GiB 的去重编码中间文件；同一原图和相同裁剪框只过一次 SigLIP，再展开到每维观察记录。它与顶层 embedding 的排列和用途不同，不能当成同名重复文件直接替换。
- `figures/`：当前主实验的论文图件。
- `source_cohort/`：删除 ViT-B 前迁出的候选清单和质量筛选记录。它是共用数据溯源，不能替代实际缓存清单；候选清单与实际训练城市存在替换。
- 多个 `runner_*.log`、`train_cached.log`、`extract_gpu*.log`：当前 ViT-L 的历史运行日志；文件存在不表示任务还在执行。

## 2. 当前主实验的分析报告：保留

| 目录 | 用途 | 本地占用 |
| --- | --- | ---: |
| `reports/vitl30_semantic_umap/` | 每个 SAE 维度一个点的图像语义 UMAP、64 组独立配色、示例证据 | 约 29 MiB |
| `reports/vitl30_instance_umap/` | 每维 Top10 图像实例的 UMAP | 约 105 MiB |
| `reports/vitl30_vlm_interpretation/` | 64 个聚类组的 VLM 解释 | 约 15 MiB |
| `reports/vitl30_dimension_vlm/` | 每个维度的 Top100 输入拼图、原始 VLM 回答、JSON/CSV/网页和后台运行状态 | 持续增长，扫描前约 209 MiB |
| `reports/vitl30_landscape_prototypes/` | NMF 景观组合探索、16/32/64 种分解和原图实例 | 约 15 MiB |
| `reports/vitl30_unit_motifs/` | 单维一致性检查、跨城市组合复现、峰值空间关系和复核图 | 约 32 MiB |

`vitl30_dimension_vlm/previous_prompt_v1/` 保存的是早期提示词的失败尝试和输入溯源，属于当前 ViT-L 工作。VLM 的标签仍是待核对的候选含义，JSON 格式通过不代表语义已验证。

`vitl30_landscape_prototypes/` 和 `vitl30_unit_motifs/` 是两个不同阶段的探索结果。后者发现大量组合可能来自语义冗余，因此不能把所有显著组合直接解释为不同的景观原型。

## 3. 较早的 BatchTopK 实验：目前仍保留

这些也使用 ViT-L，与刚删除的 ViT-B 实验不同。

| 位置 | 实验内容 | 本地占用 |
| --- | --- | ---: |
| `formal/gpu_run.py`、`formal/slurm/`、`formal/watchdog*.sh` | 较早的 307 城市候选训练、缓存和监控入口 | 小量代码 |
| `formal/analysis/` | 字典宽度对照、K16/32/64 消融、Sankey、语义 UMAP、共现和模板分析 | 小量代码 |
| `reports/scaling/` | W512→1024→2048 的交互流向网页和图像实例 | 约 668 MiB |
| `reports/sae_4096/` | W4096 下 K16、K32、K64 的对照图库 | 约 294 MiB |
| `reports/data/semantic_umap_w2048_k64_top100/` | 较早 W2048/K64 的 Top100 裁剪、embedding、UMAP 和网页 | 约 3.6 GiB |
| `reports/data/semantic_umap_w4096_k64_top100/` | 较早 W4096/K64 的同类结果 | 约 6.5 GiB |
| `reports/data/feature_combinations_w2048_k64/` | W2048/K64 的组合显著性和空间检验 | 约 49 MiB |
| `reports/data/scaling_heldout/`、`reports/figures/scaling_heldout/` | 字典尺度的留出样本统计和结果图 | 约 15 MiB |
| `reports/data/semantic_umap_pipeline_smoke/` | 32 张图的真实流程测试，不能作为主实验结果 | 约 372 KiB |

两份较早的 Top100 语义图库合计约 10 GiB，是目前历史结果占用的主体。它们不能与当前 W4096/K32 的 Top100 证据混用。

**`formal/` 不能整目录直接移走。** 当前 ViT-L 的语义 UMAP、逐维 VLM、实例 UMAP、景观原型和组合分析，仍从 `formal.analysis.semantic_feature_umap` 导入公共函数；`formal/quality/` 也保留共用的质量筛选工具。

## 4. 已归档的非 BatchTopK 内容

| 位置 | 内容 | 本地占用 |
| --- | --- | ---: |
| `archive/non_batchtopk/formal/` | Feature-MAE 时代的聚类、解释、共享性、稳定性、图件和旧网站 | 约 822 MiB |
| `archive/non_batchtopk/paper/` | 旧稿、Feature-MAE 和 graph visual vocabulary 报告、图、表和 PDF | 约 374 MiB |
| `archive/non_batchtopk/scripts/` | 旧训练、采样、图分析、可视化和数据处理代码 | 约 1.7 MiB |
| `archive/non_batchtopk/outputs/`、`results/`、`dashboard/` | 旧结果、统计 CSV 和仪表板 | 合计约 50 MiB |
| `archive/road_mrlu/` | 旧道路 MRLU 流水线、作业脚本和 README | 约 172 KiB |

归档整体约 1.2 GiB。这里的许多图像资产由 Git 跟踪；从工作目录删除它们，也不会自动清除提交历史里的副本。

## 5. 配置、文字和旧图中容易混淆的内容

### 城市和参数配置

- `configs/formal_city_manifest.json`：旧的 **37 城市、每城目标 12,800** 方案。
- `configs/final_global_300_google_city_map.json`、`final_global_300_sample_3000.json`：300 个 Google 城市的映射和抽样目标。
- `configs/final_global_7_baidu_city_map.json`：上海、杭州、南京、北京、天津、广州、深圳的百度目录映射。
- `configs/final_global_307_cities.csv`、`final_global_307_sample_3000.json`：合并后的 307 城市名单和抽样目标。
- `configs/dinov3_patch_batchtopk_w1024_k8.yaml`：较早的 W1024/K8 配置。

当前主实验应查自己的 `experiment.json` 和缓存映射，不能根据上述旧配置推断实际训练数据。

### 旧文字报告

| 文件 | 实际描述的版本 / 问题 |
| --- | --- |
| `reports/chapter5_results_brief.md` | 旧的 30 城市 W1024/K8 简报，仍有旧样本数和术语 |
| `reports/urban_scene_primitives_30city_writing_plan.md` | 旧的 30 城市 W1024/K8 写作提纲 |
| `reports/semantic_umap_latest_results.md` | “最新”指 2026-10-12 的较早 W4096/K64，并非现在的主实验 |
| `reports/semantic_feature_umap_plan.md` | 最初的 UMAP 方案，含当时找不到 checkpoint 的阶段性说明 |
| `reports/urban_scene_primitives_dictionary_scaling.md` | W512/1024/2048、K8 的历史尺度比较 |
| `reports/feature_combination_significance_w2048_k64.md` | 较早 W2048/K64 的组合显著性报告 |

前四份尤其容易被误当成当前说明；本次记录其问题，未重写历史报告。

`reports/figures/chapter5/` 仍有较早的论文结果图、流程图、panorama-aware 图和漫画；`reports/figures/legacy_12city/` 是更早的 loss 图。当前主实验图件位于主实验自己的 `figures/`。已对 `reports/figures/` 与当前 ViT-L `figures/` 做逐字节哈希比较，没有发现完全相同的图件副本；同一主题的不同版本仍可能并存。

## 6. 日志、缓存与 Git

- 又发现的 **16 个 ViT-B 遗留日志**（根 `outputs/nas_*.log`）已按前面的清理要求删除，合计 48,101 字节。完整路径和哈希追加到 `vitb_cleanup.json`。
- `formal/logs/` 内 **6 个 PID 文件**均已失效：3 个进程不存在，另 3 个是僵尸进程，没有运行中的训练命令；这些文件不能作为训练状态的证据。
- `.pytest_cache/` 和各目录 `__pycache__/` 是可再生缓存，占用很小。
- `.git/` 约 **4.6 GiB**，主要为历史对象。它不是当前模型输出，本次没有改写 Git 历史。
- 当前多份 `panorama_vitl30*.py`、`sae_experiments/models/` 和分析文件仍未被 Git 跟踪；本地存在不代表已经提交到 GitHub。

## 7. 本次扫描产物

此次记录了 **869 个代码、配置、文字和关键元数据文件**。本地工作目录共计约 **166 万个普通文件**，其中绝大多数是激活裁剪和历史图像资产；另有约 1.5 万个符号链接。扫描未遇到文件读取错误，Git 内部对象不计入这一文件数。后台 VLM 在继续写入，文件总数会变化。

- `remaining_code_and_metadata.csv`：代码、配置、文字报告和关键元数据的逐文件清单，含绝对路径、类别、体积、Git 跟踪状态。大量 crop 图像按目录汇总。
- `remaining_directories.csv`：各目录的文件数、逻辑体积、文件实际磁盘占用与符号链接数。
- `remaining_inventory.json`：扫描时间、文件类型统计、最大文件、目录统计和扫描错误。
- `figure_duplicates.json`：上述限定范围内的图件哈希重复检查。
- `vitb_cleanup.json`：之前清理及本次补删的 ViT-B 路径记录。

下一步整理时，可以将较早的尺度分析、消融和语义图库按实验版本归档，并给旧文字报告标明日期和模型配置。保留当前主实验、数据溯源和共用依赖。
