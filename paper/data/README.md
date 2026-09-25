# Paper data snapshots

本目录只包含支撑论文表格和图形的冻结结果快照。文件按科学问题划分，而不是按脚本划分：视觉词汇、城市构成、元素共现、异常视觉组合、空间组织、UMAP 和语义解释。

主流水线的规范输出仍位于项目根目录 `results/`。重新运行实验时先更新 `results/`，通过核验后再刷新本目录；不要直接把本目录作为训练输出位置。

`umap/image_umap_coordinates.parquet` 包含 378,818 张质控后图像的二维坐标和标签。体积更大的 CSV 版本只保留在根目录 `results/`，避免论文包内重复两种等价格式。

`atypical_cooccurrence/` 保存新指标的完整冻结结果。其中 `unexpected_score` 是原始 Pearson residual × E+D+P distance，`visual_bridge_index` 是 log-lift × distance，`city_atypicality_index` 是 city-versus-rest log-odds difference × distance。

`image_level_atypicality/` 是严格的同图版本：一个元素只有在同一张方向图中主导至少4/196个 patch 才算出现。它同时保存元素对分数、每张图的最大及 top-3 异常度，以及用于激活叠加图的代表样本。

`image_graph_compact_archetypes/` 保存 201 维紧凑图编码实验的冻结摘要：64 维面积组成、1 维边界密度与 136 维组成校正谱拓扑。大型逐图特征、模型和 parquet 文件仍仅保留在本地。
