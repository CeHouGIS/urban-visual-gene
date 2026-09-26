# 多地区四方向 Feature-MAE / Graph 结果

本实验选择 10 个不同城市中的代表性 panorama，每处均使用同一点的 0°、90°、180°、270° 四张方向图。原缓存缺少的方向从原始 TAR 补齐，并将全部 40 张图统一通过相同的 DINOv3 与冻结 Feature-MAE 重算。

- 地区数：`10`
- 城市数：`10`
- 方向图总数：`40`
- 原缓存已有方向：`17`
- 补齐方向：`23`
- 代表性覆盖：A01–A08，并增加伊斯坦布尔、巴黎两个地区

照片激活总图：`paper/figures/supplementary/multi_area_four_directions/Fig_Multi_Area_Four_Direction_Atlas.png`

Graph 总图：`paper/figures/supplementary/multi_area_four_directions/Fig_Multi_Area_Four_Direction_Graphs.png`

方向 archetype 矩阵：`paper/figures/supplementary/multi_area_four_directions/Fig_Multi_Area_Archetype_Matrix.png`

地点级四方向拼接、重叠融合 14×56 heatmap 与环形 graph：`paper/figures/supplementary/multi_area_four_directions/Fig_Multi_Area_Stitched_Heatmap_Graph.png`

## 跨方向一致性修正

为避免四张方向图分别经过 Feature-MAE 后在边界处形成 512D 激活断层，本版在原有 0°、90°、180°、270° 窗口之间增加 45°、135°、225°、315° 四个跨边界 token 窗口。冻结 Feature-MAE 对 8 个窗口重新推理，使用 sine-squared 权重先融合原始 512D 激活，最后才选择 winner dimension 并映射到 64 个 F 类别。

- 512D 边界平均余弦相似度：`0.6774 → 0.8316`
- winner dimension 边界一致率：`11.96% → 26.61%`
- F category 边界一致率：`13.04% → 35.36%`
- 峰值 GPU 显存：`0.075 GiB`
- Graph：直接从融合后的 `14×56` map 构建，包含最右列到最左列的环形邻接，共 `1,512` 对 patch adjacency

推理报告：`overlap_panorama_inference_report.json`

Graph QA：`overlap_panorama_graph_report.json`

像素级重叠窗口对照实验：`PIXEL_OVERLAP_EXPERIMENT.md`

三种方法对照图：`paper/figures/supplementary/multi_area_four_directions/Fig_Pixel_Overlap_Panorama_Comparison.png`

交互网页：`dashboard/four-directions.html`

详细结果见 `multi_area_summary.csv` 和 `multi_area_direction_assignments.csv`。
