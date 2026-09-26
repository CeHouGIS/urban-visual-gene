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

详细结果见 `multi_area_summary.csv` 和 `multi_area_direction_assignments.csv`。
