# 同一地点四方向 Feature-MAE / Graph 结果

- 城市：`UnitedStates/NewYorkCity`
- Panorama：`BFlilbUSNYggO3YcnRY3kg`
- 坐标：`40.895175, -73.859796`
- 方向：`0° / 90° / 180° / 270°`
- 原分析缓存已有：`2` 个方向
- 从原始 TAR 补算：`2` 个方向
- 一致性处理：四个方向均重新通过同一 DINOv3 + 冻结 Feature-MAE
- 四方向 archetype：`A05 / A07 / A01 / A04`

主结果图：`paper/figures/supplementary/four_direction_area/Fig_Four_Direction_Result_Stitch.png`

照片激活拼接图：`paper/figures/supplementary/four_direction_area/Fig_Four_Direction_Photo_Activation_Stitch.png`

主图按列固定为北、东、南、西；按行依次是照片与 F 激活叠加、14×14 F map、image graph、过滤小支持后的 512D winner 激活。512D 图只显示至少占 4/196 patches 的维度，完整未过滤计数保留在 `winner_patch_counts.csv`。
