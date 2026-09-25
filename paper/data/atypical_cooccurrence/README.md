# Atypical visual co-occurrence

本目录是 `results/atypical_cooccurrence/` 的论文冻结快照，使用 378,818 张质控后方向图像、64 个细粒度视觉元素和每图 top-8 元素定义。

- `visual_element_distance.csv`：冻结 E+D+P 层级中的64元素两两距离及距离敏感性版本。
- `global_pair_statistics.csv`：全部2,016个元素对的观察值、期望值、Pearson residual、log lift、FDR和视觉距离加权指标。
- `global_top_pairs.csv`：按原始 `unexpected_score` 排序的前100个显著正向组合。
- `city_pair_statistics.csv`：30城市 × 2,016元素对的完整 city-versus-rest 统计量。
- `city_top_atypical_pairs.csv`：每个城市最高的20个显著 ACI 组合。
- `report.json`：运行参数与汇总结果。

三个主要指标为：

```text
unexpected_score = max(Pearson residual, 0) × visual distance
visual_bridge_index = max(log observed/expected, 0) × visual distance
city_atypicality_index = max(log OR city − log OR rest, 0) × visual distance
```

语义名称仅用于解释和制表，不参与距离、显著性或指标计算。
