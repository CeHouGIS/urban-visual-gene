# Image-level atypical co-activation

本目录使用严格的同图共同激活定义。对于方向图像 $s$，细粒度元素 $i$ 只有在至少主导4个 DINOv3 patch 时才视为出现：

```text
q_si >= 4 / 196 = 2.0408%
```

元素对分数为 `unexpected_score = max(Pearson residual, 0) × E+D+P visual distance`。该元素对在具体图片中的分数进一步乘以双方 patch 占比的较小值：

```text
sample_pair_score = unexpected_score × min(q_si, q_sj)
```

单图异常度分别取该图最高 pair score，以及最高三个 pair score 的平均值。

- `global_pair_statistics.csv`：全部2,016个元素对的严格同图统计。
- `global_top_pairs.csv`：按 `unexpected_score` 排序的前100个显著组合。
- `image_atypicality.parquet`：378,818张图片的逐图异常度与最高分元素对。
- `top_atypical_images.csv`：去除重复 panorama 后最高分的500张方向图。
- `representative_pair_samples.csv`：前20个元素对各5张、每对 panorama 不重复的代表图片。
- `report.json`：阈值、样本量与验证摘要。

语义名称是层级冻结后的事后注释，不参与指标计算；解释时应同时检查元素 ID 与红/青 patch overlay。
