# Directional-image visual scene archetypes

聚类单元是单张方向街景，而不是城市或 1 km 网格。主表征将 14×14 patch winner map 映射到冻结的 64 个 E+D+P 细粒度元素，并汇聚成 3×3 画布。每个空间格先独立归一化、再赋予 1/9 权重，因此不会因 14 无法被 3 整除而让边缘格权重偏低。

主实验对 `64×3×3=576` 维空间组成做 Hellinger 变换和 PCA，在每城 2,000 张、panorama 去重的均衡样本上比较 K={6,8,10,12,15,20} 的 MiniBatchKMeans，然后给全部图片赋类。64 维全图元素占比是 composition-only baseline；`32×3×3` 粗元素空间布局和 512 原始维度全图占比是敏感性分析。

- `image_layout_archetypes.parquet`：全部图片的原型、质心距离和分配置信度。
- `archetype_profiles.csv`：原型规模、熵、主导元素和主导原始维度。
- `archetype_element_profiles.csv`：每个原型的 64 元素全图占比。
- `archetype_spatial_profiles.csv`：每个原型在 3×3 各格中的 64 元素条件占比。
- `archetype_dimension_profiles.csv`：每个原型的原始 512 维画布占比。
- `representative_images.csv`：最接近质心且 panorama 去重的代表图。
- `city_archetype_prevalence.csv`：事后城市分布；城市不参与聚类。
- `k_selection_metrics.csv`：空间表征与纯组成基线的 K 选择指标。
- `model_parameters.npz`：样本索引、PCA 参数、聚类中心与标签重排。

语义名称只用于事后说明，不参与聚类。结果应称为“方向街景视觉场景原型”或“街景画布原型”，不宜直接解释为平面城市形态。
