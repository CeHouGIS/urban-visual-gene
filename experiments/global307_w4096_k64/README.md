# 全球 W4096 / K64：Top100 语义 UMAP

引用 [K 消融](../global307_w4096_k_sweep/README.md) 的 K64 模型：D1024、W4096、K64，304 城、3,000 万 patch、60 epochs。

证据为 295 城 × 4 个训练排除全景 = 1180 全景、4720 方向图。完整全景共同计算 BatchTopK，按方向图最大 patch 响应排名，取每维 Top100；峰值连通区域经 SigLIP、逐例归一化、逐维均值后 cosine UMAP。

历史结果：400240 个裁剪；3885 维完整 Top100、196 维有20–99例，共4081维进入UMAP，15维证据不足。排名只适用于这批图像。

- code/run_latest_semantic_umap.sh：当时的运行脚本，latest 不表示当前主实验。
- outputs/semantic_top100/：本机完整约6.5GB证据、裁剪和交互页。
- reports/REPORT.md：历史报告；results/：GitHub 摘要、坐标与图。
- 共用实现：sae_experiments/semantic_tools/，兼容入口 formal.analysis.semantic_feature_umap。

```bash
python -m formal.analysis.semantic_feature_umap --help
```

checkpoint 在 NAS formal_batchtopk_307city_w4096_k64/sae_448_k4096.pt。旧 reports/data/semantic_umap_w4096_k64_top100/ 是链接；不与主实验 W4096/K32 Top100 混用。
