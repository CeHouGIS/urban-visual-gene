# 字典尺度：W512 → W1024 → W2048，K=8

在全球系列同一 3,000 万 patch 缓存上，固定 D=1024、K=8，比较 W=512/1024/2048。研究哪些特征稳定、哪些可能细分、哪些未匹配。不同模型的编号不直接对应，按 decoder 及同样本激活匹配。

## 文件

| 路径 | 内容 |
| --- | --- |
| code/analyze_dictionary_scaling.py | 固定 100,000 patch 的 decoder / 激活对照 |
| code/analyze_heldout_scaling.py | 验证全景匹配与细分候选 |
| code/plot_scaling_sankey.py | 流向图 |
| code/build_scaling_atlas.py | 父维度的四方向原图交互页 |
| code/discover_scene_templates.py | 早期场景组合探索 |
| outputs/heldout/ | 本机配对表、样本与统计 |
| reports/website/、reports/figures/ | 完整交互页与图 |
| results/ | GitHub 报告与精选静态图 |

初始原图证据为 295 城 × 4 全景 = 1180，后续扩充至每城 40 全景。具体连边使用哪个样本，应核对对应 run.json；增加展示实例不意味着全部连边已重算。

```bash
python -m formal.analysis.analyze_dictionary_scaling --help
python -m formal.analysis.analyze_heldout_scaling --help
```

模型位于 NAS 的 formal_batchtopk_307city_ablation_k512/、formal_batchtopk_307city_qc_filtered/、formal_batchtopk_307city_ablation_k2048/。旧 reports/scaling/ 与 reports/data/scaling_heldout/ 为本目录链接。

no assigned 表示在当前样本与阈值下没有得到匹配，不能直接等同新语义。细分需要结合多个独立原图实例核查。
