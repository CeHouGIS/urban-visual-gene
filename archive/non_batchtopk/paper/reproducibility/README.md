# Reproducibility records

- `analysis_provenance.json`：当前内部分析的输入、参数、随机种子和输出来源记录。
- `atypical_cooccurrence_report.json`：异常视觉组合分析的样本量、阈值、距离定义、显著性检验和输出摘要。实现脚本位于 `scripts/multicity/analyze_atypical_cooccurrence.py`。
- `image_level_atypicality_report.json`：严格同图四-patch激活分析的阈值、样本量、元素对数量和逐图计分定义。实现脚本位于 `scripts/multicity/analyze_image_level_atypicality.py`。

训练 checkpoint、日志和大型中间数组仍保存在项目根目录的 `outputs/` 与 `logs/`，避免在论文包中复制。
