# 全球 W2048 / K64：语义与特征组合

ViT-L/16、D1024、28×28 patch / 方向、全景 context mixing；304 城市、3,000 万 patch、60 epochs，W2048/K64。checkpoint 在 NAS formal_batchtopk_307city_w2048_k64/sae_448_k2048.pt。

## 分析

- 逐维语义：295 城各 4 个训练排除全景，共 1180 全景、4720 方向图；每维 Top100 区域经 SigLIP 汇总后 UMAP。
- 共现显著性：295 城各 40 全景，共11800；207 城发现、88 城验证。发现集选 100 对候选；验证集按城市和活跃度分层，9999 次置换、BH 校正。以语义 embedding 相似度筛去近重复；100 对均 q=0.0001，median lift=2.131。
- 空间跟进：前 5 对，每对至多 120 个验证全景，环形平移作为参考。关联仍可能受语义冗余和空间依赖影响。

code/ 保存语义运行、共现、空间检验与示例图脚本。共用工具在 sae_experiments/semantic_tools/。

```bash
python -m formal.analysis.test_feature_combinations --help
python -m formal.analysis.test_feature_pair_spatial --help
```

outputs/semantic_top100/ 与 outputs/feature_combinations/ 保存本机完整证据；原报告在 reports/feature_combinations.md；GitHub 摘要和表格在 results/。旧 reports/data/*w2048_k64* 为链接。
