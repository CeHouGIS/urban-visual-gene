# Feature-MAE Scale-up 实验计划

**版本日期：** 2026-09-23

**状态：** 设计完成，尚未启动正式训练

**数据范围：** 质控后的30个城市街景图像，不使用外部数据

## 1. 研究问题

本实验回答两个相互独立的问题：

1. 当每个城市使用更少的训练图像时，Feature-MAE 是否仍能恢复与完整数据相似的潜在维度及32/64级视觉类别？
2. 当 Feature-MAE 的 latent width 改为128、256或1024时，是否仍能恢复与512维参考模型相似的视觉类别？

这里的“相同”不指维度编号相同。不同随机初始化和不同模型宽度下，潜在维度可以重新排列、拆分或合并，因此必须通过跨模型表征匹配进行评价。

## 2. 核心假设

- **H1（样本效率）：** 随着每城市样本数增加，维度和类别稳定性逐渐提高，并在某一规模后进入平台期。
- **H2（容量稳定性）：** 256、512和1024维模型能够恢复大部分共同视觉类别；1024维可能将部分已有类别进一步细分。
- **H3（容量下界）：** 128维模型仍可恢复主要粗粒度视觉家族，但在64类细粒度层级上可能出现容量不足。
- **H4（下游稳定性）：** 如果视觉词汇稳定，城市构成、城市相似性和共存结构也应在不同设置之间保持较高一致性。

## 3. 数据划分

所有实验只使用质量控制后的378,818张图像。划分必须按 `panoid` 分组，避免同一全景的不同方向跨越训练、验证和公共评价集。

每个城市固定保留：

```text
validation set: 500 images
common evaluation set: 500 images
training pool: up to 10,000 images
```

公共评价集共约15,000张图像，所有模型均在这组完全相同的图像上计算跨模型匹配和下游稳定性。划分时尽量平衡0°、90°、180°和270°方向。若按全景分组导致目标数量不能精确整除，允许最后一个全景内确定性截断，但该全景不能进入其他划分。

训练子集采用嵌套设计：

```text
500 subset of 1,000
1,000 subset of 2,000
2,000 subset of 4,000
4,000 subset of 8,000
8,000 subset of 10,000
```

这样可以减少不同样本量之间由抽样内容变化引起的额外方差。数据划分种子固定，模型初始化种子独立变化。

## 4. 统一训练配置

除被研究的样本量或 latent width 外，其余设置保持不变：

```text
DINOv3 backbone: frozen ViT-B/16
input: 196 x 768 patch tokens
mask ratio: 75%
encoder depth: 4
encoder heads: 8
decoder width: 256
decoder depth: 2
decoder heads: 8
objective: cosine reconstruction loss on masked patches
optimizer: AdamW
early stopping: validation loss, patience = 3
hierarchy views: E + D + P
hierarchy cuts: K = 32 and K = 64
```

主实验保持相同的 optimizer update 数，以避免小数据模型仅因更新次数更少而表现较差。固定 epoch 的自然训练结果可作为补充分析。

## 5. 实验A：每城市训练样本量

固定 latent width 为512。

| Run group | 每城市训练图像 | 总训练图像 | 模型种子 |
|---|---:|---:|---|
| A0500 | 500 | 15,000 | 42 |
| A1000 | 1,000 | 30,000 | 42, 43, 44 |
| A2000 | 2,000 | 60,000 | 42 |
| A4000 | 4,000 | 120,000 | 42, 43, 44 |
| A8000 | 8,000 | 240,000 | 42 |
| A10000 | 10,000 | 300,000 | 42, 43, 44 |

该部分共12个正式训练任务。A10000的三个种子构成受控参考模型，并用于估计同配置训练的自然随机波动上限。

### 5.1 同宽度维度匹配

所有模型均为512维，因此可以进行一对一维度匹配：

1. 分别计算每个维度的编码器方向 E、解码器方向 D 和空间响应 P；
2. 对三个相似度矩阵分别进行rank normalization；
3. 等权融合 E、D、P，不加入城市分布 C 或共存 Q；
4. 使用 Hungarian algorithm 得到512个维度的一对一对应；
5. 在匹配后的维度集合上比较32类和64类划分。

### 5.2 主要指标

```text
mean and median matched-dimension similarity
dimension-match similarity distribution
32-cluster ARI and NMI
64-cluster ARI and NMI
category-prototype cosine similarity
representative-image overlap at 10 and 50
patch-level category ARI and NMI
coarse-to-fine parent agreement
```

## 6. 实验B：Feature-MAE latent width

固定每城市10,000张训练图像，使用与A10000完全相同的训练、验证和公共评价划分。

| Run group | Latent width | 平均维度数/64类 | 模型种子 | 定位 |
|---|---:|---:|---|---|
| B128 | 128 | 2 | 42, 43, 44 | 容量下界压力测试 |
| B256 | 256 | 4 | 42, 43, 44 | 低容量模型 |
| B512 | 512 | 8 | 42, 43, 44 | 参考模型，与A10000共用 |
| B1024 | 1,024 | 16 | 42, 43, 44 | 高容量模型 |

该部分有12个配置实例，其中B512的3个实例与A10000重复，因此只新增9个训练任务。两个实验合计21个正式训练任务。

128维模型仍切分为32和64类，以直接测试当前类别分辨率的容量下界。但由于64类平均仅包含2个维度，必须同时报告32类结果，且不能仅凭64类不稳定就断言视觉结构不存在。

### 6.1 不同宽度的维度匹配

不同宽度不能强制全部维度一对一对应。维度层面采用：

```text
bidirectional nearest-neighbour similarity
reciprocal nearest-neighbour rate
small-to-large representation coverage
large-to-small representation coverage
dimension split/merge rate
```

必要时使用最优传输作为补充，但不将其作为唯一结论来源。

### 6.2 不同宽度的类别匹配

每个模型独立使用 E+D+P 构建层次树并切成32类和64类。随后：

1. 在公共评价集上计算每个类别的图像激活原型和空间原型；
2. 对32类和64类分别构造类别相似度矩阵；
3. 使用 Hungarian algorithm 进行等数量类别的一对一匹配；
4. 将每个评价patch的获胜维度转换为其所属类别；
5. 比较匹配后的patch级类别分配、代表图像和层级父子关系。

类别匹配不使用城市身份、城市分布、共存结构或语义标签。

## 7. 下游结论稳定性

维度或类别相似并不自动意味着论文结论稳定。每个模型还需在公共评价集上重新计算：

```text
30-city category prevalence matrix
30 x 30 city cosine-similarity matrix
30 x 30 city Jensen-Shannon matrix
city visual-diversity ranking
global category co-occurrence matrix
city-specific co-occurrence similarity
```

比较指标包括Pearson、Spearman、矩阵上三角相关性以及top-k城市近邻重合率。城市构成和共存仅作为下游评价，不参与 E+D+P 类别定义。

## 8. 统计判定

首先使用A10000/B512三个种子的两两比较，估计同配置模型的可重复性上限。较小样本量或其他宽度若满足以下条件，可认为基本恢复了参考视觉词汇：

1. 主要稳定性指标达到参考模型跨种子中位数的至少95%；
2. 95% bootstrap置信区间与参考模型跨种子区间重叠；
3. 城市构成和城市相似性矩阵与参考结果保持高度相关；
4. 结论不依赖单个异常城市或少数超高频类别。

同时描述性报告匹配相似度大于0.70和0.80的类别比例。置信区间以城市为bootstrap单位，避免将同一城市内高度相关的图像当作独立样本。

“最小充分样本量”定义为：稳定性曲线首次进入参考水平95%范围，并在更大样本量下不再出现实质下降的最小每城市图像数。

## 9. 建议图表

```text
Fig_Sample_Size_Stability
  x = images per city (log scale)
  y = matched similarity / ARI / NMI / downstream correlation

Fig_Latent_Width_Stability
  x = 128, 256, 512, 1024
  y = coarse and fine category recovery

Fig_Cross_Model_Category_Matching
  category-matching matrices for each width against W512

Fig_Category_Recovery_Examples
  matched representative images and activation maps

Fig_Dimension_Split_Merge
  dot-matrix representation of dimension splitting and merging
```

不使用容易造成路径混乱的Sankey作为主图；拆分和合并关系优先使用匹配矩阵或dot plot。

## 10. 输出目录

```text
results/scaling/
    experiment_registry.csv
    split_audit.csv
    sample_size_stability.csv
    latent_width_stability.csv
    dimension_matching_summary.csv
    category_matching_32.csv
    category_matching_64.csv
    downstream_stability.csv
    failure_and_resume_log.csv

paper/figures/supplementary/
    Fig_Sample_Size_Stability.png
    Fig_Latent_Width_Stability.png
    Fig_Cross_Model_Category_Matching.png
    Fig_Category_Recovery_Examples.png
    Fig_Dimension_Split_Merge.png
```

大型checkpoint、逐图激活和中间缓存保留在 `outputs/experiments/`，不复制到 `paper/` 或GitHub。

## 11. 分阶段执行顺序

1. 冻结按panoid分组的数据划分并完成审计；
2. 对W128、W256、W512和W1024各执行短程smoke test；
3. 训练A10000/B512的seed 42参考模型；
4. 训练样本量曲线的seed 42模型；
5. 训练B128、B256、B1024的seed 42模型；
6. 确认指标和资源消耗正常后补充seed 43和44；
7. 冻结全部统计层级后再生成代表图像和语义解释；
8. 汇总21个正式任务并绘制论文图。

如果seed 42阶段已经显示某个配置训练失败或表示完全坍缩，应先诊断原因，不直接并行启动其余种子。
