# Chapter 5 Graph Visual Vocabulary 实验通俗解读

## 1. 一句话概括

这次实验最重要的结果不是“找到了 39 种街景物体”，而是：

> Feature-MAE 的 512 个视觉响应维度并不是杂乱无章的。根据它们在模型内部的表示关系和在完整四方向全景中的空间响应关系，可以自然组织成大约 35 个相对稳定的视觉社区。这些社区在真实图像空间中也具有显著的一致性，并且在不同城市中表现出不同的比例、组合和空间分布。

---

## 2. 输入数据到底是什么？

原研究计划假设每个样本是一张独立方向影像，对应 `14×14` patches。但当前 pipeline 已经把同一个街景点的四个方向联合处理：

```text
0°、90°、180°、270°
→ 拼接为 2560×640 全景
→ 等比例缩放为 896×224
→ DINOv3 提取 14×56×768 tokens
→ Feature-MAE 输出 14×56×512 latent activations
```

所以现在一个样本代表一个完整的四方向街景采样点，而不是一个单独方向。

每个街景点包含：

\[
14\times56=784
\]

个 patch positions，每个 patch 对应 512 个 Feature-MAE latent dimensions。

本实验实际使用：

- 30 个城市；
- 296,462 个完整街景采样点；
- 232,426,208 个 patches；
- 512 个冻结的 Feature-MAE latent dimensions。

因此空间 profile 应为：

```text
P.shape = 512 × 784
```

而不是原计划中的 `512×196`。

---

## 3. 512 个 latent dimensions 是什么？

可以把 512 个 dimensions 想象成 512 个视觉感受器。

有些维度可能主要响应：

- 道路或建筑的水平边缘；
- 绿色植被纹理；
- 建筑立面的重复结构；
- 天空和高亮区域；
- 阴影或强对比区域；
- 图像底部的道路结构；
- 某种颜色、材质、几何或空间位置。

实验开始时没有给这些维度人工语义标签。实验希望回答：

> 这些维度能否根据它们自身的关系，自动组织成若干反复出现的视觉模式？

---

## 4. Graph 中的节点和边是什么？

Graph 不是城市之间的网络，也不是图像之间的网络。

- 每个 node 是一个 Feature-MAE latent dimension；
- 一共 512 个 nodes；
- 两个 dimensions 越相似，它们之间的 edge weight 越大。

因此，这个 graph 表示的是：

> Feature-MAE 内部 512 个 latent visual responses 之间的关系网络。

最终融合图包含：

- 512 个 nodes；
- 24,673 条 edges；
- density = 0.1886；
- 1 个 connected component。

最后一点表示全部 512 个 dimensions 都处于同一个关系网络中，没有完全孤立的子系统。

---

## 5. 为什么使用 Encoder、Decoder 和 Spatial 三个 views？

实验从三个角度判断两个 dimensions 是否相似。

### 5.1 Encoder view

Encoder direction 表示一个 latent dimension 如何读取 DINOv3 的 768 维输入。

通俗地说，它回答：

> 这个 dimension 喜欢接收什么样的视觉信息？

### 5.2 Decoder view

Decoder direction 表示一个 latent dimension 激活后，如何映射回 DINO feature space。

它回答：

> 这个 dimension 倾向于表达什么样的视觉方向？

### 5.3 Spatial view

Spatial profile 统计一个 dimension 在完整四方向全景的哪些位置成为 patch winner。

例如：

- 某个 dimension 总在全景上方响应；
- 某个 dimension 主要在底部道路区域响应；
- 某个 dimension 在四个方向都有类似响应；
- 某个 dimension 集中在某些横向位置。

它回答：

> 这个 dimension 通常在全景的什么位置出现？

三个 views 可以概括为：

```text
Encoder：喜欢读取什么
Decoder：倾向于表达什么
Spatial：通常在哪里出现
```

---

## 6. 为什么不直接拼接三组向量？

三个 views 的维度和数值分布不同：

```text
Encoder: 512×768
Decoder: 512×768
Spatial: 512×784
```

如果直接 concatenate，某个 view 可能因为维数或数值尺度更大而主导最终结果。

所以实验采用：

1. 每个 view 单独计算相似度；
2. 每个 view 单独构建 kNN graph；
3. 每个 graph 单独归一化；
4. 三个 graph 等权融合。

最终：

\[
W=\frac{\bar W_E+\bar W_D+\bar W_P}{3}.
\]

可以理解为让三个 views 各投一票。

---

## 7. kNN graph 是什么意思？

每个 dimension 只保留最相似的 30 个邻居，而不是与其余 511 个 dimensions 全部连接。

这样做可以：

- 保留局部关系；
- 去掉大量很弱的连接；
- 避免 graph 变成缺少结构的全连接网络；
- 让 Leiden 更容易识别局部社区。

实验采用 symmetric kNN union：只要 `i` 把 `j` 当作近邻，或者 `j` 把 `i` 当作近邻，就保留 edge `(i,j)`。

---

## 8. 为什么 Spatial graph 有 12 个 connected components？

有 11 个 dimensions 从未成为任何 patch 的 Top-1 winner：

```text
D026, D043, D071, D076, D087, D134,
D367, D402, D475, D484, D493
```

所以它们的 Top-1 spatial profile 全部为零。

这不代表它们从来没有激活，只代表它们从来没有成为 512 个 dimensions 中激活最高的那一个。

因此，只看 Spatial graph 时存在：

- 1 个主要 component；
- 11 个零空间 profile 节点；
- 合计 12 个 components。

这些节点在 Encoder 和 Decoder views 中仍然存在有效关系，所以三个 views 融合后，整个 graph 自然变成一个 connected component。实验没有人工补边。

---

## 9. Leiden community detection 在做什么？

Leiden 在 graph 中寻找：

- 内部连接相对强；
- 与外部连接相对弱

的节点群体。

一个 community 可以理解为：

> 一组 representational 和 spatial characteristics 相对相似的 latent visual responses。

它不一定直接对应“树”“道路”“建筑”等人工语义类别，也可能对应：

- 某种纹理；
- 某种材质；
- 某种边缘或几何；
- 某种光照；
- 某种空间位置；
- 多种相关视觉响应的组合。

---

## 10. Gamma 是什么意思？

Gamma 控制 Leiden 划分的精细程度。

- gamma 较小：社区较少、范围较大；
- gamma 较大：社区较多、划分较细。

可以把它理解成地图的缩放尺度：低 gamma 类似把一个国家分成几个大区域，高 gamma 则继续划分成省、市或县。

本实验扫描：

\[
\gamma=0.05,0.10,\ldots,3.00.
\]

研究目标不是寻找恰好产生 32 或 64 类的 gamma，而是寻找：

> 哪一段连续 gamma 改变以后，社区数量和节点分组仍然比较稳定？

---

## 11. 为什么每个 gamma 要运行 20 个随机种子？

Leiden 存在随机性。同一个 gamma 在不同随机种子下可能得到稍有差别的 partition。

如果一个 gamma 下 20 次结果都很相似，说明社区结构不是某一次随机初始化偶然造成的。

实验使用两个指标：

### NMI

衡量两个 partitions 的信息结构是否相似：

- 1：几乎相同；
- 0：基本没有一致性。

### ARI

衡量节点对是否被一致地分到同一个 community。ARI 通常比 NMI 更严格。

每个 gamma 的 representative partition 不是 modularity 最大的单次结果，而是与其余 19 次运行平均 NMI 最高的 medoid partition。

---

## 12. Stable plateau 是什么？

稳定结构不应该只在某一个 gamma 偶然出现，而应该在一段连续 gamma 中持续存在。

本实验要求候选 plateau：

- 至少连续 3 个 gamma points；
- mean seed NMI 达到阈值；
- adjacent-resolution NMI 达到阈值；
- 区间内 `max(K)-min(K)≤2`；
- 排除 K=1 的平凡稳定 partition。

在严格阈值 0.90 下没有找到 plateau。按照预先规定的规则，将阈值放宽到 0.85 后发现：

\[
\gamma=2.65\sim2.75
\]

是一个稳定区间。

该区间中：

- community number 约为 35–37；
- mean seed NMI = 0.856；
- mean adjacent-resolution NMI = 0.886。

最终选择：

\[
\gamma=2.70,\quad K=35.
\]

这就是当前最有证据支持的稳定视觉组织尺度。

---

## 13. K=35 和 K=39 是什么关系？

这是整个结果中最容易混淆的部分。

### K=35：稳定主尺度

```text
gamma = 2.70
K = 35
status = stable
```

K=35 位于满足稳定性规则的 plateau 中，是论文方法上最可靠的主要结果。

### K=39：探索性 fine scale

```text
gamma = 2.85
K = 39
status = exploratory_secondary
```

K=39 提供更细粒度划分，适合：

- 生成 community evidence sheets；
- 计算城市 prevalence；
- 进行 PPMI 共现分析；
- 观察更小的视觉模式。

但是 K=39 没有处于第二个稳定 plateau 中，所以只能称为 exploratory fine scale。

论文中适合写：

> 实验识别出一个 K=35 的稳定主尺度，并使用 K=39 作为探索性的细粒度辅助尺度。

不适合写：

> 实验发现了两个同等稳定的 coarse/fine 层级。

---

## 14. 为什么 coarse 和 fine 不是严格嵌套？

不同 gamma 下的 Leiden partitions 是分别计算的。

一个 fine community 可能有 80% 的 dimensions 来自某个 coarse community，另外 20% 来自另一个 coarse community。因此 fine communities 不一定完整包含在某个 coarse community 中。

实验计算结果：

- mean fine-to-coarse purity = 0.843；
- median purity = 0.955；
- minimum purity = 0.167；
- 整体不是严格嵌套结构。

所以更准确的名称是：

> multiscale community structure

而不是：

> strict hierarchical taxonomy

---

## 15. Community medoid 是什么？

每个 community 都选择一个代表 dimension。

选择的不是编号最小的维度，也不是随机维度，而是：

> 在该 community 内部，与其他成员连接总强度最大的 dimension。

例如：

```text
F001
community size = 27
medoid = D031
```

D031 可以看作 F001 在 graph 中最居中的 representative，适合用于寻找代表图片和生成 evidence sheet。但一个 medoid 不能完全替代整个 community 内所有 dimensions 的视觉差异。

---

## 16. Conductance 为什么较高？

Conductance 衡量一个 community 有多少连接流向外部：

- conductance 较低：community 相对封闭；
- conductance 较高：community 与其他 communities 仍有较多连接。

本实验不少 communities 的 conductance 较高，可能因为：

- 融合 graph 包含 24,673 条边，整体较密；
- 三个 views 使用 kNN union；
- Feature-MAE 的视觉响应可能是连续变化的；
- 一个 dimension 可能同时与多个视觉模式相关。

所以这些 communities 不应被理解为 39 个相互排斥的物体类别，而更适合解释为：

> latent response network 中相对聚集的视觉响应群体。

---

## 17. 独立视觉一致性验证在验证什么？

Graph 是根据 Encoder、Decoder 和 Spatial 三个 views 建立的。

为了避免循环论证，实验另外取每个 dimension 的 Top-10 activated panorama crops，用冻结 DINOv3 提取独立图像 embedding。这些 embeddings 完全不参与 graph construction 或 Leiden community detection。

然后比较：

### 社区内部相似度

\[
S_{within}=0.3530
\]

### 社区之间相似度

\[
S_{between}=0.2989
\]

### 差值

\[
0.3530-0.2989=0.0541
\]

这说明同一个 community 中不同 dimensions 的代表图像，平均来说确实比不同 communities 之间更加相似。

---

## 18. Cohen's d=0.414 应该怎么看？

Cohen's d 衡量组间差异相对于数据波动的大小。

粗略理解：

- 0.2：较小效应；
- 0.5：中等效应；
- 0.8：较大效应。

本实验：

\[
d=0.414.
\]

因此最合适的解释是：

> Community visual coherence 明确存在，达到中等强度，但不是特别强。

不能据此声称每个 community 都是语义完全纯净的视觉类别。

---

## 19. Permutation p=0.000999 是什么意思？

实验进行了 1,000 次随机置换：

1. 保持每个 community 的大小不变；
2. 随机将 512 个 dimensions 重新分组；
3. 计算随机分组的平均社区内部图像相似度。

真实分组的内部相似度高于几乎所有随机结果。

由于使用 1,000 次 permutation，可报告的最小 p 值约为：

\[
\frac{1}{1001}=0.000999.
\]

这说明真实 graph communities 的视觉一致性不是随机社区大小造成的。

但需要区分：

- p-value 很小：结果不太可能由随机分组产生；
- Cohen's d 中等：真实差异存在，但强度不是极大。

---

## 20. 为什么 Qwen 标签很多是“城市街景”？

Graph communities 不一定对应单一物体。

一个 community 可能同时表达：

- 路面边缘；
- 建筑纹理；
- 阴影和光照；
- 图像位置；
- 某种空间构图；
- 多种相关的局部视觉响应。

Qwen 能相对明确识别的少数 communities 包括：

- F002：道路与农田；
- F023：日落街景；
- F024：稻田与道路；
- F026：工业区；
- F034：道路与田野；
- F036：黄昏街景。

但大量 communities 被保守地命名为“城市街景”，而且 confidence 偏低。这说明当前 evidence sheet 尚不足以让 VLM 稳定区分细微的材质、几何、纹理和空间位置。

因此：

> Qwen 标签只能辅助检索和人工检查，不能作为社区有效性的主要证据，也不应该直接作为最终 taxonomy 名称。

---

## 21. 城市 visual composition 是怎么计算的？

每个完整全景有 784 个 patch winners。

首先映射：

```text
winner latent dimension
→ fine community
```

然后统计某座城市全部 patches 中，每个 community 的比例：

\[
p_{ck}.
\]

结果显示：

- 39 个 fine communities 中有 36 个出现在全部 30 个城市；
- 三个未在所有城市出现的 communities 都是极低 prevalence 的小社区；
- 城市之间的 composition similarity 普遍较高。

最相似城市对：

```text
Jakarta–Manila
cosine similarity = 0.997
```

相对最不相似城市对：

```text
Dhaka–Osaka
cosine similarity = 0.893
```

即使最低值仍然较高，说明 30 个城市在宏观上共享相当大的一部分 latent visual vocabulary。城市差异更多表现为同一套 communities 的 prevalence 不同。

---

## 22. 城市 entropy 是什么意思？

Entropy 衡量一座城市的 community prevalence 是否均匀。

### 低 entropy

少数 communities 占比较高，composition 更集中。

### 高 entropy

多个 communities 的占比更加平均。

本实验 entropy 较低的城市包括：

- Amsterdam；
- London；
- Mumbai；
- Toronto；
- Los Angeles。

较高的包括：

- Buenos Aires；
- Johannesburg；
- Mexico City；
- Seoul；
- Bangkok。

这里的 entropy 不能直接解释为城市是否有趣、是否宜居或是否具有更好的规划。它只表示当前 39 个 communities 下 prevalence distribution 的均匀程度。

---

## 23. PPMI 共现在研究什么？

Composition 只回答每种 visual community 有多少。

PPMI 回答：

> 哪些 visual communities 经常在同一个完整全景中一起出现，而且其共同出现次数高于独立随机预期？

实验对每个全景选择 patch count 最高的 Top-8 communities，并沿用旧实验的 PPMI 和 support threshold 定义。

共得到 104 条正且满足 support threshold 的 community pairs。最高的包括：

- F009–F028；
- F018–F028；
- F013–F014；
- F007–F028；
- F008–F011。

这些 community IDs 仍需要结合 evidence sheets 人工解释，不能仅依赖当前宽泛的 Qwen 标签。

---

## 24. 为什么 composition 和 co-occurrence 不一样？

两个城市可能拥有相似的视觉比例，但组合方式不同。

例如两个城市都包含：

```text
20% 植被
30% 道路
30% 建筑
20% 天空
```

但是城市 A 可能经常出现：

```text
植被 + 低层住宅
```

城市 B 可能经常出现：

```text
植被 + 高层建筑
```

因此：

- composition 描述“有什么、各有多少”；
- co-occurrence 描述“它们如何一起出现”。

本实验两个城市相似度矩阵的 Spearman correlation 为：

\[
\rho=0.457.
\]

说明二者相关，但明显不是同一个结构。

---

## 25. Spatial homogeneity 是什么？

实验将城市划分成 1 km grids，每个 grid 都有一个 39 维 visual composition。

然后比较同一城市中不同 grids 的 composition similarity。

### Homogeneity 高

城市内部不同地区的视觉组成比较相似。

### Homogeneity 低

城市内部不同地区差异更大。

homogeneity 较高的城市包括：

- Singapore；
- Mumbai；
- Bogota；
- Lagos；
- Sao Paulo。

较低的包括：

- Dhaka；
- Moscow；
- Vienna；
- Paris；
- Seoul。

这些只是当前 representation 与 grid definition 下的相对描述，不代表城市规划质量或空间优劣。

---

## 26. Local continuity 是什么？

Local continuity 只比较空间相邻的 grid cells。

它回答：

> 一个地区和旁边地区的视觉组成是否连续？

本实验 local continuity 较高的城市包括：

- Bogota；
- Lagos；
- Singapore；
- Mumbai；
- Cape Town。

Homogeneity 和 continuity 并不相同：

- homogeneity 比较全城市任意地区；
- continuity 只比较空间邻近地区。

一个城市可以在整体上比较异质，但局部变化仍然平滑连续。

---

## 27. Moran's I 是什么？

Moran's I 检验某个 community 在城市中的空间分布是否聚集。

- Moran's I > 0：相似值在空间上聚集；
- Moran's I ≈ 0：接近随机分布；
- Moran's I < 0：相邻地区倾向于不同。

本实验对：

\[
30\text{ cities}\times39\text{ communities}=1170
\]

个 city-community combinations 计算了 Moran's I。

它回答：

> 某种视觉模式在城市中是集中成片，还是分散出现？

---

## 28. 新 Leiden partition 和旧 Ward partition 一样吗？

不一样，但保留了一部分共同结构。

旧 64 类与新 39 类：

\[
NMI=0.557,\qquad ARI=0.119.
\]

旧 32 类与新 35 类：

\[
NMI=0.497,\qquad ARI=0.149.
\]

NMI 中等，说明两种方法捕捉到一部分共有结构。ARI 较低，说明具体哪些 dimensions 被分在一起发生了明显变化。

因此，新方法不是简单地给旧 32/64 类换名字，而是用 graph-community formulation 重新组织了 latent dimensions。

---

## 29. 为什么城市结果较相似，但 PPMI 几乎不同？

新旧 city composition similarity matrices 的相关性为：

\[
\rho=0.667.
\]

空间 homogeneity 和 continuity 排名的相关性约为：

\[
0.65\sim0.66.
\]

说明宏观城市关系保留得比较好。

但 dimension-lifted global PPMI correlation 只有：

\[
\rho=0.009.
\]

几乎没有一致性。

原因是 PPMI 对类别边界非常敏感。一个旧类别被拆开或几个旧类别被重新组合时：

- 城市总体比例可能变化不大；
- 具体哪些类别在同一张图里出现会显著变化。

因此相对稳健的是：

- 城市宏观组成；
- 城市空间排序。

相对不稳健的是：

- 具体哪一条 community pair 是强共现边。

---

## 30. 这个实验最终支持什么？

### 30.1 512 个 dimensions 不是无结构的

它们能够根据 Encoder、Decoder 和 Spatial relationships 形成有组织的 graph communities。

### 30.2 存在一个稳定的视觉组织尺度

实验在 γ=2.65–2.75 发现稳定 plateau，主分区为：

\[
K=35.
\]

该数量不是预先指定的。

### 30.3 城市大体共享视觉 vocabulary

36/39 个 fine communities 出现在全部 30 个城市中。

城市之间的差异主要表现为：

- visual community prevalence 不同；
- community co-occurrence 不同；
- community spatial organization 不同。

这可以形成下面的研究逻辑：

```text
共享的视觉词汇
+ 不同的表达比例
+ 不同的组合关系
+ 不同的空间组织
= 不同的城市视觉环境
```

---

## 31. 当前不能证明什么？

目前不能证明：

1. 世界上恰好存在 35 或 39 种城市视觉类别；
2. K=35 和 K=39 是两个同等稳定的层级；
3. 每个 community 都对应一个明确的语义对象；
4. Qwen 自动名称就是最终 taxonomy 名称；
5. 新旧方法得到了相同的共现网络；
6. 所有小型 communities 都稳定且可复现。

最稳妥的论文表述是：

> Feature-MAE latent representation 中存在一个可由多视图关系图识别的稳定视觉社区尺度。更高 resolution 提供了探索性的细粒度划分，但尚未形成第二个同等稳定的组织尺度。

---

## 32. 最终科学解释

这次实验支持：

> The frozen Feature-MAE representation contains a structured network of related latent visual responses, from which a stable visual community scale emerges without prescribing the number of categories in advance.

进一步支持：

> Cities largely draw upon a shared latent visual vocabulary, while differing in the prevalence, co-occurrence, and spatial organization of its constituent visual communities.

用中文概括就是：

> 不同城市并不是拥有完全不同的视觉元素，而是主要共享一套底层视觉词汇。城市的差异来自这些视觉词汇出现多少、如何共同出现，以及如何在城市空间中组织。

---

## 33. 相关文件

正式实验报告：

```text
paper/GRAPH_VISUAL_VOCABULARY_EXPERIMENT_REPORT.md
```

本通俗解读：

```text
paper/GRAPH_VISUAL_VOCABULARY_EXPLAINED.md
```

核心数据：

```text
paper/data/graph_visual_vocabulary/
```

核心图片：

```text
paper/figures/graph_visual_vocabulary/
```

大型中间结果：

```text
outputs/experiments/dinov3_multicity/graph_visual_vocabulary/
```
