# Spatial View Ablation：E+D 与 E+D+P Visual Graph 对比

## 1. 实验目的

本消融实验回答：

> 在 Urban Visual Vocabulary 的 multiview graph 中去掉 spatial activation profile，只使用 encoder direction 和 decoder direction，会对 graph structure、Leiden community number 和 resolution stability 产生什么影响？

正式主实验使用：

\[
W_{EDP}=\frac{\bar W_E+\bar W_D+\bar W_P}{3}.
\]

消融实验改为：

\[
W_{ED}=\frac{\bar W_E+\bar W_D}{2}.
\]

其中：

- \(E\)：Feature-MAE encoder direction；
- \(D\)：linearized decoder direction；
- \(P\)：基于完整四方向全景 `14×56` positions 的 Top-1 spatial activation profile。

消融实验没有改变任何 E、D graph，也没有重新训练 DINOv3 或 Feature-MAE。所有计算均基于正式实验已经冻结的 512 个 latent dimensions 和 local-scaled kNN graphs。

---

## 2. 输入与计算设置

两个版本共享：

```text
nodes: 512 Feature-MAE latent dimensions
k-neighbours: 30
kNN construction: symmetric union
local scaling: enabled
per-view nonzero-weight normalization: enabled
Leiden partition: RBConfigurationVertexPartition
seeds per gamma: 20
gamma range: 0.05–3.00
gamma step: 0.05
```

正式 E+D+P 结果使用既有输出。E+D 消融重新运行完整 60-point resolution sweep，每个 gamma 使用 seeds 0–19，共 1,200 次 Leiden partitions。

Plateau 判断规则与正式实验完全相同：

1. 连续至少3个 gamma points；
2. 排除 K=1 的平凡 partition；
3. mean seed NMI 不低于阈值；
4. adjacent-resolution NMI 不低于阈值；
5. 区间内 `max(K)-min(K)≤2`；
6. 阈值依次尝试 0.90、0.85、0.80。

---

## 3. Graph structure 对比

| Graph | Edges | Density | Components | Degree min | Degree mean | Degree max |
|---|---:|---:|---:|---:|---:|---:|
| Encoder \(G_E\) | 8,515 | 0.0651 | 1 | 30 | 33.26 | 52 |
| Decoder \(G_D\) | 8,474 | 0.0648 | 1 | 30 | 33.10 | 46 |
| Spatial \(G_P\) | 9,627 | 0.0736 | 12 | 0 | 37.61 | 78 |
| E+D | 16,404 | 0.1254 | 1 | 54 | 64.08 | 85 |
| E+D+P | 24,673 | 0.1886 | 1 | 58 | 96.38 | 133 |

去掉 spatial view 后：

- edges 从 24,673 降至 16,404；
- 减少 8,269 条边，即下降约 33.5%；
- graph density 从 0.1886 降至 0.1254；
- mean degree 从 96.38 降至 64.08。

E+D graph 仍然是 connected graph，因此后续碎片化不是由 disconnected components 直接造成的，而是因为局部网络中缺少 spatial view 提供的大量跨表征连接。

三个 view 的 edge support 统计为：

| View support | Edges |
|---:|---:|
| 只存在于一个 view | 22,790 |
| 存在于两个 views | 1,823 |
| 同时存在于三个 views | 60 |

这说明 E、D、P 提供的近邻关系高度互补，而不是三份近似重复的信息。

Encoder 和 Decoder graph 的 union 包含 16,404 条边，而两者独立边数之和为 16,989，因此只有约585条边同时被 E 和 D 支持。Spatial graph 与 E+D union 相交的边约为1,358条，并增加约8,269条独特关系。

---

## 4. 相同 gamma 下的直接对比

### 4.1 γ=2.70

这是正式 E+D+P 主尺度使用的 gamma。

| 指标 | E+D+P | E+D |
|---|---:|---:|
| Representative K | 35 | 77 |
| K across 20 seeds | 稳定区间约35–37 | 71–78 |
| Mean seed NMI | 约0.856 | 0.704 |
| Mean seed ARI | — | 0.240 |

E+D 在同一 gamma 下产生约两倍数量的 communities，表现出明显的过度碎片化。

E+D representative partition 与正式 E+D+P K=35 partition 的一致性为：

```text
NMI = 0.421
ARI = 0.0038
```

NMI 表明两种分区仍共享少量宏观信息，但接近零的 ARI 表明具体 dimension pairs 的共同归属几乎完全改变。

### 4.2 γ=2.85

这是正式实验 exploratory fine partition 使用的 gamma。

| 指标 | E+D+P | E+D |
|---|---:|---:|
| Representative K | 39 | 82 |
| K across 20 seeds | — | 76–85 |
| Mean seed NMI | — | 0.722 |
| Mean seed ARI | — | 0.252 |

E+D partition 与正式 E+D+P K=39 partition 的一致性为：

```text
NMI = 0.453
ARI = 0.0054
```

因此，去掉 P 并不是让现有 communities 发生少量边界变化，而是产生一套显著不同且更细碎的节点组织。

---

## 5. E+D 完整 resolution sweep

E+D graph 在不同 gamma 下的代表性结果为：

| Gamma | Communities K | Mean seed NMI | Mean seed ARI |
|---:|---:|---:|---:|
| 0.50 | 1 | 1.000 | 1.000 |
| 1.00 | 7 | 0.177 | 0.123 |
| 1.50 | 31 | 0.419 | 0.128 |
| 2.00 | 50 | 0.576 | 0.166 |
| 2.50 | 70 | 0.672 | 0.216 |
| 3.00 | 84 | 0.733 | 0.261 |

低 gamma 下的 K=1 是平凡稳定 partition，不代表有意义的视觉组织尺度。

当 gamma 从1.0增加到3.0时，E+D graph 的 community number 从7持续增长到84。虽然高 gamma 下的 seed NMI 逐渐提高，但没有形成同时满足以下条件的连续区间：

- seed NMI 达到稳定阈值；
- adjacent-resolution NMI 达到稳定阈值；
- K 的变化范围不超过2。

按照与主实验相同的阈值顺序：

```text
0.90 → no non-trivial plateau
0.85 → no non-trivial plateau
0.80 → no non-trivial plateau
```

因此，E+D 在扫描范围 `0.05–3.00` 内没有识别出符合预注册规则的非平凡 stable plateau。

---

## 6. 将 E+D 调整到约35类是否可行？

如果只要求 community number 接近正式 K=35，而不要求稳定性，E+D 在：

```text
gamma = 1.70
K = 36
```

时最接近目标。

但其稳定性较弱：

```text
mean seed NMI = 0.500
mean seed ARI = 0.149
adjacent-resolution NMI = 0.560
```

并且 E+D K=36 与正式 E+D+P K=35 的一致性较低：

```text
NMI = 0.309
ARI = 0.0082
```

这表明单纯通过降低 gamma 把 community number 调回35，并不能恢复 E+D+P 所发现的稳定结构。

为了获得接近35类而人为选择 γ=1.70，也会违反本研究“不以目标 community number 反向调节 gamma”的原则。

---

## 7. 为什么去掉 Spatial view 会导致碎片化？

Encoder 和 Decoder views 描述的是参数空间中的表征方向：

- E 表示 latent dimension 喜欢读取什么 DINO information；
- D 表示 latent dimension 倾向于重建什么 DINO direction。

Spatial view 提供的是另一类关系：

> 两个 dimensions 是否经常在完整全景的相似空间位置成为 winner？

例如多个 dimensions 可能分别响应不同类型的道路纹理或边缘，但都集中在全景下部；多个 dimensions 可能具有不同的外观方向，但都与天空、建筑上缘或道路—建筑交界相关。

只使用 E+D 时，这些 dimensions 因为参数方向不同而容易分裂成多个小社区。加入 P 后，具有相似全景空间角色的 dimensions 得到额外连接，形成更大的稳定视觉组织。

因此，在当前 graph formulation 中，spatial view 起到了：

1. 补充 E/D 中缺失的关系；
2. 连接 appearance direction 不完全相同但空间角色相似的 dimensions；
3. 抑制高 gamma 下的过度碎片化；
4. 提高跨 seed 和跨 resolution 稳定性。

它可以被理解为一种基于全景空间行为的结构约束或 graph regularization view。

---

## 8. 去掉 Spatial view 的潜在优点

虽然 E+D 的稳定性明显较弱，但它仍有两个理论上的优点：

1. 减少对固定地平线位置、拍摄构图和全景方向的依赖；
2. 得到更偏向 appearance/representation、相对位置不敏感的 grouping。

如果研究问题是：

> 不考虑视觉响应出现在什么位置，只分析 latent dimensions 在参数空间中表达了什么视觉内容？

那么 E+D 可以作为合理的对照模型。

但是本研究的目标是发现同时包含视觉表征和空间组织信息的 urban visual patterns。在这一目标下，P 是具有理论意义的核心 view，而不是应当去除的 nuisance variable。

---

## 9. 当前消融能够支持的结论

当前结果支持：

> Removing the spatial activation view substantially sparsified the visual-relation graph and produced fragmented, seed-sensitive Leiden partitions. No non-trivial stable resolution plateau was identified for the E+D graph under the preregistered stability criteria.

进一步可以写：

> The spatial view contributed complementary edges that linked latent dimensions with similar panorama-level spatial roles, thereby stabilizing the emergent community structure beyond what could be recovered from encoder and decoder directions alone.

中文概括：

> Spatial activation profile 不是冗余输入。它提供了 E、D 参数方向中缺少的大量互补关系，是当前稳定 Urban Visual Vocabulary 形成的重要条件。

---

## 10. 当前消融不能证明的内容

本次诊断尚未为 E+D partitions 重新运行：

- community evidence sheets；
- independent DINO visual coherence；
- Qwen post-hoc semantic interpretation；
- city composition；
- PPMI co-occurrence；
- within-city spatial organization。

因此，目前不能直接断言：

- E+D communities 的视觉语义一定更差；
- E+D 的 image-space coherence 一定低于 E+D+P；
- E+D 下游城市结果一定完全失效。

当前能够确定的是：

1. E+D graph 更稀疏；
2. 相同 gamma 下 communities 明显增多；
3. 调整 gamma 后仍无法找到满足规则的 stable plateau；
4. E+D partition 与正式 E+D+P partition 差异很大。

如果需要把该消融作为正式 supplementary experiment，下一步应冻结一个透明的 E+D exploratory scale，再补充 image-space coherence 和 downstream stability，而不是只比较 community number。

---

## 11. 建议

### 主实验

继续使用：

\[
E+D+P.
\]

原因是它具有：

- 唯一明确的非平凡 stable plateau；
- 更高的跨 seed stability；
- 更高的 adjacent-resolution stability；
- 更合理的 community granularity；
- 已通过 independent image-space coherence validation；
- 与城市视觉空间组织问题直接对应的理论含义。

### Supplementary ablation

将 E+D 作为消融对照，重点报告：

- graph edge/density reduction；
- same-gamma fragmentation；
- lack of non-trivial stable plateau；
- low agreement with the E+D+P partition。

这会加强主方法的论证：稳定的 visual vocabulary 不是仅由 Feature-MAE 参数方向自动产生的，panorama-level spatial behavior 提供了关键的互补结构。

---

## 12. 核心结论

去掉 spatial view 后，E+D graph 仍然连通，但会从一个具有稳定 K≈35 组织尺度的图，变成一个随 resolution 快速分裂、缺少非平凡稳定 plateau 的图。

因此，当前实验最合理的判断是：

> Spatial view 对主实验不是可有可无的辅助项，而是形成稳定 multiview visual communities 的关键组成部分。E+D 更适合作为 supplementary ablation，而不适合作为当前 Urban Visual Vocabulary 的主定义。
