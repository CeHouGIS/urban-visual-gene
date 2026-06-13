# 基于图预测的城市最小视觉空间单元识别技术路线

## 1. 研究目标

本研究旨在基于街景图像识别城市中的**最小视觉空间单元**（Minimum Visual-Spatial Unit, MVSU），即内部视觉状态保持相对一致、边界处发生显著视觉突变的最小连续城市空间片段。

## 2. 研究意义

该方法可以避免预设固定格网、街区或街道边界，从街景视觉特征本身出发，数据驱动地发现城市空间中自然涌现的最小视觉组织单元。

---

## 3. 总体技术路线

整体框架分为两个主要阶段：

1. **街景图像特征提取**：将每个街景点编码为 scene-level visual feature，并进一步转换为全局可比较的 scene state。
2. **Graph 上的视觉状态预测**：将街景点组织到道路约束 graph 上，训练 AI 模型预测相邻节点的视觉状态，并利用预测失败产生的 surprise score 识别视觉突变边界，进而提取最小视觉空间单元。

整体流程如下：

```text
Street-view images / panoramas
        ↓
Scene-level feature extraction
        ↓
Pano-level visual embeddings
        ↓
Global scene tokenizer
        ↓
Global scene states
        ↓
Road-constrained street-view graph
        ↓
Graph-augmented scene prediction
        ↓
Prediction surprise
        ↓
Visual transition boundaries
        ↓
Minimum visual-spatial units
```

---

## 4. Stage 1：Scene-level Feature Extraction

### 4.1 输入数据

输入数据包括：

```text
street-view images / panoramas
pano_id
city
lat
lon
heading
image_path
road_id / matched street segment
road network / logical street geometry
```

如果一个 panorama 包含多个方向图像，例如前后左右四个方向，则每个方向图像先单独提取视觉特征，之后再聚合为 pano-level feature。

---

### 4.2 图像特征提取

使用视觉基础模型提取每张街景图像的视觉特征：

\[
z_{i,h} = f_\theta(I_{i,h})
\]

其中：

- \(I_{i,h}\)：第 \(i\) 个 pano 在 heading \(h\) 下的街景图像；
- \(f_\theta\)：视觉基础模型，例如 DINOv2、DINOv3、CLIP 或 SigLIP；
- \(z_{i,h}\)：该方向图像的 scene-level embedding。

---

### 4.3 Pano-level Feature Pooling

如果一个 pano 有多个 heading，则将多个方向的图像特征聚合为一个 pano-level feature：

\[
z_i = Pool(z_{i,0}, z_{i,90}, z_{i,180}, z_{i,270})
\]

第一版可以使用 mean pooling：

\[
z_i = \frac{1}{H}\sum_h z_{i,h}
\]

随后进行 L2 normalization：

\[
\hat{z}_i = \frac{z_i}{\|z_i\|_2}
\]

这样，每个街景点被表示为一个 scene-level visual feature。

---

### 4.4 Global Scene Tokenizer

为了使不同城市之间的街景视觉状态具有可比性，需要训练一个 global scene tokenizer：

\[
\hat{z}_i \rightarrow s_i
\]

其中：

- \(\hat{z}_i\)：pano-level scene embedding；
- \(s_i\)：该 pano 所属的 global scene state；
- \(s_i \in \{1,2,...,K\}\)。

建议主实验采用：

```text
tokenizer: spherical K-means
K = 128
robustness: K = 64 / 256
sampling strategy: city-balanced sampling
```

训练完成后，每个 pano 都会被分配到一个跨城市共享的 scene state。

---

### 4.5 Stage 1 输出

第一阶段输出一个 node table：

```text
pano_id
city
lat
lon
road_id
scene_embedding
scene_state
scene_confidence
```

示例：

```text
paris_001 | Paris | road_12 | S07 | 0.82
paris_002 | Paris | road_12 | S07 | 0.79
paris_003 | Paris | road_12 | S19 | 0.76
```

---

## 5. Stage 2：Street-view Graph Construction

### 5.1 节点定义

每个 pano 被定义为 graph node：

\[
v_i = (\hat{z}_i, s_i, lat_i, lon_i, road_i)
\]

其中：

- \(\hat{z}_i\)：scene embedding；
- \(s_i\)：scene state；
- \(lat_i, lon_i\)：空间坐标；
- \(road_i\)：匹配到的道路或 logical street。

---

### 5.2 边定义

构建道路约束下的 street-view graph：

\[
G = (V, E)
\]

其中：

- \(V\)：street-view pano nodes；
- \(E\)：节点之间的空间或拓扑相邻关系。

边类型包括：

```text
same_road_next      同一道路上的相邻 pano
intersection_turn   路口处相邻道路之间的 pano
spatial_near        空间近邻弱边，可选
```

每条边保存以下属性：

```text
src_pano_id
dst_pano_id
edge_type
road_distance_m
euclidean_distance_m
bearing_diff
visual_distance
edge_confidence
```

---

## 6. Stage 3：Graph-augmented Scene Prediction

### 6.1 预测任务

对于 graph 中每条有方向边 \(e_{ij}\)，训练模型预测目标节点 \(j\) 的 scene state：

\[
P(s_j \mid sequence_i, graph_i, e_{ij})
\]

其中：

- \(s_j\)：目标节点的真实 scene state；
- \(sequence_i\)：节点 \(i\) 在同一道路上的局部序列上下文；
- \(graph_i\)：节点 \(i\) 的 graph neighborhood；
- \(e_{ij}\)：从节点 \(i\) 到节点 \(j\) 的边特征。

---

### 6.2 Graph-augmented 模型结构

模型由三部分组成：

```text
Scene state embedding
        ↓
Sequence encoder
        ↓
Graph neighbor encoder
        ↓
Fusion layer
        ↓
Prediction head
        ↓
P(target scene state)
```

形式化表达为：

\[
h_i^{seq} = SeqEncoder(s_{i-L},...,s_i)
\]

\[
h_i^{graph} = GraphEncoder(\mathcal{N}(i))
\]

\[
h_i = Fuse(h_i^{seq}, h_i^{graph}, e_{ij})
\]

\[
\hat{p}_{ij} = Softmax(W h_i)
\]

其中：

- sequence encoder 用于捕捉同一道路上的视觉连续性；
- graph neighbor encoder 用于引入路口、相邻街道和近邻节点的上下文；
- fusion layer 将序列上下文、graph 上下文和边特征融合；
- prediction head 输出目标节点属于各个 scene state 的概率。

---

### 6.3 训练目标

主要训练目标是预测真实相邻节点的 scene state：

\[
\mathcal{L}_{next} = -\log P(s_j^{true} \mid sequence_i, graph_i, e_{ij})
\]

可选加入 masked graph prediction：

\[
\mathcal{L}_{mask} = -\sum_{i \in M}\log P(s_i \mid \mathcal{N}(i))
\]

总损失函数为：

\[
\mathcal{L} = \mathcal{L}_{next} + \lambda \mathcal{L}_{mask}
\]

---

## 7. Stage 4：Prediction Surprise Calculation

训练完成后，对每条 graph edge 计算 prediction surprise：

\[
Surprise_{ij} = -\log P(s_j^{true} \mid sequence_i, graph_i, e_{ij})
\]

其中：

- surprise 低：说明该视觉转移在模型看来是正常的；
- surprise 高：说明该视觉转移在模型看来是意外的，可能对应视觉突变边界。

输出 edge-level prediction table：

```text
src_pano_id
dst_pano_id
true_scene_state
predicted_scene_state
prediction_probability
surprise
edge_type
edge_confidence
```

---

## 8. Stage 5：Visual Boundary Detection

基于每个城市内部的 surprise 分布设定自适应阈值：

\[
\tau_c = Q_{0.90}^{city}(Surprise)
\]

或：

\[
\tau_c = Q_{0.95}^{city}(Surprise)
\]

定义高 surprise 边为视觉边界：

\[
E_{boundary} = \{e_{ij}: Surprise_{ij} > \tau_c\}
\]

输出 visual boundary table：

```text
boundary_edge_id
src_pano_id
dst_pano_id
surprise
visual_distance
edge_type
edge_confidence
```

---

## 9. Stage 6：Minimum Visual-Spatial Unit Extraction

从原始 graph 中移除 high-surprise boundary edges：

\[
G' = (V, E - E_{boundary})
\]

然后在剩余 graph 中提取 connected components：

\[
U_1, U_2, ..., U_n = ConnectedComponents(G')
\]

每个 connected component 是一个候选 minimum visual-spatial unit。

---

## 10. Unit Filtering and Characterization

对每个 candidate unit 计算以下指标。

### 10.1 内部视觉一致性

\[
IntraVar(U) =
\frac{1}{|U|}
\sum_{i \in U}
(1-\cos(z_i,\bar{z}_U))
\]

---

### 10.2 Scene State Entropy

\[
H(U) = -\sum_k p_U(k)\log p_U(k)
\]

---

### 10.3 边界强度

\[
BoundarySurprise(U) =
mean(Surprise_{boundary})
\]

---

### 10.4 空间支持

记录：

```text
n_panos
road_length_m
geometry
```

最终输出 unit table：

```text
unit_id
city
geometry
n_panos
road_length_m
dominant_scene_state
scene_entropy
intra_visual_variance
mean_boundary_surprise
unit_confidence
```

---

## 11. Baseline Comparison

主方法为：

```text
Graph-augmented prediction surprise
```

对比 baseline 包括：

```text
fixed grid
full street segment
cosine distance threshold
local MAD threshold
change-point detection
Markov transition model
sequence-only prediction
graph-only prediction
```

比较指标包括：

```text
intra-unit visual variance
boundary visual contrast
scene entropy
unit size distribution
boundary stability
cross-city transfer performance
```

---

## 12. 最终输出文件

```text
1. pano_scene_features.parquet
2. global_scene_tokenizer.pkl / scene_centers.npy
3. street_view_graph_nodes.parquet
4. street_view_graph_edges.parquet
5. edge_prediction_surprise.parquet
6. visual_boundaries.geojson
7. minimum_visual_units.geojson
8. unit_statistics.csv
9. baseline_comparison.csv
```

---

## 13. 方法总结

该技术路线首先将街景图像编码为 scene-level visual feature，并通过 global scene tokenizer 转换为跨城市可比较的 scene state；随后将 pano 点组织为道路约束 graph，训练 graph-augmented prediction model 预测相邻节点的视觉状态；最后利用 prediction surprise 识别视觉突变边界，并将高 surprise 边界分割出的低 surprise 连通区域定义为城市最小视觉空间单元。
