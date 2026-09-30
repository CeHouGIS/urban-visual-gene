# 基于路网的城市景观基础特征与最小空间单元识别系统 — REQUIREMENT.md

> **版本**：v2.0-road-basis  
> **工程根目录**：`/workplace/urban_visual_gene/`  
> **实现对象**：Claude Code / Codex 代码实现  
> **核心变化**：本版不再采用“next feature prediction / graph surprise”作为主路线，而是采用 **road-network-organized feature basis learning**。  
> **核心公式**：
>
> \[
> Y = Enc(I)
> \]
>
> \[
> Z_{road} = \mathcal{P}_{G_{road}}(Y)
> \]
>
> \[
> Z_{road} \approx A X
> \]
>
> \[
> U = \mathcal{R}(A, G_{road})
> \]

---

## 0. 研究目标与核心定义

### 0.1 研究目标

本系统目标是从街景图像中识别城市中的**最小路网景观空间单元**（Minimum Road-based Landscape Unit, **MRLU**）。

MRLU 指道路网络上的连续子图，其内部具有稳定一致的城市景观基础特征组合，其边界处存在显著的景观激活变化。

注意：

- 街景图像不是最小空间单元，只是观测样本。
- 固定网格不是主方法，只能作为 baseline 或上层汇总。
- `X` 不是人工写死的“开敞度/绿荫度/商业度”等类型。
- `X` 不是直接由孤立街景点构成的图像基底。
- `X` 应该是通过道路网络组织后的 `Z_road` 学习出来的**路网景观基础特征**。

---

## 0.2 核心变量

| 符号 | 工程对象 | 含义 |
|---|---|---|
| `I` | street-view images | 街景图像输入 |
| `Y` | `pano_features.parquet` | 点级街景视觉特征，来自 DINO/CLIP/SigLIP 等视觉模型 |
| `G_road` | `road_graph_edges.parquet` / OSM road network | 道路网络图 |
| `P_road` | road projection / aggregation operator | 把点级街景特征组织到路网上的算子 |
| `Z_road` | `road_context_features.parquet` | 基于路网构建的路段/路网节点景观表征 |
| `X` | `models/road_landscape_basis.npy` | 从 `Z_road` 中学习出的全球共享路网景观基础特征 |
| `A` | `road_basis_activation.parquet` | 每个路网节点/路段对 `X` 的激活权重 |
| `U` | `minimum_road_landscape_units.geojson` | 最终识别出的最小路网景观空间单元 |

---

## 0.3 总体流程

```text
Street-view images / panoramas
        ↓ Stage 1
Point-level visual feature extraction: Y = Enc(I)
        ↓ Stage 2
Map-match panoramas to road network
        ↓ Stage 3
Build road-network-organized landscape features: Z_road = P_Groad(Y)
        ↓ Stage 4
Learn shared road landscape bases: Z_road ≈ A X
        ↓ Stage 5
Construct road-based basis activation field A on G_road
        ↓ Stage 6
Road-constrained spatial unit extraction: U = R(A, G_road)
        ↓ Stage 7
Characterization, visualization, baseline comparison, tests
```

---

## 0.4 与旧版路线的区别

旧版路线可能是：

```text
街景点 feature → tokenizer → graph prediction → surprise → boundary → units
```

新版路线是：

```text
街景点 feature → 路网组织后的 feature Z_road → basis decomposition Z_road ≈ A X → road graph segmentation → units
```

最重要的变化：

> `X` 必须从路网组织后的 `Z_road` 中学习，而不是直接从孤立 pano feature `Y` 中学习。

---

## 1. 目录结构要求

在 `/workplace/urban_visual_gene/` 下实现以下结构：

```text
urban_visual_gene/
├── scripts/
│   ├── stage1_extract_pano_features.py
│   ├── stage2_map_match_road.py
│   ├── stage3_build_road_context_features.py
│   ├── stage4_train_road_basis_model.py
│   ├── stage5_infer_road_basis_activation.py
│   ├── stage6_extract_road_units.py
│   ├── stage7_evaluate_and_baselines.py
│   ├── road_basis_model.py
│   ├── road_graph_utils.py
│   └── io_utils.py
├── tests/
│   ├── conftest.py
│   ├── test_stage1_features.py
│   ├── test_stage2_map_match.py
│   ├── test_stage3_road_context.py
│   ├── test_stage4_basis_model.py
│   ├── test_stage5_activation.py
│   ├── test_stage6_road_units.py
│   ├── test_stage7_baselines.py
│   └── test_pipeline_e2e.py
├── models/
│   ├── road_basis_model.pt
│   ├── road_landscape_basis.npy
│   └── training_report.json
└── outputs/
    └── <country>/<city>/
        ├── pano_features.parquet
        ├── road_matched_panos.parquet
        ├── road_graph_nodes.parquet
        ├── road_graph_edges.parquet
        ├── road_context_features.parquet
        ├── road_basis_activation.parquet
        ├── road_activation_boundaries.geojson
        ├── minimum_road_landscape_units.geojson
        ├── unit_statistics.csv
        └── stage_reports/
```

---

## 2. 数据契约

### 2.1 输入：`panoramas.parquet`

| 列名 | 类型 | 说明 |
|---|---|---|
| `pano_id` | str | 全局唯一 panorama ID |
| `city` | str | 城市名 |
| `country` | str | 国家名，可选但推荐 |
| `lat` | float64 | WGS84 纬度 |
| `lon` | float64 | WGS84 经度 |
| `heading` | float64 | 拍摄朝向，0–360 |
| `image_path` | str | 图像路径，绝对或相对 |
| `timestamp` | str/int | 可选，拍摄时间 |

### 2.2 输入：`road_network_edges.geojson` 或 `logical_streets.geojson`

至少包含：

| 字段 | 类型 | 说明 |
|---|---|---|
| `road_id` | str/int | 道路边或逻辑街道 ID |
| `geometry` | LineString | 道路几何 |
| `road_class` | str | 可选，OSM highway class |
| `length_m` | float | 可选，可由 geometry 计算 |

如果项目已有 `logical_streets.geojson`，优先使用它作为 road graph 基础。

---

## 2.3 Stage 1 输出：`pano_features.parquet`

| 列名 | 类型 | 约束 |
|---|---|---|
| `pano_id` | str | 唯一 |
| `city` | str | 非空 |
| `lat` | float64 | [-90, 90] |
| `lon` | float64 | [-180, 180] |
| `heading` | float64 | [0, 360] |
| `image_path` | str | 可选保留 |
| `pano_embedding` | ndarray(float32, 4D) | 四方向 concat 后 L2 norm ≈ 1.0 |
| `feature_model` | str | 如 `dinov2_vitb14` |

可接受视觉模型（D 为单方向维度）：

```text
DINOv2-ViT-B/14   D=768   推荐第一版，最终 pano_embedding = 3072 维
DINOv2-ViT-L/14   D=1024  最终 pano_embedding = 4096 维
CLIP-ViT-B/32     D=512   最终 pano_embedding = 2048 维
SigLIP-B/16       D=768   最终 pano_embedding = 3072 维
```

**多方向 pano 处理规则（重要）**：

- 对四个方向（0°、90°、180°、270°）分别用视觉模型提取 embedding：`f_0, f_90, f_180, f_270`，每个形状 `(D,)`。
- **concat**（不是 mean pool）：`pano_embedding = concat([f_0, f_90, f_180, f_270])`，形状 `(4D,)`。
- 最后对整体做 L2 normalization：`pano_embedding /= ||pano_embedding||_2`。
- 如果某方向图像缺失，用零向量填充该方向，并在 report 中统计缺失比例。

---

## 2.4 Stage 2 输出：`road_matched_panos.parquet`

| 列名 | 类型 | 约束 |
|---|---|---|
| `pano_id` | str | 存在于 `pano_features` |
| `city` | str | 非空 |
| `lat` | float64 | — |
| `lon` | float64 | — |
| `matched_road_id` | str/int | 非空 |
| `road_distance_m` | float32 | 点到道路距离，≥0 |
| `chainage_m` | float32 | 沿 road edge 起点距离，≥0 |
| `match_confidence` | float32 | (0, 1] |
| `pano_embedding` | ndarray(float32, D) | 保留 |

要求：

- 默认最大 map-match 距离：`30m`。
- 超过阈值的 pano 标记为 unmatched，不参与主流程。
- 输出 matched ratio 到 report。

---

## 2.5 Stage 2 输出：`road_graph_nodes.parquet` 和 `road_graph_edges.parquet`

### `road_graph_nodes.parquet`

| 列名 | 类型 | 说明 |
|---|---|---|
| `road_node_id` | str | 路网节点 ID，可为道路采样点或 road edge ID |
| `road_id` | str/int | 所属道路 |
| `lat` | float64 | 节点纬度 |
| `lon` | float64 | 节点经度 |
| `chainage_m` | float32 | 沿路距离 |
| `geometry` | Point/WKB | 可选 |

### `road_graph_edges.parquet`

| 列名 | 类型 | 说明 |
|---|---|---|
| `src_node_id` | str | 起点 node |
| `dst_node_id` | str | 终点 node |
| `edge_type` | str | `same_road_next` / `intersection_connect` / `road_adjacent` |
| `network_distance_m` | float32 | 路网距离 |
| `bearing_diff` | float32 | 可选 |
| `edge_confidence` | float32 | (0, 1] |

要求：

- road graph 必须是图结构，不是普通二维 grid。
- 主流程以 `road_graph_nodes` 为基本空间载体。
- 如果以 road edge 为 node，也可以，但需要在 report 中注明 node 定义。

---

## 2.6 Stage 3 输出：`road_context_features.parquet`

这是新版方法最关键的中间产物。`Z_road` 必须由路网组织后的街景点 feature 构建。

| 列名 | 类型 | 说明 |
|---|---|---|
| `road_node_id` | str | 路网节点 ID |
| `city` | str | 城市 |
| `road_id` | str/int | 所属 road |
| `lat` | float64 | — |
| `lon` | float64 | — |
| `n_panos` | int | 聚合到该 node 的 pano 数（0 表示插值填充） |
| `total_weight` | float32 | 聚合权重之和，0 表示无 pano 覆盖 |
| `context_radius_m` | float32 | 路网邻域半径 |
| `road_context_embedding` | ndarray(float32, 4D) | `Z_road`，维度同 pano_embedding，L2 norm ≈ 1.0 |
| `aggregation_method` | str | `multi_road_decay`（主方法）/ `graph_smooth`（可选后处理） |

---

## 2.7 Stage 4/5 输出：`road_basis_activation.parquet`

| 列名 | 类型 | 说明 |
|---|---|---|
| `road_node_id` | str | 路网节点 ID |
| `city` | str | 城市 |
| `road_id` | str/int | 所属道路 |
| `lat` | float64 | — |
| `lon` | float64 | — |
| `a_000` ... `a_{K-1}` | float32 | 路网节点对 basis `X` 的激活 |
| `reconstruction_error` | float32 | `||z - aX||` 或 cosine error |
| `activation_entropy` | float32 | 激活熵 |
| `active_basis_count` | int | 激活超过阈值的 basis 数 |

要求：

- `A >= 0`。
- `A` 建议稀疏。
- `A` 是路网上的 activation field。

---

## 2.8 Stage 6 输出：`road_activation_boundaries.geojson`

每个 feature 是道路网络上的候选景观突变边界。

| 属性 | 类型 | 说明 |
|---|---|---|
| `boundary_id` | str | 唯一 ID |
| `src_node_id` | str | 边起点 |
| `dst_node_id` | str | 边终点 |
| `activation_distance` | float32 | `1 - cosine(a_i, a_j)` 或 L2 |
| `boundary_score` | float32 | 标准化边界分数 |
| `edge_type` | str | road graph edge type |
| `network_distance_m` | float32 | 路网距离 |

geometry 应为 LineString，沿 road graph edge 生成。

---

## 2.9 Stage 6 输出：`minimum_road_landscape_units.geojson`

每个 feature 是路网 constrained unit，可采用 MultiLineString 作为主 geometry。

```json
{
  "type": "Feature",
  "geometry": {"type": "MultiLineString", "coordinates": [...]},
  "properties": {
    "unit_id": "str",
    "city": "str",
    "n_road_nodes": 18,
    "n_panos": 52,
    "road_length_m": 820.5,
    "mean_within_activation_variance": 0.07,
    "mean_boundary_contrast": 0.42,
    "dominant_basis_id": 12,
    "activation_entropy": 1.73,
    "unit_confidence": 0.81
  }
}
```

### `unit_statistics.csv`

同 GeoJSON properties，另加：

| 列名 | 类型 | 说明 |
|---|---|---|
| `centroid_lat` | float64 | 单元质心纬度 |
| `centroid_lon` | float64 | 单元质心经度 |
| `top_basis_ids` | str | 如 `12|4|37` |
| `top_basis_weights` | str | 如 `0.31|0.22|0.11` |
| `stability_score` | float32 | 多 seed / 多阈值稳定性，可后续实现 |

---

## 3. 功能需求

## Stage 1：点级街景特征提取 `Y`

### F1.1

实现 `scripts/stage1_extract_pano_features.py`，支持从 `panoramas.parquet` 读取街景图像路径，提取视觉 embedding。

### F1.2

四方向特征提取与 concat：

```python
# 伪代码
headings = [0, 90, 180, 270]
feats = []
for h in headings:
    img = load_image(pano_id, heading=h)   # 缺失则返回 zeros
    f = model.encode(img)                  # shape (D,)
    feats.append(f)
pano_embedding = np.concatenate(feats)     # shape (4D,)
pano_embedding /= np.linalg.norm(pano_embedding)  # L2 norm
```

所有输出 embedding 必须 L2 normalize，验证：

```python
np.allclose(np.linalg.norm(emb, axis=1), 1.0, atol=1e-5)
```

### F1.3

支持 `--model` 参数：

```bash
python scripts/stage1_extract_pano_features.py \
  --input data/panoramas.parquet \
  --output outputs/<country>/<city>/pano_features.parquet \
  --model dinov2_vitb14 \
  --batch-size 64
```

### F1.4

如果用户已有 feature 文件，则允许 `--skip-extraction --feature-path ...` 直接进入 Stage 2。

---

## Stage 2：街景点 map-match 到道路网络

### F2.1

实现 `scripts/stage2_map_match_road.py`。

输入：

```text
pano_features.parquet
road_network_edges.geojson 或 logical_streets.geojson
```

输出：

```text
road_matched_panos.parquet
road_graph_nodes.parquet
road_graph_edges.parquet
```

### F2.2

map-match 规则：

- 对每个 pano 找最近 road geometry。
- 默认最大距离 `30m`。
- 计算 `matched_road_id`、`road_distance_m`、`chainage_m`。
- unmatched pano 不参与 Stage 3，但保留 report。

### F2.3

路网节点构建第一版采用**沿道路重采样节点**：

- 对每条 road edge / logical street 按固定间距采样节点。
- 默认 node spacing = `25m`。
- 每个 road node 收集其附近的 matched pano。

可配置：

```bash
--node-spacing-m 25
--max-match-distance-m 30
```

### F2.4

构建 road graph edges：

- 同一路段相邻 node 连边：`same_road_next`。
- 道路交叉口相连 node 连边：`intersection_connect`。
- 如 logical street 已有拓扑，可直接使用。

### F2.5

graph 输出应保证：

- ≥ 95% road nodes 在最大连通分量中，除非城市道路数据本身分裂。
- 每条 edge 双向存储或在算法中按无向图处理。

---

## Stage 3：构建路网组织后的景观表征 `Z_road`

### F3.1

实现 `scripts/stage3_build_road_context_features.py`。

本阶段目标：从点级街景特征 `Y` 构建路网组织后的表征 `Z_road`。

### F3.2

**核心设计：pano → 多路段距离衰减贡献**

每个 pano 点不仅匹配到最近路段，而是在一定半径内向所有路段贡献特征，权重由距离衰减决定。

#### Step 1：判断逻辑路（Logical Road 归属）

在聚合之前，先判断周围路段是否属于同一条逻辑路：

- 如果输入已有 `logical_streets.geojson`（含 `LS_id`），直接使用 `LS_id` 作为逻辑路 ID。
- 如果使用原始 OSM 路网，按 `name` + `highway` 类型聚类相同方向的平行段为同一逻辑路。
- 同一逻辑路的多条几何段作为一个整体接收来自 pano 的贡献。

#### Step 2：pano 到多路段的距离衰减贡献

对每个 pano 点 `p`，找其半径 `R=100m` 内的所有路段（按逻辑路去重后，可能是多条不同逻辑路）：

\[
w(p \to e) = \exp\!\left(-\frac{d(p,e)^2}{2\sigma^2}\right)
\]

其中 `d(p, e)` 为 pano 到路段几何的垂直距离，`σ = 30m`（默认）。

pano `p` 向路段 `e` 的贡献**不归一化**（即每条路段独立接收全部加权特征）：

\[
\text{contribution}(p \to e) = w(p \to e) \cdot y_p
\]

原因：若 pano 在十字路口同时靠近多条路，每条路都应完整地收到该视觉信息，而非被稀释。

#### Step 3：路网节点特征聚合

对每个 road node `v`（沿路每 25m 采样），聚合其所在路段以及路网邻域内的 pano 贡献：

\[
z_v = \frac{\displaystyle\sum_{p \in \mathcal{N}(v)} w(p \to \text{road}(v)) \cdot y_p}
           {\displaystyle\sum_{p \in \mathcal{N}(v)} w(p \to \text{road}(v)) + \epsilon}
\]

`𝒩(v)` 为路网距离 `context_radius_m=100m` 内所有对该 road node 所在路段有贡献的 pano。

若某 road node 半径内无 pano（贡献权重之和为 0），使用最近有特征的相邻 road node 的 `z` 插值，并标记 `n_panos=0`。

默认参数：

```text
search_radius_m   = 100    # pano 搜索半径（贡献到多路段）
kernel_sigma_m    = 30     # 距离衰减 σ
context_radius_m  = 100    # road node 聚合 pano 的路网半径
node_spacing_m    = 25     # road node 采样间距
```

#### 方法 B（可选）：`graph_smooth`

在 road graph 上对上述聚合结果做 Laplacian smoothing：

\[
Z^{t+1} = (1-\alpha)Z^0 + \alpha D^{-1}WZ^t
\]

默认：`alpha=0.5, n_iter=3`，作为可选后处理。

### F3.3

主实验第一版推荐：**pano 多路段距离衰减聚合**（Step 1–3），不做额外 graph smoothing。

### F3.4

所有 `road_context_embedding` 必须 L2 normalize。

### F3.5

输出 `road_context_features.parquet` 和 `stage_3_report.json`。

---

## Stage 4：学习路网景观基础特征 `X`

### F4.1

实现 `scripts/stage4_train_road_basis_model.py` 和 `scripts/road_basis_model.py`。

核心模型：带约束的 sparse autoencoder / dictionary learning。

输入：

```text
Z_road ∈ R^{M × D}
```

目标：

\[
Z_{road} \approx A X
\]

其中：

```text
A ∈ R^{M × K}
X ∈ R^{K × D}
```

### F4.2 模型结构

推荐第一版模型：

```text
Input z_e ∈ R^D
Encoder: MLP(D → hidden → K)
Activation: Softplus or ReLU
Latent: a_e ∈ R^K, non-negative
Decoder: Linear(K → D), decoder weight = X
Output: z_hat_e
```

重要要求：

- Decoder 必须保持 linear。
- 不要使用 deep nonlinear decoder，否则 `X` 不再是清晰的 basis。
- `X` 行向量需要单位化。

### F4.3 损失函数

第一版必须实现：

\[
\mathcal{L} =
\mathcal{L}_{recon}
+ \lambda_{sparse}\|A\|_1
+ \lambda_{spatial}\sum_{(i,j)\in E} w_{ij}\|a_i-a_j\|^2
+ \lambda_{div}\|XX^T-I\|_F^2
\]

其中：

| 项 | 含义 |
|---|---|
| `L_recon` | 重构 `Z_road`，推荐 cosine loss 或 normalized MSE |
| `L_sparse` | 让 `A` 稀疏，地点只激活少数基底 |
| `L_spatial` | 让 `A` 沿路网空间连续 |
| `L_div` | 让 `X` 非冗余，但不是强制严格正交 |

### F4.4 `A` 的城市语境约束

必须实现：

- 非负：`A >= 0`，通过 Softplus/ReLU 实现。
- 稀疏：L1 penalty。
- 路网连续：graph Laplacian smoothness over `G_road`。
- 可聚合：后续任何区域表达只能由 `A` 聚合得到，不得重新训练尺度特异 `X`。

### F4.5 `X` 的城市语境约束

必须实现：

- 从 `Z_road` 学习，而不是直接从 `Y_pano` 学习。
- 全局共享：跨城市训练时所有城市共用同一套 `X`。
- 单位化：每个 basis `x_k` L2 norm = 1。
- 去冗余：弱 diversity penalty。
- 稳定性：支持多 seed 训练，输出 basis matching report。

### F4.6 参数默认值

```text
K = 100
hidden_dim = 512
lambda_sparse = 1e-4
lambda_spatial = 1e-3
lambda_div = 1e-3
learning_rate = 1e-3
batch_size = 1024
epochs = 100
seed = 42
```

### F4.7 命令行示例

```bash
python scripts/stage4_train_road_basis_model.py \
  --input outputs/<country>/<city>/road_context_features.parquet \
  --road-edges outputs/<country>/<city>/road_graph_edges.parquet \
  --output-model models/road_basis_model.pt \
  --output-basis models/road_landscape_basis.npy \
  --K 100 \
  --lambda-sparse 1e-4 \
  --lambda-spatial 1e-3 \
  --lambda-div 1e-3
```

---

## Stage 5：推理路网 basis activation `A`

### F5.1

实现 `scripts/stage5_infer_road_basis_activation.py`。

输入：

```text
road_context_features.parquet
models/road_basis_model.pt
```

输出：

```text
road_basis_activation.parquet
```

### F5.2

输出必须包含：

- `road_node_id`
- `road_id`
- `lat`, `lon`
- `a_000` ... `a_{K-1}`
- `reconstruction_error`
- `activation_entropy`
- `active_basis_count`

### F5.3

`active_basis_count` 定义：

```python
active_basis_count = (a_i > activation_threshold).sum()
```

默认：

```text
activation_threshold = 0.01
```

如使用 normalized A，可调整为相对阈值。

---

## Stage 6：基于路网的最小空间单元提取

### F6.1

实现 `scripts/stage6_extract_road_units.py`。

核心输入：

```text
road_basis_activation.parquet
road_graph_edges.parquet
road_graph_nodes.parquet
```

核心输出：

```text
road_activation_boundaries.geojson
minimum_road_landscape_units.geojson
unit_statistics.csv
```

### F6.2 边界分数

对每条 road graph edge `(i, j)` 计算 activation distance：

\[
 d_{ij} = 1 - \cos(a_i, a_j)
\]

或可选：

\[
 d_{ij} = \|a_i-a_j\|_2
\]

默认使用 cosine distance。

### F6.3 自适应阈值

每个城市内自适应：

\[
\tau_c = Q_{0.90}(d_{ij})
\]

边界边：

\[
E_{boundary} = \{(i,j): d_{ij} > \tau_c\}
\]

参数：

```text
boundary_quantile = 0.90
```

### F6.4 提取路网单元

从 road graph 中移除 boundary edges，在剩余图上取 connected components：

```text
G' = G_road - E_boundary
U = connected_components(G')
```

每个 component 是一个候选 MRLU。

### F6.5 过滤规则

默认过滤：

```text
min_road_nodes = 3
min_road_length_m = 50
min_panos = 3
```

过滤后小碎片可选择：

- 合并到 activation 最相似的邻近 unit；或
- 标记为 `small_fragment`。

第一版允许直接丢弃，并在 report 中统计比例。

### F6.6 单元统计指标

每个 unit 计算：

| 指标 | 说明 |
|---|---|
| `mean_activation` | unit 内 `A` 均值 |
| `within_activation_variance` | unit 内 activation variance |
| `activation_entropy` | mean activation entropy |
| `dominant_basis_id` | mean activation 最大维度 |
| `mean_boundary_contrast` | 与相邻 unit 边界的平均 activation distance |
| `road_length_m` | unit 内 road graph edge 长度总和 |
| `n_road_nodes` | unit 内 road node 数 |
| `n_panos` | unit 关联 pano 数 |
| `unit_confidence` | 根据内部一致性和边界对比综合计算 |

建议：

\[
unit\_confidence = sigmoid(mean\_boundary\_contrast - within\_activation\_variance)
\]

第一版可用简单归一化实现。

---

## Stage 7：评估与 baseline

### F7.1 Baselines

实现 `scripts/stage7_evaluate_and_baselines.py`。

需要实现：

| Baseline | 说明 |
|---|---|
| `fixed_grid_100m` | 100m 固定网格，不作为主方法 |
| `fixed_grid_500m` | 500m 固定网格 |
| `road_edge_as_unit` | 每条 road edge 作为一个 unit |
| `cosine_threshold_on_Y` | 直接在 pano feature `Y` 上按相邻差异切分 |
| `cosine_threshold_on_Zroad` | 在 `Z_road` 上按相邻差异切分 |
| `no_road_context_basis` | 直接用 `Y ≈ AX` 学 basis，然后与 road-based X 对比 |

### F7.2 指标

| 指标 | 越大/小 | 说明 |
|---|---|---|
| `within_activation_variance` | 小 | unit 内部一致性 |
| `boundary_activation_contrast` | 大 | 边界清晰度 |
| `unit_size_median` | 适中 | 避免过碎或过大 |
| `fragment_ratio` | 小 | 小碎片比例 |
| `road_connectivity_valid` | True | unit 必须是 road graph connected component |
| `cross_seed_boundary_stability` | 大 | 多 seed 边界稳定性 |
| `heldout_city_reconstruction` | 小 | `X` 跨城市通用性 |

### F7.3 输出

```text
outputs/baseline_comparison.csv
outputs/<country>/<city>/evaluation_report.json
outputs/<country>/<city>/evaluation_figures/
```

---

## 4. 自动化测试规范

### 4.1 测试目标

所有测试必须能通过：

```bash
cd /workplace/urban_visual_gene
pytest tests/ -v --tb=short
```

合成测试数据要求：

- 不依赖真实图像。
- 在内存生成 synthetic road network、synthetic pano features。
- 合成城市包含 3 个已知路网景观区域。
- 每个区域内部 `Z_road` 相似，跨区域差异大。
- E2E 测试运行时间 < 60s。

---

### 4.2 `tests/conftest.py` 合成数据要求

必须构建：

```text
N_ROADS = 6
N_PANOS_PER_ROAD = 10
N_ROAD_NODES_PER_ROAD = 10
N_REGIONS = 3
D = 64
K = 16 或 32 用于测试
```

设计：

- 每 2 条 road 属于同一个 synthetic landscape region。
- 区域内 pano embedding 高相似。
- 区域之间 embedding 低相似。
- road graph 中包含跨区域连接边，用于验证 boundary detection。

Fixtures：

```python
synthetic_pano_features
synthetic_road_network
synthetic_road_matched_panos
synthetic_road_context_features
synthetic_road_graph_edges
```

---

### 4.3 Stage 测试要求

#### `test_stage1_features.py`

测试：

- 输出 schema 完整。
- embedding L2 norm ≈ 1。
- `pano_id` 唯一。
- feature dimension 正确。

#### `test_stage2_map_match.py`

测试：

- 每个 pano 成功匹配到 road。
- `road_distance_m >= 0`。
- `chainage_m >= 0`。
- road graph connected ratio 合理。
- same road nodes 按 chainage 连边。

#### `test_stage3_road_context.py`

测试：

- `road_context_features.parquet` schema 完整。
- 每个 road node 有 `road_context_embedding`。
- embedding L2 norm ≈ 1。
- 同 synthetic region 内 `Z_road` 相似度高于跨 region。
- `network_kernel` 在无 pano node 上可以通过邻域插值得到 feature。

#### `test_stage4_basis_model.py`

测试：

- `RoadBasisAutoEncoder` 可实例化。
- forward shape 正确：输入 `(B, D)`，输出 `A=(B,K)` 和 `Z_hat=(B,D)`。
- `A >= 0`。
- decoder basis `X` shape 为 `(K,D)`。
- 训练若干 epoch 后 reconstruction loss 下降。
- `X` row norm ≈ 1。

#### `test_stage5_activation.py`

测试：

- `road_basis_activation.parquet` schema 完整。
- `a_000 ... a_{K-1}` 存在。
- 所有 activation 非负。
- `activation_entropy >= 0`。
- `active_basis_count >= 0`。

#### `test_stage6_road_units.py`

测试：

- boundary GeoJSON 为 LineString。
- units GeoJSON 为 MultiLineString 或 LineString collection。
- 每个 unit 是 road graph connected component。
- 过滤后 `n_road_nodes >= min_road_nodes`。
- synthetic 数据中至少恢复 2 个不同景观单元。
- 跨 synthetic region 的 edge 更容易成为 boundary。

#### `test_stage7_baselines.py`

测试：

- baseline comparison csv 存在。
- 包含主方法和所有 baseline。
- 主方法 `within_activation_variance` 不高于 `fixed_grid_500m`。
- 主方法 unit geometry 依附 road graph。

#### `test_pipeline_e2e.py`

E2E 测试：

```text
synthetic pano features
→ map match
→ road context feature
→ train basis model smoke
→ infer activation
→ extract road units
→ evaluate
```

要求：

- 全流程不报错。
- 输出文件存在。
- 至少生成 2 个 road landscape units。
- E2E < 60s。

---

## 5. 实现优先级

### Phase 1：先跑通 road-based pipeline

必须完成：

```text
Stage 2 map-match
Stage 3 network_kernel road context
Stage 4 sparse autoencoder without spatial loss first
Stage 5 activation inference
Stage 6 road graph boundary + connected components
E2E synthetic test
```

### Phase 2：加入城市语境约束

加入：

```text
A spatial smoothness loss
X diversity penalty
X row normalization
city-balanced training
```

### Phase 3：稳定性与跨城验证

加入：

```text
multi-seed training
leave-one-city-out reconstruction
automatic basis matching report
cross-city activation comparison
```

---

## 6. Go / No-Go Checkpoints

| Stage | Checkpoint | 通过条件 |
|---|---|---|
| Stage 1 | Feature norm | `||y_i||_2 ≈ 1` |
| Stage 2 | Map-match ratio | `matched_ratio >= 0.90` |
| Stage 2 | Road graph connectivity | 最大连通分量覆盖 ≥ 0.90 road nodes |
| Stage 3 | Road context quality | 同 region 相似度 > 跨 region 相似度 + 0.2 |
| Stage 4 | Reconstruction | train loss 明显下降，final loss < initial loss × 0.8 |
| Stage 4 | A nonnegative | min(A) >= 0 |
| Stage 4 | X normalized | 每个 `x_k` L2 norm ≈ 1 |
| Stage 5 | Activation sparsity | median active_basis_count < K × 0.5 |
| Stage 6 | Boundary ratio | 约等于 `1 - boundary_quantile`，允许 ±5% |
| Stage 6 | Unit size | median n_road_nodes ∈ [3, 100] |
| Stage 7 | Road validity | 所有 unit 是 road graph connected subgraph |

任一 checkpoint 不通过，对应 stage 应打印：

```text
[CHECKPOINT FAIL] <reason>
```

并以非零 exit code 退出。

---

## 7. 命令行主流程示例

```bash
cd /workplace/urban_visual_gene

# Stage 1: extract Y
python scripts/stage1_extract_pano_features.py \
  --input data/panoramas.parquet \
  --output outputs/Austria/Vienna/pano_features.parquet \
  --model dinov2_vitb14

# Stage 2: road map matching + road graph
python scripts/stage2_map_match_road.py \
  --pano-features outputs/Austria/Vienna/pano_features.parquet \
  --roads data/logical_streets.geojson \
  --output-dir outputs/Austria/Vienna \
  --node-spacing-m 25 \
  --max-match-distance-m 30

# Stage 3: build Z_road
python scripts/stage3_build_road_context_features.py \
  --matched-panos outputs/Austria/Vienna/road_matched_panos.parquet \
  --road-nodes outputs/Austria/Vienna/road_graph_nodes.parquet \
  --road-edges outputs/Austria/Vienna/road_graph_edges.parquet \
  --output outputs/Austria/Vienna/road_context_features.parquet \
  --method network_kernel \
  --context-radius-m 100 \
  --kernel-sigma-m 50

# Stage 4: train basis model Z_road ≈ A X
python scripts/stage4_train_road_basis_model.py \
  --input outputs/Austria/Vienna/road_context_features.parquet \
  --road-edges outputs/Austria/Vienna/road_graph_edges.parquet \
  --output-model models/road_basis_model.pt \
  --output-basis models/road_landscape_basis.npy \
  --K 100 \
  --epochs 100

# Stage 5: infer A
python scripts/stage5_infer_road_basis_activation.py \
  --input outputs/Austria/Vienna/road_context_features.parquet \
  --model models/road_basis_model.pt \
  --output outputs/Austria/Vienna/road_basis_activation.parquet

# Stage 6: extract MRLU
python scripts/stage6_extract_road_units.py \
  --activation outputs/Austria/Vienna/road_basis_activation.parquet \
  --road-nodes outputs/Austria/Vienna/road_graph_nodes.parquet \
  --road-edges outputs/Austria/Vienna/road_graph_edges.parquet \
  --output-dir outputs/Austria/Vienna \
  --boundary-quantile 0.90

# Stage 7: evaluate
python scripts/stage7_evaluate_and_baselines.py \
  --city-dir outputs/Austria/Vienna \
  --output outputs/baseline_comparison.csv
```

---

## 8. 关键实现提示

### 8.1 不要把 `X` 写死成语义类型

错误：

```python
basis_names = ["openness", "greenery", "commerciality"]
```

正确：

```python
basis_id = 0, 1, ..., K-1
```

语义解释只能作为后处理：

```text
top activating road nodes / panos
external variable correlation
manual interpretation
```

### 8.2 不要直接在 pano-level `Y` 上训练主模型

错误主线：

```text
Y_pano ≈ A X
```

正确主线：

```text
Z_road = P_Groad(Y_pano)
Z_road ≈ A X
```

### 8.3 空间单元不是 grid

主输出应为 road-based unit：

```text
minimum_road_landscape_units.geojson
```

geometry 应依附 road network，优先使用 LineString / MultiLineString。

如需 polygon，可在后续阶段由 road unit 派生，不作为主实现。

### 8.4 正交不是严格约束

`X` 的 diversity 约束是弱约束，不要强迫完全正交。

原因：城市景观基本特征可能相关。

实现上：

```text
lambda_div = small value, e.g. 1e-3
```

### 8.5 最小空间单元来自 `A` 的路网组织

最终空间单元定义：

> 道路网络上连续、内部 `A` 相似、边界 `A` 突变、且具有一定稳定性的 connected subgraph。

---

## 9. 最终交付物

每个城市：

```text
outputs/<country>/<city>/pano_features.parquet
outputs/<country>/<city>/road_matched_panos.parquet
outputs/<country>/<city>/road_graph_nodes.parquet
outputs/<country>/<city>/road_graph_edges.parquet
outputs/<country>/<city>/road_context_features.parquet
outputs/<country>/<city>/road_basis_activation.parquet
outputs/<country>/<city>/road_activation_boundaries.geojson
outputs/<country>/<city>/minimum_road_landscape_units.geojson
outputs/<country>/<city>/unit_statistics.csv
outputs/<country>/<city>/evaluation_report.json
```

全局：

```text
models/road_basis_model.pt
models/road_landscape_basis.npy
models/training_report.json
outputs/baseline_comparison.csv
```

测试：

```text
pytest tests/ → 0 failures, 0 errors
```

---

## 10. 最终一句话

本系统要实现的不是“从孤立街景点学习视觉基底”，而是：

> 先将街景视觉特征组织到道路网络上，形成路网景观表征 `Z_road`；再从 `Z_road` 中学习全球共享的路网景观基础特征 `X` 和路网上的激活场 `A`；最终根据 `A` 在道路网络上的连续性与突变，识别城市最小路网景观空间单元 `U`。

