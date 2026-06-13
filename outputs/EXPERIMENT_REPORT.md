# 道路型最小景观单元（MRLU）实验报告

**日期**：2026-06-13
**数据**：全部使用本机 `./data/SVIs/GSV`（已从 NAS 复制，运行期间不访问 NAS）
**样本**：每城随机抽样 500 个街景点（`random_state=42`），4 方向（0°/90°/180°/270°）DINOv2-ViT-B/14 特征 concat → D=3072，L2 归一化
**模型**：稀疏自编码器 K=32，hidden=512，epochs=50，`lambda_sparse=5e-3`，`lambda_spatial=1e-3`，`lambda_div=1e-3`

---

## 1. 总体结果

| 指标 | Vienna | HongKong |
|------|-------:|---------:|
| 抽样街景点 | 500 | 500 |
| 缺失图片 | 0 | 0 |
| 匹配成功率 | 500/500 (100%) | 499/500 (99.8%) |
| 道路图节点数 | 237,590 | 247,772 |
| 道路图边数 | 509,346 | 548,074 |
| 有 pano 直接覆盖的节点 | 9,656 (4.1%) | 17,806 (7.2%) |
| 训练用节点（覆盖子集） | 8,691 train / 965 val | 16,026 train / 1,780 val |
| 重建误差 (mean cosine) | 0.220 | 0.237 |
| 激活稀疏度 median active | 18 / 32 | 9 / 32 |
| 激活边界数 | 20,206 | 43,558 |
| **提取的 MRLU 单元数** | **611** | **911** |
| 单进程耗时 | 89 s | 106 s |

> 两城均跑通完整 7 阶段流程（Stage1 特征 → Stage2 地图匹配 → Stage3 Z_road 上下文 → Stage4 稀疏基学习 → Stage5 激活推断 → Stage6 单元提取）。

---

## 2. 单元统计（unit_statistics.csv）

| 统计量 | Vienna (611 units) | HongKong (911 units) |
|--------|-------------------:|---------------------:|
| 每单元路网节点 mean / median | 165.5 / 96 | 84.0 / 37 |
| 每单元 pano mean / median | 12.3 / 10 | 13.0 / 8 |
| 单元道路长度 (m) mean / median | 3,696 / 2,214 | 1,822 / 825 |
| 激活熵 mean | 3.35 | 3.33 |
| 单元置信度 mean | 0.53 | 0.53 |
| 边界对比度 mean | 0.11 | 0.11 |
| 主导基（dominant basis）种类 | 32 / 32 全用到 | 32 / 32 全用到 |

**解读**：香港的单元明显更细碎（中位 37 节点 / 825m），Vienna 更粗（中位 96 节点 / 2.2km）。这与稀疏度一致——香港 median_active=9 比 Vienna=18 更稀疏、激活更具区分度，因此切出更多、更小的单元，符合香港高密度异质街景的直觉。

---

## 3. 产物文件

每城 `outputs/<国家>/<城市>/`：
- `pano_features.parquet` — Stage1 特征（D=3072）
- `road_matched_panos.parquet` / `road_graph_nodes.parquet` / `road_graph_edges.parquet` — Stage2 路网图
- `road_context_features.parquet` — Stage3 Z_road（1.6G）
- `road_basis_activation.parquet` — Stage5 激活 A
- `minimum_road_landscape_units.geojson` — **MRLU 单元（含 MultiLineString 几何）**
- `road_activation_boundaries.geojson` — 激活边界
- `unit_statistics.csv` — 单元统计表
- `stage_reports/*.json` — 各阶段 checkpoint 报告

公共：`models/road_basis_model.pt`、`models/road_landscape_basis.npy`（K×D 基矩阵 X）

---

## 4. 运行中修复的问题（真实数据踩坑）

1. **覆盖节点过滤失效**：Stage3 插值后所有节点 `total_weight>0`，导致 Stage4 误训全部 23 万节点（曾耗时 4.4h）。改用 `n_panos>0` 锁定真实覆盖节点 → Stage4 降到 ~20-28s。
2. **Stage4 torch deploy 崩溃**：spatial loss 对 torch tensor 做 Python 逐元素迭代触发 `cannot allocate PyObject ... torch deploy interpreter`。改用纯 Python int 列表。
3. **Stage6 全表布尔扫描崩溃 + 0 单元**：每条边界边扫 50 万行边（numpy 维度错误），且 `road_graph_nodes` 无 `n_panos` 列导致 `min_panos` 过滤掉全部单元。改为一次性构建 `edge_meta` 字典，并从 Z_road 合并 `n_panos` 进节点。
4. **香港 Stage2 段错误**：`sample_road_nodes` 用 shapely `geom.interpolate()`，香港某条 OSM 几何让 GEOS 在 C 层 segfault。改为**纯 numpy 沿折线线性插值**（更稳更快）。
5. **DINOv2 + geopandas 同进程 segfault**：同一进程先用 torch.hub 把 DINOv2 加载到 CUDA、再做 Stage2 重几何运算会崩。将 Stage1 与 Stage2-6 拆为独立进程。
6. **稀疏度调参**：扫描得 `lambda_sparse=5e-3`（median_active 32→16/9），较 1e-4 明显改善。

---

## 5. 已知局限与建议

- **boundary 阈值 τ=0**：Stage6 取激活余弦距离的 0.90 分位作边界阈值，但两城该分位数都为 0（>90% 相邻节点激活完全相同），实际成了「只要激活有任何差异即设边界」。后果是边界略偏多、单元划分对激活噪声敏感。
  - 建议：改用相对阈值（如非零距离的分位数）或对 τ 设下限；或在 Stage4 用 ReLU 末端激活（可得精确 0）/ top-k 激活以增强稀疏性与区分度。
- **覆盖率低（4–7%）**：500 抽样点只覆盖城市路网很小一部分，LCC 偏碎（Vienna 77.7%、HK 59.9%）。大量单元几何来自插值节点，置信度普遍在 0.53 附近。
  - 建议：扩大抽样量（5k–全量）可显著提升覆盖与单元可靠性。本机已含两城各 500 抽样图；Vienna 全量图已在本地（3.2G），香港为部分（215G/938G）。
- **Vienna 单元偏粗**：median_active=18 仍偏稠密，建议对 Vienna 单独提高 `lambda_sparse` 或增大 epochs。

---

## 6. 复现命令

```bash
# 特征已在 outputs/ 下；如需重跑（读本地 data/，不碰 NAS）：
python run_experiment.py --city Vienna   --max-panos 500 --K 32 --epochs 50 --skip-stage1
python run_experiment.py --city HongKong --max-panos 500 --K 32 --epochs 50 --skip-stage1
# 注意：Stage1（DINOv2）需与 Stage2-6 分进程运行，避免 native 库 segfault
```
