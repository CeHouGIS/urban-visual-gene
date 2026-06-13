# Urban Visual Gene — 道路型最小景观单元（MRLU）

基于街景视觉特征，将城市路网自动切分为**视觉同质的最小道路景观单元**（Minimum Road-based Landscape Unit, MRLU）。

## 方法概览

```
Y = Enc(I)                 街景图像 → 4 方向 DINOv2 特征 concat (D=3072, L2 归一化)
Z_road = P_Groad(Y)        点特征经距离衰减聚合到路网节点（多路段非归一化加权）
Z_road ≈ A · X             稀疏自编码器学习视觉基 X (K×D) 与激活 A (M×K, ≥0)
U = R(A, G_road)           相邻节点激活余弦距离超阈值处断边 → 连通分量 = MRLU
```

完整 7 阶段：Stage1 特征提取 → Stage2 地图匹配 → Stage3 Z_road 上下文 → Stage4 稀疏基学习 → Stage5 激活推断 → Stage6 单元提取 →（Stage7 baseline）。

## 目录结构

```
scripts/        各阶段实现（均可 import，也支持 CLI）
  stage1_extract_pano_features.py    4 方向 DINOv2 特征
  stage2_map_match_road.py           UTM 地图匹配 + 路网图
  stage3_build_road_context_features.py
  stage4_train_road_basis_model.py   稀疏自编码器
  stage5_infer_road_basis_activation.py
  stage6_extract_road_units.py       MRLU 提取
  road_graph_utils.py / road_basis_model.py / io_utils.py
  copy_data.py                       从 NAS 复制数据到本地 ./data
  plot_results.py                    出图
tests/          48 个合成数据单元测试（pytest，CPU，无需真实数据/GPU）
run_experiment.py    端到端实验入口（Vienna / HongKong）
outputs/        EXPERIMENT_REPORT.md + figures/ + 统计（大产物已 gitignore）
```

## 运行

```bash
# 依赖
pip install pytest numpy pandas scipy networkx geopandas shapely torch matplotlib

# 测试
pytest tests/ -q

# 实验（读本地 ./data，需先用 scripts/copy_data.py 准备数据）
#   注意：Stage1(DINOv2) 需与 Stage2-6 分进程运行，避免 native 库 segfault
python run_experiment.py --city Vienna   --max-panos 500 --K 32 --epochs 50 --skip-stage1
python run_experiment.py --city HongKong --max-panos 500 --K 32 --epochs 50 --skip-stage1
```

## 实验结果（500 抽样点 / 城）

| | Vienna | HongKong |
|---|--:|--:|
| 匹配率 | 500/500 | 499/500 |
| 激活稀疏度 (median active /32) | 18 | 9 |
| **MRLU 单元数** | **611** | **911** |

详见 [`outputs/EXPERIMENT_REPORT.md`](outputs/EXPERIMENT_REPORT.md) 与 `outputs/figures/`。

## 数据

街景图像与路网元数据**不随仓库分发**（`data/` 已 gitignore）。用 `scripts/copy_data.py` 从数据源准备到本地 `./data/SVIs/GSV`。
