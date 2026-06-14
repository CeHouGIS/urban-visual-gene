# CLAUDE.md — 项目工作准则

本文件是 Claude Code 在此项目中的行为规范。每次开始工作前请先读完。

你的每次回复之前都要带上Boss。

## 项目状态

这是一个**从零开始的新实现**。当前目录只有 `PRE.md`（技术路线）和 `REQUIREMENTS.md`（需求规格）。  
目标：按 REQUIREMENTS.md 把完整的 MVSU 识别 pipeline 实现出来，并让 `pytest tests/` 全部通过。

---

## 必须先读的文件

1. `REQUIREMENTS.md` — 数据契约、功能需求、测试规范（含 conftest.py 设计）
2. `PRE.md` — 算法原理与公式推导

读完再动手。不要从记忆中猜测接口，以 REQUIREMENTS.md §1 的 schema 为准。

---

## 目录结构（需要创建的）

```
urban_visual_gene/
├── scripts/
│   ├── model.py                      # ScenePredictionModel
│   ├── stage1_feature_extraction.py
│   ├── stage1_train_tokenizer.py
│   ├── stage2_build_graph.py
│   ├── stage3_train_model.py         # 调用 train_prediction_model.train()
│   ├── train_prediction_model.py     # train() 函数，供测试直接 import
│   ├── stage4_compute_surprise.py    # 导出 compute_surprise()
│   ├── stage5_detect_boundaries.py   # 导出 detect_boundaries()
│   ├── stage6_extract_mvsu.py        # 导出 extract_units()
│   └── stage7_baseline_comparison.py
├── tests/
│   ├── conftest.py
│   ├── test_stage1_features.py
│   ├── test_stage2_graph.py
│   ├── test_stage3_model.py
│   ├── test_stage4_surprise.py
│   ├── test_stage5_boundaries.py
│   ├── test_stage6_mvsu.py
│   ├── test_stage7_baselines.py
│   └── test_pipeline_e2e.py
├── models/                           # 训练产物（gitignore 大文件）
└── outputs/                          # 各城市输出
```

---

## 实现顺序

按以下顺序推进，**每个 stage 写完后立即跑对应测试**，通过再往下走：

1. `tests/conftest.py` — synthetic_city + synthetic_edges fixture
2. `scripts/model.py` — ScenePredictionModel，跑 test_stage3_model.py 的 T3-1/T3-2
3. `scripts/train_prediction_model.py` — train() 函数，跑 T3-3/T3-4
4. `scripts/stage4_compute_surprise.py` — compute_surprise()，跑 test_stage4_surprise.py
5. `scripts/stage5_detect_boundaries.py` — detect_boundaries()，跑 test_stage5_boundaries.py
6. `scripts/stage6_extract_mvsu.py` — extract_units()，跑 test_stage6_mvsu.py
7. `tests/test_pipeline_e2e.py` — 跑端到端集成测试
8. 其余 Stage 1/2/7（特征提取、图构建、baseline）

---

## 核心约束

### 脚本必须可以被 import

每个 stage 脚本既要支持 CLI（`python scripts/stage4_compute_surprise.py --city Vienna`），也要把核心逻辑封装为可以被测试直接 import 的函数：

```python
# 正确写法
def compute_surprise(panos, edges, K, model_path=None, use_random=False):
    ...

if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ...
    compute_surprise(...)
```

测试直接 `from scripts.stage4_compute_surprise import compute_surprise`，不走 CLI。

### 测试不依赖真实数据、不依赖 GPU

- `conftest.py` 的合成 fixture 在内存中生成，不读磁盘文件
- 模型在测试中用小参数（D_emb=8, hidden=16, K=16 或 128）
- 测试不调用 DINOv2/CLIP 等大模型，scene_embedding 直接用 fixture 提供的随机向量
- 所有测试在 CPU 上跑，不要求 CUDA

### 随机性

所有随机操作传 `seed=42` 或 `random_state=0`，确保测试可复现。

### Checkpoint 即 assert

每个 stage 函数内部按 REQUIREMENTS.md §6 的 Checkpoint 表做校验，不通过时 raise ValueError 并附上诊断信息：

```python
if not np.allclose(norms, 1.0, atol=1e-5):
    raise ValueError(f"[CHECKPOINT FAIL] L2 norm偏差过大: max={abs(norms-1).max():.6f}")
```

---

## 测试运行

```bash
# 安装依赖
pip install pytest numpy pandas scipy networkx geopandas shapely torch

# 跑全部测试
pytest tests/ -v --tb=short

# 跑单个文件
pytest tests/test_stage3_model.py -v

# 端到端
pytest tests/test_pipeline_e2e.py -v
```

**验收标准**：`pytest tests/` 的最终输出为 `N passed, 0 failed, 0 error`。

---

## 数据契约速查

以下是测试最常核对的约束，从 REQUIREMENTS.md §1 提取：

| 字段 | 约束 |
|------|------|
| scene_embedding | float32，L2 norm = 1.0 ± 1e-5 |
| scene_state | int，[0, K-1]，K 默认 128 |
| scene_confidence | float32，(0, 1] |
| surprise | float32，≥ 0，等于 -log(prediction_probability) |
| prediction_probability | float32，(0, 1] |
| edge_type | str，only: same_road_next / intersection_turn / spatial_near |
| visual_distance | float32，[0, 2]（= 1 - cosine similarity） |
| unit_id | str，全局唯一 |
| intra_visual_variance | float32，[0, 1) |

---

## 关键约束：进程隔离（防 OpenMP 崩溃）

PyTorch（Intel OpenMP/MKL，`libiomp5`）与 numpy/scipy/geopandas（GNU OpenMP，`libgomp`）
在**同一进程**会冲突，导致随机 native segfault（详见 `crash_report_20260613.md`）。

- **切勿在同一进程同时 `import torch` 与做 scipy/geopandas 重运算。**
- 真实数据流水线统一经 `run_experiment.py`（纯子进程编排器）运行，每个 stage 独立进程。
- 任何新脚本若用 numpy/scipy/torch，第一行 import `scripts._env`（锁定线程池为 1）。

## 禁止事项

- **不要跳过测试直接实现下一个 stage**。测试是唯一验收标准。
- **不要在测试中 mock 核心计算逻辑**（如 compute_surprise 的数学计算）。只 mock 文件 I/O 和大模型调用。
- **不要修改 conftest.py 中的 synthetic_city 固定参数**（N_PANOS=60, N_ROADS=6, N_REGIONS=3, seed=42），测试用例的期望值依赖这些参数。
- **不要为了让测试通过而在实现里硬编码合成数据的特征**。
