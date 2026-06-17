# scripts/ 模块结构

代码按职责分层为子包。根目录只留全局基础设施:`_env.py`(锁线程池,几乎被所有模块首行 import)、
`copy_data.py`(一次性数据搬运)、`__init__.py`。

> 约定:用 `/opt/conda/bin/python3`(系统 python 无 pandas/torch/geopandas);从**项目根**运行,
> import `scripts.*` 的入口需 `PYTHONPATH=$(pwd)`;重 IO/GPU 任务**串行** + `OMP_NUM_THREADS=1`
> (见 memory `check-resources-first`)。torch 与 scipy/geopandas 不能同进程(见 `CLAUDE.md` / `crash_report`)。

## 包划分

| 子包 | 职责 | 主要模块 |
|---|---|---|
| `scripts/core/` | 共享基础设施 | `io_utils` `cities` `road_graph_utils` `road_basis_model` |
| `scripts/pipeline/` | 主流水线(测试 + `run_experiment.py` 依赖) | `stage1..6_*`、`run_stage1/2/3/45/6` |
| `scripts/analysis/` | 评估 / 解释 / baseline | `baseline_common` `sae_metrics` `basis_{interpret,roles,similarity,align}` `visual_syntax` `spatial_organization` `unit_coherence` `eval_segmentation` `stage7_baselines` |
| `scripts/sampling/` | 街景采样与 sweep | `pano_sampling` `select_panos` `bench_sampling` `run_sampling_sweep` `run_k_eval` `run_basis_train` |
| `scripts/quality/` | 图像质量 | `image_quality` `train_quality_model` `validate_glare` |
| `scripts/viz/` | 出图 | `plot_results` `plot_pca_maps` `plot_umap` `plot_sweep` `plot_k_sweep` |
| `scripts/dash/` | Dashboard 数据 / 服务 / 图 | 见下 |
| `scripts/dedup/` | pano 特征 / 去重 / 联合基 / 建筑感知 | 见下 |
| `scripts/buildings/` | 建筑轮廓下载 | `download_buildings` |

依赖方向(单向,无环):`core` ← {`pipeline`,`analysis`,`sampling`,`quality`,`viz`,`dash`,`dedup`};
`pipeline` ← {`dash`,`dedup`,`tests`,`run_experiment`};`_env` 被所有包依赖。

### scripts/dash/ — Dashboard 数据与服务
| 脚本 | 作用 |
|---|---|
| `build_dashboard_data.py` | 地图资产:节点 pos/rgb/attr 二进制、道路边、单元/边界 geojson、meta(读 `CITIES` 指向实验目录的 `*_joint` 产物) |
| `build_dashboard_data2.py` | 特征空间联合 UMAP `*_embed.bin` + `analysis.json`(含转移矩阵) |
| `rebuild_analysis_json.py` | 只重拼 `analysis.json`(跳过 UMAP),并算共现矩阵/层次聚类树 |
| `build_sim_pairs.py` | 相似街景页 `sim_pairs_<city>.json`(15m 内 pano 对 + 逐方向余弦) |
| `build_node_pano_bind.py` | 节点↔最近 pano 绑定 `<city>_bind.bin` |
| `filter_binding_buildings.py` | 绑定线穿建筑判定 `<city>_bindblock.bin`(用 `data/buildings`) |
| `build_dropped_panos.py` | 去重被剔除点 `<city>_dropped_*`(灰色图层) |
| `plot_similarity_examples.py` / `plot_basis_hierarchy.py` | 静态图(相似度示例 / 基层次聚类) |
| `serve.py` | gzip + CORS + no-cache 静态服务器:`python scripts/dash/serve.py 8765` |

### scripts/dedup/ — pano 特征 / 去重 / 联合基 / 建筑感知实验
| 脚本 | 作用 |
|---|---|
| `extract_full_pano_features.py <City>` | 全量库 DINOv2 特征(GPU,多 worker,`model.half()`,断点续跑)→ `outputs/full_feats/` |
| `export_panos_full.py` | 全量有图 pano 坐标/id → `dashboard/data/panos_*` |
| `dedup_chain.py <City> [tau]` | 两段去重:新规则(5m+任一方向>0.6+最新)→ 旧规则(15m+整体≥0.90+medoid) |
| `build_dedup_pano_features.py <City> <out> [max]` | 用去重点构造 `pano_features.parquet`(子集 full_feats,HK 可抽稀) |
| `joint_stage5.py` / `joint_stage6.py` | 用联合基(`outputs/transfer/joint`)推断激活 + 重提单元(`USE_DEDUP_BLD=1` 切 dedup_bld) |
| `run_stage3_bld.py --out <dir> --buildings <pq>` | 建筑感知 Stage3(剔除穿楼的 pano→节点贡献) |

---

## 端到端常用流程(均从项目根、串行)

**主流水线**(每 stage 独立子进程):
```
/opt/conda/bin/python3 run_experiment.py --city Vienna           # 编排 stage1→6
/opt/conda/bin/python3 run_experiment.py --city both --skip-stage1
```

**A. 重建 Dashboard 数据**(改了某实验目录后)
```
/opt/conda/bin/python3 scripts/dash/build_dashboard_data.py      # 地图(FPE 偶发,失败重跑)
/opt/conda/bin/python3 scripts/dash/build_dashboard_data2.py     # UMAP+分析
/opt/conda/bin/python3 scripts/dash/build_node_pano_bind.py      # 绑定
for c in Vienna HongKong; do /opt/conda/bin/python3 scripts/dash/filter_binding_buildings.py $c; done
python scripts/dash/serve.py 8765
```

**B. pano 去重**:`dedup/extract_full_pano_features`(GPU)→ `dedup/export_panos_full` → `dedup/dedup_chain` → `dash/build_dropped_panos`

**C. 建筑感知重跑**(dedup_bld):`dedup/build_dedup_pano_features` → `pipeline/run_stage2`(复用图)→ `dedup/run_stage3_bld`(建筑)→ `dedup/joint_stage5/6`(USE_DEDUP_BLD=1)

**D. 发布**:站点 = `dashboard/`;GitHub Pages 经 `gh-pages` 分支 + Actions workflow;图片走 `config.js` 的 `IMG_BASE` 指向公网 HTTPS 图片后端(`serve.py` 已带 CORS)。

## 验收
`OMP_NUM_THREADS=1 /opt/conda/bin/python3 -m pytest tests/ -q` 应 **51 passed**。
测试直接 import `scripts.pipeline.*` / `scripts.core.*` / `scripts.sampling.pano_sampling`,
移动模块时务必同步更新测试 import 并复跑。

> 已删除的废弃脚本:`export_panos.py`(被 `dedup/export_panos_full` 取代)、`dedup_panos.py`(被 `dedup/dedup_chain` 取代)。
