# MRLU 实验设计（Experiment Design）

> 目标：把当前"工程原型"升级为有证据支撑的研究。设计围绕评审的核心质疑展开：
> 扩样本消除插值伪影、用 baseline 证明方法必要性、用归一化指标与外部/人工验证证明单元真实。
> 每项标注**因变量**、**对照**与在本仓库的**可执行性**（✅可执行 / 🟡需新数据 / 🔴需新工程）。

---

## 0. 研究问题与假设

| # | 研究问题 (RQ) | 假设 (H) |
|---|---|---|
| RQ1 | 街景采样量如何影响分割结果？ | H1：覆盖率随 N 上升，插值伪影下降；**归一化指标（单元密度、边界密度）在足够 N 后收敛**（绝对单元数不收敛）。 |
| RQ2 | 稀疏视觉基（SAE）是否必要？ | H2：SAE 分割在视觉一致性/边界对比/稳定性上**优于** DINO 直切、PCA、k-means、纯空间等 baseline。 |
| RQ3 | 单元是否对应真实风貌变化？ | H3：MRLU 边界处的外部形态变量（建筑/土地利用/POI/路网等级）发生显著跳变，且与人工标注边界一致。 |
| RQ4 | 视觉基是否可解释、可跨城迁移？ | H4：每个 basis 对应稳定的语义模式；basis 在跨城/跨种子下可对齐（CKA/Procrustes）。 |
| RQ5 | 各组件与超参的贡献？ | H5：去掉 SAE / 空间平滑 / 多向拼接会显著降低指标（消融）。 |

**核心因变量（贯穿全部实验）**：
- `within_unit_var`（单元内视觉方差，越低越好）
- `boundary_contrast`（边界两侧激活差异，越高越好）
- `covered_unit_density`、`boundary_per_km`（**归一化**，跨 N/跨城可比）
- `split_half_IoU`、`seed_stability`（稳定性）
- `external_jump`（边界处外部变量变化显著性）、`human_F1`（与人工标注一致）

---

## 1. 数据与采样协议（RQ1）

**城市**：核心两城 Vienna（OSM `edges.shp`，全量图本地）、Hong Kong（全量 938G/270 万点在 NAS）；泛化扩展见 §5。

**采样规模**（自变量 N）：`500 → 2000 → 10k → 50k → 全量`，每城每个 N 用 `random_state ∈ {0..19}` 抽 **20 个独立子样**（用于稳定性，§4）。

**两种分割域（对照）**：
- **full-graph**（插值全路网，现状）
- **covered-only**（仅 pano 覆盖节点，`stage6 --covered-only`）——**作为主结论域**。

**因变量随 N 记录**：coverage、frac_positive_edges、covered-subgraph LCC、within_var、boundary_contrast、covered_unit_density、boundary_per_km。
**判据**：归一化指标在某个 N* 后趋于平台 → 报告 N*（"科学结论所需的最小采样量"）。

> 工程前置（🔴）：大样本需**分块 + 断点续跑**的流水线（见 §6），否则环境不稳定撑不住。

---

## 2. Baseline 对照（RQ2）

在**同一覆盖子图、同一评估指标**下比较（消除"方法只是更复杂"的质疑）：

| 组 | 方法 | 说明 | 可行性 |
|---|---|---|---|
| 本文 | SAE 视觉基 + 激活不连续 | 主方法 | ✅ |
| B1 | DINO 特征直切 | 相邻节点原始 DINO 余弦距离 > τ 切边 | ✅ |
| B2 | PCA(d) + 直切 | 降维后切 | ✅ |
| B3 | k-means / GMM + 连通分量 | 聚类标签变化处设边界 | ✅ |
| B4 | spectral / normalized cut | 图谱分割 | 🟡 |
| B5 | 空间约束聚类（SKATER/区域化） | 仅几何+特征 | 🟡 |
| B6 | 纯路网拓扑 / 纯几何分割 | 无视觉 | ✅ |
| B7 | **随机特征 / 打乱特征** | 负对照（下界） | ✅ |

**实现**：新增 `scripts/stage7_baselines.py`，输入同一 `road_context_features` + 图，输出各方法的单元划分；`scripts/eval_segmentation.py` 统一打分。

---

## 3. 评估指标框架（RQ2/RQ3，回应"无定量评估"）

`scripts/eval_segmentation.py` 对任意分割输出：
1. **within_unit_var**：单元内节点视觉特征方差均值；与**随机连通分区**做置换检验（n=1000），报告 z / p。
2. **boundary_contrast**：边界边特征距离 vs 内部边；置换检验。
3. **split_half**：把街景点随机二分各自建图分割，计算边界 IoU / boundary displacement（容差 25/50/100m）。
4. **seed_stability**：20 个采样种子间单元/边界一致性（ARI / IoU）。
5. **归一化**：covered_unit_density、boundary_per_km、unit_per_km²（回应跨城标准化）。
6. **calibration**：把 `confidence` 与人工/外部边界对齐校准；否则降级为 `contrast_score`。

**最小可发表门槛**：本文方法在 1–4 上**显著优于 B6/B7，且不劣于 B1–B3**。

---

## 4. 稳定性实验（RQ1）

- **采样稳定性**：每城每 N × 20 种子 → 边界/单元分布；报告均值±标准差、变异系数。
- **训练稳定性**：固定数据，5 个训练种子 → basis 对齐（§4 of RQ4）与单元一致性。
- 当前仅 `random_state=42` 一次，无法排除偶然——这是必补项。

---

## 5. 跨城泛化与 basis 可解释性（RQ4）

**迁移协议（明确 train/test，杜绝"联合训练却称迁移"）**：
- T1 Vienna→HK 推断；T2 HK→Vienna；T3 两城联合→第三城；T4 leave-one-city-out。
- `run_stage45` 加 `--train-out / --infer-out` 拆分训练与推断域。
- 指标：跨域 within_var/contrast、basis 对齐（CKA、Procrustes、匈牙利匹配）。

**basis 可解释性**（`scripts/basis_interpret.py`，✅本地有图）：
- 每 basis top-/bottom-activated 街景拼图；
- 用语义分割（Mask2Former/SegFormer）统计每 basis 与 sky/building/vegetation/road/sidewalk/signage 等类别相关性；
- basis usage 的 Gini / entropy / dead-ratio（替代"32/32 被使用"）。

**泛化城市集（🟡需取数）**：在两城基础上加 ≥5 座形态差异城市（欧规则街区 / 亚高密 / 北美低密 / 山地 / 历史城区），做 leave-one-city-out。

---

## 6. 消融（RQ5）

逐一关闭/扫描，看 §3 指标变化：
- **架构**：去 SAE（=B1）、Softplus vs **ReLU vs top-k** 激活、去空间平滑 `λ_spa∈{0,1e-4,1e-3,1e-2}`、去多向拼接（concat vs mean/max pool vs **道路走向对齐** vs attention）。
- **超参网格**：`K∈{8,16,32,64,128}`、`σ∈{15,30,50,100}m`、`R∈{50,100,200}m`、`δ(采样间距)`、`τ_c 分位`。
- **特征**：DINOv2 取 CLS vs mean-patch vs 多层。
> 每格用 §3 指标 + 稳定性汇总成热图/曲线。

---

## 7. 外部 / 人工验证（RQ3，回应"靠直觉"）

- **外部形态**（🟡）：MRLU 边界 vs 非边界处，比较 building footprint density、building height var、land-use entropy、POI diversity、road-class transition、block size、green/sky/building view ratio、规划/保护区边界——做边界 vs 内部的显著性检验。
- **人工标注**（🟡）：选 Vienna/HK 各若干走廊，≥3 名标注者标"风貌变化点/同质段"；算算法边界的 precision/recall/F1 + boundary displacement；标注者间一致性（Fleiss κ）。

---

## 8. 工程前置（撑住大样本，🔴）

1. **分块 + 断点续跑流水线**：按地理网格分块跑 stage1–3，落盘；stage4 全局训练、stage5 分块推断。已有重试 + `stack_embeddings` 兜底为基础。
2. **环境稳定**：按 `crash_report_20260613.md` 升级 PyTorch ≥2.4 或固定 numpy/scipy 版本，从根上减少 OpenMP/ABI 崩溃。
3. **指标/baseline/评估三件套脚本**：`stage7_baselines.py`、`eval_segmentation.py`、`basis_interpret.py`、`plot_sweep.py`（已建）。

---

## 9. 分阶段执行计划

| 阶段 | 内容 | 回答 | 产出 |
|---|---|---|---|
| **P-pre** | 分块续跑 + 升级环境 | 工程前置 | 稳定可扩的流水线 |
| **P0** | 采样扫描至全量（覆盖子图为主）+ 稳定性 20 种子 | RQ1 | N* 收敛曲线、归一化指标 |
| **P0** | baseline 套件 + 评估框架 | RQ2 | 方法显著性对照表 |
| **P1** | basis 解释 + 跨城迁移协议 | RQ4 | basis 语义图、迁移矩阵 |
| **P1** | 消融（架构/超参/特征） | RQ5 | 消融热图 |
| **P2** | 外部形态 + 人工标注验证 | RQ3 | F1 / 外部跳变显著性 |
| **P2** | 多城市泛化 | RQ4 | leave-one-city-out |

**里程碑判据**：P0 完成（收敛曲线 + 显著优于负对照）→ 可投应用型期刊；P0+P1+P2 齐备（多城市 + 人工/外部验证）→ 冲高水平期刊。

---

## 10. 设计表（速查）

| 自变量 | 取值 |
|---|---|
| 采样量 N | 500 / 2000 / 10k / 50k / 全量 |
| 采样种子 | 20（稳定性） |
| 分割域 | full-graph / covered-only |
| 方法 | SAE / B1–B7 |
| 城市 | Vienna / HK (+≥5 泛化城市) |
| 消融 | K, σ, R, δ, τ_c, λ_spa, 激活函数, 视角策略, 特征层 |

| 因变量 | 含义 |
|---|---|
| coverage, frac_positive_edges | 数据充分性 |
| within_var ↓, boundary_contrast ↑ | 分割质量 |
| covered_unit_density, boundary_per_km | 归一化、可比 |
| split_half_IoU, seed ARI | 稳定性 |
| CKA/Procrustes | basis 迁移 |
| external_jump, human_F1 | 真实性 |

*本设计是 `docs/REVISION_PLAN.md` 的可执行化版本；§1 采样灵敏度已有初步结果（REPORT §5.6）。*
