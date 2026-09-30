# 审稿意见回应与修改方案（Revision Plan）

> 针对评审意见（结论 Reject）。本方案分两条主线：**(A) 收敛主张、诚实重定位**；
> **(B) 用一套可执行的实验程序把证据强度补上**。每项标注优先级与在本仓库现有流水线上的**可行性**
> （✅ 现可执行 / 🟡 需新增数据或标注 / 🔴 需较大工作量）。

---

## 0. 总体判断

评审的核心批评基本成立，必须正面接受，不能仅做语言润色：

1. **τ_c = 0 边界退化**（Major 3）——这是**致命且必修**的，当前主结果不可信。
2. **缺 baseline / 缺定量评估 / 缺验证**（Major 4、10、5.1–5.2）——无法证明方法有效。
3. **数据规模与覆盖率**（Major 2）——500 点、4–7% 覆盖，结论超出数据支撑。
4. **概念与主张过强**（Major 1、5、6；Minor 1、2、8）——"minimum / objective / interpretable basis / cross-city transferable" 均需收敛或证明。

**定位结论**：当前稿件不应投综合性顶刊。两种可行路径：
- **路径 A（快）**：收敛主张 + 补 P0 实验，定位**城市计算 / GIScience 应用型期刊或会议**（如 *Computers, Environment and Urban Systems*、*IJGIS*、*Landscape and Urban Planning*、SIGSPATIAL）。
- **路径 B（慢）**：完成 P0+P1+P2 全部实验（多城市、人工标注、外部验证、机制分析），方有望冲击高水平期刊。

---

## 1. 主张收敛与重定位（回应 Major 1/12，Minor 1/2/8）

在补实验之前，先把不可证的强主张降级，避免评审一票否决：

| 当前措辞 | 问题 | 建议改为 |
|---|---|---|
| **Minimum** Road-based Landscape Unit | "最小性"无定义无证明（Major 1） | 给出可检验定义（见下），或改 **"Road-based Visual Segments / Units"** |
| **客观**分区（objective） | 含大量人为超参（Minor 2） | **data-driven / algorithmically reproducible** |
| **可解释视觉基**（interpretable visual basis） | 未证明语义（Major 5） | **visual latent basis / street-view feature basis**，待 §3.1 实验后再升级 |
| **跨城可迁移**（cross-city transferable） | 无 out-of-sample 设计（Major 6） | 暂删；补迁移协议实验后再提 |
| "缺乏从视觉证据出发的自动方法" | 过于绝对（Minor 1） | "尚缺少将自监督街景表征、路网约束与稀疏基分解结合的可复现框架" |

**给 MRLU 一个可检验定义（必做）**。建议采用"多尺度稳定 + 优化目标"双定义之一：
- **优化式**：MRLU 是在目标 `min Σ_u within_var(u) + β·(单元数)` 下、对路网图的某个分割解（β 控制粒度），即"在给定粒度惩罚下不可再合并/再分的视觉同质连通段"。
- **多尺度稳定式**：在阈值/粒度扫描中**持续稳定存在**的分割尺度上的单元（用 persistence / split-half 一致性度量）。
正文需证明"当前算法输出 ≈ 该定义的解"，否则去掉 "minimum"。

---

## 2. P0 — 必做修复（不做则结论不成立）

### P0-1　修复边界阈值退化 τ_c=0（Major 3）✅ 现可执行
**根因**：Softplus 末端激活几乎处处非零且高度平滑 → 相邻节点激活近乎相同 → 0.90 分位 = 0。
**修改（代码层，`scripts/stage6_extract_road_units.py` + `road_basis_model.py`）**：
1. **激活稀疏化**：编码器末端 Softplus → **ReLU**（产生精确 0），或加 **top-k / JumpReLU** 稀疏，使激活真正稀疏、相邻差异有意义。
2. **阈值改用非零分布**：`τ_c = Q_q({d_ij : d_ij>0})`，或对 τ 设绝对下限；并**报告边界强度分布**而非单一分位。
3. **换更稳的分割算子**：normalized cut / graph total-variation / Potts / change-point，作为对照。
4. 修复后**全部主结果重跑**（单元数、长度、地图、两城比较）。
> 现状 `median_active` Vienna=18/32、HK=9/32 已偏稠密，ReLU+top-k 是最直接抓手。

### P0-2　扩大样本 + 覆盖率敏感性（Major 2）✅ Vienna 全量可执行 / 🟡 HK 部分
- 采样规模扫描 **500 / 1k / 5k / 10k / 全量**，报告单元数、平均长度、边界位置、内部方差、边界对比度的**稳定性曲线**。
- **单独报告"仅直接覆盖道路"上的分割**（n_panos>0 子图），与插值全路网结果分开，杜绝"插值制造的伪连续/伪边界"。
- 本仓库：Vienna 全量街景图已在本地（3.2G）可直接放大；HK 为部分（215G/938G），需先补拷或在 NAS 上跑（注意进程隔离）。

### P0-3　Baseline 对比（Major 4）🔴 新增 `scripts/stage7_baselines.py`
在**同一路网图、同一评估指标**下比较：
1. 原始 DINOv2 特征直接算相邻余弦距离切边；2. PCA 降维后切边；3. k-means / GMM + 连通分量；
4. spectral clustering / normalized cut；5. 空间约束聚类（SKATER/区域化）；6. 纯路网拓扑分割；
7. **随机特征 / 打乱特征**（negative control）。
**目的**：证明 SAE 视觉基相对"DINO 直切"和"纯空间"确有增益，否则核心贡献不成立。

### P0-4　定量评估框架（Major 10、Minor 3）🔴 新增 `scripts/eval_segmentation.py`
报告并与 baseline 对比：
- **within-unit visual variance**（应显著低于随机连通分区，做置换检验）；
- **boundary contrast**（应显著高于随机边界）；
- **split-half stability**：不同街景子集 → 边界 IoU / boundary displacement；
- **seed stability**：不同随机种子下单元一致性；
- （有标注后）**precision/recall/F1** vs 人工边界。
- `confidence = σ(contrast − within_var)` 未校准 → 改名 **contrast score** 或用标注校准（Minor 3）。

### P0-5　稳定性实验（5.4、Major 10）✅ 现可执行（流水线快）
每城 **≥20 个随机采样种子** + **≥5 个训练种子**，报告边界/单元的稳定性分布。当前仅 `random_state=42` 一次，无法排除偶然。

---

## 3. P1 — 关键验证实验

### P1-1　视觉基可解释性（Major 5、5.5）✅ 图在本地，现可执行
- 为每个 basis 导出 **top-/bottom-activated 街景拼图**（我们本地有 4 角度原图）。
- 用现成语义分割（如 Mask2Former/SegFormer cityscapes）统计每个 basis 与 sky/building/vegetation/road/sidewalk/signage/vehicle 等类别的相关性。
- 报告 basis 在不同 seed/城市/样本量下的 **对齐稳定性（CKA / Procrustes / 匈牙利匹配）**。
- 证明后方可称 "interpretable basis"，否则只称 latent components。

### P1-2　跨城迁移协议（Major 6、5.9）✅ 现可执行
明确并实验：① Vienna 训→HK 推；② HK 训→Vienna 推；③ 两城联合训→第三城测；④ leave-one-city-out。
报告 basis alignment（CKA/Procrustes）与单元统计稳定性。**先在正文写清当前 `road_basis_model.pt` 的训练协议**（单城/联合/迁移）——否则"跨城"主张不能提。

### P1-3　空间平滑 vs 边界检测的自相矛盾（Major 7）✅ 现可执行
- `λ_spa` 消融（0 / 1e-4 / 1e-3 / 1e-2），报告边界数、平均单元长度、内部方差、边界对比度、人工一致性。
- 建议把均匀 L2 平滑换成 **分段平滑 / total-variation**，允许少数强边界——这与 P0-1 协同。

### P1-4　多方向拼接 + 方位对齐消融（Major 8、5.10、Minor 4）✅ 现可执行
- 比较 ①四向均值池化 ②四向 concat ③**按道路走向对齐的 front/back/left/right concat** ④rotation-invariant/attention pooling。
- 用定量指标（within-var / boundary contrast / 稳定性）证明 concat 优于 pooling，并检验"绝对方位"是否引入东西向 vs 南北向街道的伪差异。
- 同时说明 DINOv2 取的是 **CLS / mean-patch / 第几层**，并比较（Minor 4）。

### P1-5　外部城市形态验证（Major 9、5.7）🟡 需补外部数据
将 MRLU 边界与建筑密度/高度方差、土地利用混合度（land-use entropy）、POI 多样性、道路等级跃变、街区尺度、绿视率/天空率/建筑率、规划/保护区边界比较——**证明 MRLU 边界附近这些外部变量发生显著变化**（边界 vs 非边界的对照 + 置换检验）。

### P1-6　人工标注验证（5.1）🟡 需组织标注
选取若干道路走廊，多名标注者标"风貌变化点/同质段"，算 precision/recall/F1 与 boundary displacement tolerance；报告标注者间一致性（如 Fleiss κ）。

### P1-7　OSM 路网拓扑严谨性（Major 11）✅ 现可执行
- 利用 OSM `bridge/tunnel/layer/highway/oneway/footway/service` 标签，避免立交/桥隧**跨层错误连接**（当前 cKDTree 端点吸附可能在立交处误连）。
- 展示若干复杂路口案例；分道路类型报告覆盖率与分割结果。
- 强调："LCC≈100% ≠ 拓扑正确"，需逐层校验。

---

## 4. Minor — 快速文字/呈现修订

- **M1**：摘要"缺乏…"改为相对表述（见 §1 表）。
- **M2**：删"客观"，改 data-driven / reproducible。
- **M3**：confidence 改 contrast score 或校准。
- **M4**：写明 DINOv2 特征层/Token 并比较。
- **M5**：不报"32/32 被使用"，改报 **activation Gini / usage entropy / dead-basis ratio / top-k 累计激活**；说明 Softplus 导致普遍非零。
- **M6**：两城比较**按单位道路长度/单位面积标准化**（边界数/km、单元数/km²），并报告研究区面积、道路总长、路网密度、街景点密度、道路类型构成。
- **M7**：PCA-RGB 颜色配代表性街景样本，否则只作可视化、不作证据。
- **M8**：landscape basis → visual latent basis。
- **M9**：相关工作建立"点级街景指标→空间聚合→街道分区→图分割→表征可解释性"脉络，明确本文补的缺口。
- **M10**：明确"48 个单元测试 = 代码可运行 ≠ 方法有效"，不把软件测试当方法证据。

---

## 5. 执行清单与代码映射（本仓库可落地部分）

| 实验 | 优先级 | 新增/改动 | 可行性 |
|---|---|---|---|
| ReLU/top-k 稀疏 + 非零阈值边界 | P0-1 | `road_basis_model.py`、`stage6_*` | ✅ |
| 采样规模 500→全量扫描 | P0-2 | `run_experiment.py` 批量 | ✅(Vienna)/🟡(HK) |
| baseline 分割套件 | P0-3 | 新 `scripts/stage7_baselines.py` | 🔴 |
| 评估指标（var/contrast/stability/置换检验） | P0-4 | 新 `scripts/eval_segmentation.py` | 🔴 |
| seed/split-half 稳定性 | P0-5 | 批量脚本 | ✅ |
| basis top-activated 拼图 + 语义相关 | P1-1 | 新 `scripts/basis_interpret.py` | ✅(拼图)/🟡(语义分割) |
| 跨城迁移（train/infer 矩阵） | P1-2 | `run_stage45` 加 `--train-city/--infer-city` | ✅ |
| λ_spa / K / σ / R / δ 消融 | P1-3/5.3 | 批量脚本 + 汇总 | ✅ |
| 视角策略消融（含道路对齐） | P1-4 | `stage1` 加 pooling 选项 | 🟡 |
| 外部形态验证 | P1-5 | 需建筑/POI/land-use 数据 | 🟡 |
| 人工标注验证 | P1-6 | 标注协议 + 评估脚本 | 🟡 |
| OSM 分层拓扑 | P1-7 | `road_graph_utils` 用 tags | ✅ |

> 注：所有新脚本遵循仓库**进程隔离约定**（torch 阶段与 geo 阶段分进程，首行 `import scripts._env`），见 `crash_report_20260613.md` 与 README。

---

## 6. 建议的修订后论文结构

1. **Introduction**：明确缺口（自监督街景表征 × 路网约束 × 稀疏基 × 可检验单元定义）。
2. **Related Work**：补全 §M9 脉络。
3. **Problem & Definition**：给 MRLU **形式化、可检验**定义（§1）。
4. **Method**：Stage 1–6，突出原理与设计选择的**依据**（而非调试过程）。
5. **Experiments**：
   - 5.1 评估协议与指标（P0-4）；
   - 5.2 Baseline 对比（P0-3）；
   - 5.3 参数/尺度/采样敏感性与稳定性（P0-2/5、P1-3）；
   - 5.4 视角与特征层消融（P1-4）；
   - 5.5 basis 可解释性（P1-1）；
   - 5.6 跨城泛化（P1-2，多城市）；
   - 5.7 外部形态/人工验证（P1-5/6）。
6. **Discussion / Limitations**：诚实边界。
7. **Reproducibility**：**1 段**声明 + 指向 GitHub；工程细节（segfault/OpenMP/进程隔离）移入 **Supplementary / README**（回应 Major 12）。

---

## 7. 一句话回应评审的 Reject

接受当前证据不足的判断。计划：**(1) 修复 τ_c 退化并重跑主结果；(2) 加 baseline 与定量评估；(3) 扩样本并分离"覆盖 vs 插值"；(4) 收敛 minimum/objective/interpretable/transferable 等主张至可证范围；(5) 补 basis 解释、跨城协议与外部验证。** 完成 P0+P1 后重投应用型期刊；多城市 + 人工/外部验证齐备后再议高水平期刊。
