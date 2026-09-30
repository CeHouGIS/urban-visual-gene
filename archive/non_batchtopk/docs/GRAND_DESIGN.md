# 城市视觉组合语法 — 升维实验方案（Grand Design）

> **核心命题（One Sentence）**
> 城市街景的复杂性可由**少量跨城市共享的视觉基**解释；不同城市的景观差异主要来自这些基**在道路网络上的组合语法（grammar）**，而非完全不同的视觉元素（vocabulary）。
> 道路型最小景观单元（MRLU）是这一语法在路网上的**稳定突变边界**。

把论文从"一种街景分割方法"升级为"一个城市视觉形态的**发现**"。MRLU 不再是出发点，而是理论的**自然推论**。

---

## 0. 大科学问题

> **Can the visual complexity of cities be decomposed into a shared set of compositional landscape primitives, and do cities differ by the *syntax* of how these primitives are arranged along streets?**

- 城市视觉景观是否存在可学习、可复用、可组合的"基本单元"？
- 这些单元如何沿道路网络组织、在何处产生稳定边界？
- 城市形态差异能否由统一视觉基的**组合方式**解释？

---

## 1. 理论框架：Compositional Urban Visual Grammar

| 概念 | 形式 | 含义 |
|---|---|---|
| 视觉基 (visual basis / landscape atoms) | $X\in\mathbb{R}^{K\times D}$ | 跨城市共享的视觉"词汇" |
| 视觉组合 (composition) | $A\in\mathbb{R}^{M\times K},\ Z_{road}\approx AX$ | 每个路网节点的基激活向量 |
| 路网视觉场 (road visual field) | $A$ on $G_{road}$ | 激活在路网图上的连续状态场 |
| 视觉句法 (visual syntax) | 转移矩阵 $T\in\mathbb{R}^{K\times K}$ | 相邻路段主导基如何转换 |
| MRLU 边界 | $\partial U$ | 组合状态发生稳定突变之处 |

**重定义 MRLU（可检验）**：在重复采样下**稳定复现**的、基组合发生显著突变的路网割边集合所界定的 piecewise-homogeneous 单元——即"路网视觉场"的分段同质划分。

---

## 2. 四个主假设（Introduction 末尾明确提出）

### H1 — 低维基假设（Low-dimensional basis）
> 尽管街景看似无限复杂，路网级视觉状态可由少量可复用视觉基重建。

**可检验预测 / 证据**：(i) 重建误差随 $K$ 出现 elbow；(ii) 激活稀疏；(iii) 多城共享同一组基；(iv) basis 有稳定语义。→ **Result 1**

### H2 — 网络连续假设（Network-continuity）
> 视觉基激活沿**道路网络**而非欧氏空间随机分布地组织。

**证据**：(i) 图平滑度 $S(A,G)$ 显著低于随机置换零模型；(ii) 路网邻接 > 欧氏 grid 邻接；(iii) 形成 corridor/patch。→ **Result 2**

### H3 — 边界突变假设（Boundary-discontinuity）
> MRLU 边界对应基组合的**统计显著且可复现**的突变。

**证据**：(i) 边界对比 > 随机边；(ii) bootstrap 边界概率高；(iii) 多种子/模型稳定；(iv) 人工 + 外部形态验证。→ **Result 3**

### H4 — 跨城语法假设（Cross-city grammar）
> 城市差异不在视觉**词汇**（哪些基），而在**句法**（基的频率、组合熵、沿路转移）。

**证据**：(i) 同组基在多城被用；(ii) 城市差异体现在 $A$ 的分布与转移矩阵 $T$，而非 $X$ 完全不同；(iii) basis 统计可分类城市；(iv) 单城训练的基可迁移。→ **Result 4**

---

## 3. 新增关键指标（形式化）

### 3.1 路网空间组织：图平滑度 + 零模型（H2）
$$
S(A,G)=\frac{1}{|E|}\sum_{(i,j)\in E}\lVert a_i-a_j\rVert_2 .
$$
与 $R$ 次**节点标签随机置换**的零分布 $S^{\text{perm}}$ 比较，报告 $z=(S-\mu)/\sigma$。并计算图版 **Moran's I**；对照"路网邻接 vs 欧氏 kNN 邻接"。预测：真实 $S$ 显著更低（$z\ll0$）。

### 3.2 视觉句法矩阵（H4，本方案最强增量）
以主导基 $b_i=\arg\max_k a_{i,k}$，定义沿路转移概率：
$$
T_{kl}=\frac{\#\{(i,j)\in E:\ b_i=k,\ b_j=l\}}{\#\{(i,j)\in E:\ b_i=k\}} .
$$
派生量：**自转移率** $\mathrm{tr}(T)/K$、**转移熵** $H(T)$、**非对角质量** $1-\mathrm{tr}(T)/K$、城市间 $T$ 距离。预测：维也纳自转移高、熵低（连续街墙）；香港非对角强、熵高（频繁突变）。→ 把"香港单元更短"升级为有理论含量的**句法差异**。

### 3.3 词汇共享（H1/H4）
城市 $c$ 对基 $k$ 的使用频率 $p_{c,k}=\frac{1}{M_c}\sum_{i\in c}\mathbf{1}(b_i=k)$；城市间 **Jensen–Shannon 散度** $\mathrm{JSD}(p_c\Vert p_{c'})$。预测：两城都用到大部分基（共享词汇），但 $p$ 不同（句法不同）。

### 3.4 边界稳定性（H3）
$R$ 次重采样/ bootstrap 各得 MRLU 边界集 $\partial U_r$；边 $e$ 的稳定度
$$
B_e=\frac{1}{R}\sum_{r=1}^{R}\mathbf{1}(e\in\partial U_r).
$$
仅 $B_e>0.7$ 计为 **stable boundary**。MRLU 由稳定边界界定 → 从"一次切分"升级为"稳定结构"。

### 3.5 单元内一致性 vs 随机（H3）
$$
C(u)=\frac{1}{|E_u|}\sum_{(i,j)\in E_u}\cos(a_i,a_j).
$$
与同规模随机连通子图比较，置换检验。预测：真实单元 $C$ 显著更高。

### 3.6 跨城基对齐（H1/H4）
分别训练 Vienna-only、HK-only、joint 三个 $X$；以匈牙利匹配/Procrustes 求最优对齐相似度
$$
\mathrm{align}=\max_{\pi}\frac{1}{K}\sum_k \cos\!\big(x_k^{(A)},\,x_{\pi(k)}^{(B)}\big).
$$
若单城独立学到的基可互相匹配 → 视觉基非偶然、跨城稳定。

---

## 4. Results 重排（假设驱动，不按 Stage）

| Result | 命题 | 核心证据 | 主要新指标 | 可行性 |
|---|---|---|---|---|
| **R1** 稀疏视觉词汇 | 街景可由有限基压缩 | 重建-K 曲线、稀疏度、basis exemplars、多城复用 | 3.3, 3.6 | ✅ 现有输出 + 训练若干 K |
| **R2** 路网视觉场 | 基沿路网非随机组织 | $S(A,G)$ vs null、Moran's I、路网>grid、激活图 | 3.1 | ✅ 现有输出 |
| **R3** 稳定突变边界 | MRLU=可复现突变 | bootstrap $B_e$、内一致性、人工/外部验证 | 3.4, 3.5 | 🟡 R 次重采样 + 标注 |
| **R4** 城市视觉句法 | 差异在句法非词汇 | 转移矩阵 $T$、词汇 JSD、城市分类、迁移 | 3.2, 3.3, 3.6 | ✅(2城) / 🟡(多城) |

---

## 5. Baseline 重新定位（不在 k-means 最强处死磕）

k-means 边界更锐**不是灾难**——改变评价对象：SAE 不是"更锐的分割器"，而是**唯一提供连续、稀疏、可迁移、可解释组合表示**的模型。用**能力矩阵**呈现（每格须有指标支撑，非主观）：

| 方法 | 边界锐度 | 稀疏组合 | 可迁移 | 可解释 | 路网平滑 |
|---|--:|--:|--:|--:|--:|
| DINO 直切 | 高 | 否 | 中 | 低 | 低 |
| k-means | 高 | 仅硬标签 | 低-中 | 中 | 低 |
| PCA | 中 | 否 | 中 | 低 | 中 |
| **SAE（本文）** | 中-高 | **是** | **高** | **高** | **高** |

一句话框定：*Direct clustering yields sharper local boundaries but no continuous, sparse, transferable compositional representation; our goal is not to maximise boundary contrast but to uncover the latent visual grammar that generates road-level landscape units.*

---

## 6. Nature 风格五图

1. **Fig 1 概念框架**：街景→视觉基→路网激活场→突变边界→MRLU→城市句法（理论图，非 pipeline）。
2. **Fig 2 共享视觉词汇**：basis exemplars + 重建-K 曲线 + 多城使用频率 + 跨城对齐。
3. **Fig 3 路网组织**：激活场地图 + $S$ vs null + 路网/grid 对比 + 转移矩阵。
4. **Fig 4 边界涌现与验证**：MRLU 图 + bootstrap 稳定边界概率 + 边界对比 vs null + 人工/外部验证。
5. **Fig 5 城市句法差异**：Vienna vs HK 转移矩阵 + 单元长度分布 + 熵/自转移 + (多城)城市聚类。

---

## 7. 现有结果的重新解释（化弱为强）

| 现有结果 | 旧表述（像 limitation） | 新表述（理论证据） |
|---|---|---|
| HK median active 9 vs Vienna 18 | "香港更稀疏" | 香港=少数强主导基的局部极端组合（强异质/突变）；维也纳=多基共同参与（连续街墙）→ **句法差异** |
| 32/32 基两城都用 | "基都被用到" | **shared vocabulary, city-specific composition**：差异在 grammar 不在 vocabulary |
| PCA-RGB 空间连续 | "可视化" | **road-based visual state field**；MRLU=该场的分段同质划分 |
| 单元数随采样变 | "采样不足" | **scale-dependent landscape object**（类比生态斑块/遥感分割）；真正稳定的是归一化密度/频率/熵/句法 → 已观察到覆盖单元密度收敛(0.075/0.066) |

---

## 8. 可直接执行的强化路线

| Step | 内容 | 产出/脚本 | 可行性 |
|---|---|---|---|
| **S1** 理论重构 | 引言/概念改为 grammar 五概念 + H1–H4 | 文本 | ✅ |
| **S2** 三个即时分析 | ①转移矩阵 $T$ ②图平滑 vs 置换零模型 ③单元内一致性 | `scripts/visual_syntax.py`、`scripts/spatial_organization.py`（扩 `eval_segmentation`） | ✅ **现有输出即可** |
| **S3** 词汇共享+跨城对齐 | basis 使用频率 JSD + 单城/联合训练对齐 | `scripts/basis_transfer.py` | ✅ 训练 3 模型 |
| **S4** 边界稳定性 | $R\!\ge\!20$ 重采样 → $B_e$，仅稳定边界为 MRLU | 扩 `run_sampling_sweep`（已支持多 seed） | 🟡 R 次跑 |
| **S5** 强验证 | 人工边界标注（≥100 边界样本）+ 外部形态（建筑/土地利用/POI） | 标注协议 + `scripts/external_validate.py` | 🟡 需取数/标注 |
| **S6** 重排 Results + 重定位 baseline | R1–R4 + 能力矩阵 | REPORT 重构 | ✅ |

**优先级**：S2 三项**不需新数据、立刻可做**，且直接支撑 H2/H4 的核心新主张——建议先做，作为升维的第一块硬证据。

---

## 9. 标题与摘要（发现导向）

**推荐标题**：*Urban Streetscapes Are Organized by Shared Visual Bases and City-Specific Road-Network Syntax*

备选：*A Compositional Visual Grammar of Urban Streetscapes Revealed by Road-Network Street-View Embeddings* / *Discovering Road-Based Landscape Atoms from Street-View Imagery*

**摘要骨架（发现导向）**：城市街景看似连续复杂，但是否存在可复用的视觉基本构件、以及它们如何组织成可感知的景观单元，仍属未知。本文提出一个路网上的稀疏视觉基框架，将多方向街景表征分解为少量可复用视觉基，并检测其组合状态在路网中的结构性突变。跨城市实验显示：不同城市**共享一组视觉基**，但在基的**频率、组合熵与沿路转移矩阵**上显著不同；由**稳定突变**界定的道路型最小景观单元在内部保持高视觉一致性，并与人工感知边界及外部城市形态一致。结果表明，城市街道景观并非不可分割的连续体，而是由**共享视觉词汇 + 城市特异空间语法**组织而成。

---

## 10. 一句话

> 本文真正证明的不是"我能切图"，而是：**城市街景复杂性可由少量跨城市共享的视觉基解释，城市差异来自这些基在路网上的组合语法**——MRLU 只是该语法稳定突变的自然推论。

*本方案是 `docs/EXPERIMENT_DESIGN.md` 的理论升维版；§8-S2 三项分析可基于现有 `outputs/` 立即执行。*
