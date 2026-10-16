# 实验版本索引

目录按数据和 SAE 参数隔离。global307 系列是较早的全球实验，vitl30 是当前主实验；旧论文 30 城版本另行留档。

| 版本 | 输入 D / 字典 W / 稀疏预算 K | 数据 | 研究目的 |
| --- | --- | --- | --- |
| [主实验](vitl30_panorama_w4096_k32/README.md) | 1024 / 4096 / 32 | 30 城市、89,031 全景 | panorama-aware 视觉单元、语义与组合 |
| [当前局部要素组合](vitl30_local_compositions/README.md) | 冻结主 SAE；四方向自适应区域 | 29 城、1,392 训练＋464 开发全景 | 跨街景重复关系的探索性检索、置换对照及实例遮罩 |
| [要素关系原型](vitl30_relational_prototypes/README.md) | 冻结主 SAE；14/21/28区域；32槽位 | 29城、1,160全景小试 | 当前路线：要素表、区域输入、三组小试完成；关系图开发误差高于完整布局，语义待核查 |
| [多尺度景观原型](vitl30_multiscale_prototypes/README.md) | 冻结主 SAE；秩 4 卷积模板 | 29 城市、86,038 全景；空间块划分 | 局部语义组合、模板稳定性、置乱及 NMF 对照 |
| [原型覆盖率试验](vitl30_prototype_coverage/README.md) | 冻结主 SAE；每尺度32模板 | 同29城；新训练60,233／开发12,755／封存测试13,050全景 | 响应优化诊断、两组4轮试训及语义覆盖验证准备 |
| [逐图80% patch覆盖](vitl30_patch_coverage/README.md) | 冻结主 SAE；每尺度32模板 | 同29城；复用上述划分和训练集校准 | 按每张方向图惩罚patch覆盖缺口，固定NMSE≤0.5，报告实际达标比例与未解释区域 |
| [全球基线](global307_baseline_w1024_k8/README.md) | 1024 / 1024 / 8 | 307 候选，304 城缓存 | 大范围城市均衡训练 |
| [字典尺度](global307_dictionary_scaling_k8/README.md) | 1024 / 512,1024,2048 / 8 | 全球系列 | 稳定、分流、未匹配关系 |
| [K 消融](global307_w4096_k_sweep/README.md) | 1024 / 4096 / 16,32,64 | 全球系列 | 固定宽度，改变稀疏预算 |
| [W2048/K64](global307_w2048_k64/README.md) | 1024 / 2048 / 64 | 全球系列 | 语义 UMAP、共现与空间显著性 |
| [W4096/K64](global307_w4096_k64/README.md) | 1024 / 4096 / 64 | K 消融的 K64 模型 | Top100 图像语义 UMAP |
| [早期论文 30 城市](legacy30_paper_w1024_k8/README.md) | 1024 / 1024 / 8（论文记载） | 论文原有 30 城 | 历史文字与图片 |
| [早期 12 城市](legacy12city/README.md) | 以历史记录为准 | 12 城 | 图片留档 |
| [37 城市计划](city37_plan/README.md) | 以计划为准 | 每城目标 12,800 点 | 方案留档 |
| [语义试运行](semantic_pipeline_smoke/README.md) | 以 run.json 为准 | 8 城、32 方向图 | 流水线验证 |

## 版本关系

- 字典尺度、K 消融和两套 K64 语义分析共用较早的 304 城市、3,000 万 patch 缓存。共用训练器不复制。
- W4096/K64 语义分析引用 K 消融同一 checkpoint，只单列分析产物。
- 主实验是新的 30 城缓存身份，使用 squared-L2 loss；全球系列用 cosine loss，不能直接横向比较 loss 数值。
- 同一 feature ID 只对指定 checkpoint 有效，结果清单中的 SHA256 用于核对来源。

## 保存规则

code/、configs/、tests/、README 与 results/ 供 GitHub 查看。outputs/、完整 reports/ 和 runtime/ 是服务器工作目录，只提交说明或选出的静态报告。NAS 原图及全量特征原位保留。

历史模块名通过包搜索路径定位到各版本 code/，旧配置与报告目录是相对符号链接。请在实体目录修改，避免保存重复版本。
