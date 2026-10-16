# 多尺度卷积景观原型实验

在冻结的主实验 SAE 上，学习可在不同街景、不同位置重复使用的局部激活模板。模板同时包含 SAE 维度组合和相对空间布局。**96 是第一轮候选容量，不代表已经发现 96 种景观。**

本目录独立保存代码、缓存、模型和分析；现有 SAE、Top100、逐维 VLM、论文和网页结果继续保留。

## 数据与身份

- 复用 `../vitl30_panorama_w4096_k32/outputs` 的 panorama-aware DINOv3 ViT-L/16 缓存，以及 W4096/K32 冻结 SAE。每方向特征是 `28×28×1024`，不重复提取 DINO 特征或训练 SAE。
- 29 个原图映射已经核对的城市，**86,038 个完整全景 / 344,152 张方向图像**。Moscow 的原图映射未验证，暂不使用。不是仅使用原有 3,712 个验证全景。
- 对每个全景独立执行 BatchTopK：`4×784=3136` 个 patch 共用 `3136×32` 的激活预算。各 patch 的非零数量可以不同，批处理不跨全景竞争预算。
- 保存所有正激活，逻辑形状为 `4×28×28×4096`；按 patch 存 CSR，索引 uint16、值 float16。推理与 TopK 选择是 FP32，关闭 TF32；浮点压缩存在小幅量化误差。
- SAE checkpoint SHA256：`fccf7c29fea802341802e365f07fa3bdb4077203ba784b03f046ac89823113e0`。
- 本轮语料身份：`e6439911655396bf8eddaada805851f65339f974e5df799057c0827dc08e734a`。完整城市与来源记录在 `outputs/corpus.json`。

每城按 1 km 空间块做固定哈希划分，同一全景四方向始终同组：**68,685 训练全景 / 17,353 验证全景**。逐维尺度使用原型训练块中正激活的 RMS，之后截断到 8。

这个划分检验的是新增的原型层；冻结 SAE 的旧训练集与原型验证集有交集，不能据此声称整个 DINO→SAE→原型体系在从未见过的数据上独立验证。清单保留 `sae_split`。

## 模型与训练

输入为某方向的完整 SAE 激活图 `4096×28×28`。方向间已有 panorama context，但本层卷积在各方向内部进行，不把方向边界当作连续像素，也不宣称检测了跨接缝单元。

三个尺度：3×3、7×7、15×15，每尺度 32 个模板。为控制计算量，每个非负模板采用秩 4 分解：

```text
B[m,d,y,x] = Σ_r channels[m,r,d] × layout[m,r,y,x]
C = CNN([observed_activation, explicit_hidden_mask])
A_hat = Σ_m transposed_convolution(C[m], B[m])
```

编码器：4097→64 的 1×1 卷积、按尺度的 depthwise 卷积、各尺度 32 通道的 ReLU 响应。解码器用分组转置卷积和通道混合实现上述模板叠加。模板非负、完整 L2 范数约束为 1；秩 4 是表达能力约束，需要在后续容量实验中检验。

AdamW，8 轮，batch=8，初始学习率 0.001，梯度裁剪、余弦学习率、BF16。各城市均衡访问全景，少样本城市重复采样；第 `epoch % 4` 个方向进入该轮，8 轮中每方向访问两次。每次读入该方向的全部 784 个 patch。

目标为 `positive_MSE + 32×zero_MSE + 0.01×response_mean + 0.02×channel_L1`，正值和零值误差各自按元素数量归一化。零值权重经过独立小样本训练行为检查调整，避免弥散的假阳性响应。前两轮不遮挡，之后随机隐藏 20% patch，并以权重 1 加入隐藏位置的正激活恢复误差；显式 mask 通道区分被隐藏位置和真实零值。实际权重、完整配置与代码身份保存于 `outputs/models/*/config.json`。

固定每城 16 个验证全景、全部四方向用于每轮监控和选择 checkpoint；训练结束后评估**全部 17,353 个验证全景的四方向**。

## 第一轮模型和对照

| Run | 数据 | 用途 |
| --- | --- | --- |
| `real_seed42` | 原激活 | 主候选模板库 |
| `real_seed43` | 原激活 | 随机种子稳定性 |
| `spatial_shuffle_seed42` | 每方向共享一个 patch 位置排列，所有维度一起移动 | 保留各 patch 的维度组合，打乱空间结构 |
| `feature_shuffle_seed42` | 同城、同方向的 batch 内，每维独立交换完整激活图 | 保留各维度图分布，打乱维度间场景配对 |

四组模型容量、访问预算及训练策略相同。小 batch 的交换会保留一部分原配对，是有限范围的条件随机化，不能称为全局独立检验。

另外用相同标准化/截断后的全景平均激活拟合 96 分量 MiniBatchNMF。NMF 仅拟合全景向量，**只比较全景向量重建误差**，不将其与 CNN 的局部 patch 图重建误差混比。

## 输出和可核查证据

完整验证记录全局误差的分子分母、每城和每空间块统计、原型使用率及无响应模板。城市均衡效果给出城市 bootstrap 区间，不能把同全景四张图当独立样本。

固定原模型后，在每城另取 16 个验证全景（排除选择 checkpoint 时监控的全景）进行两类各 39 次置乱测试；保存每次结果和检验范围。加一 Monte Carlo p 值的最小分辨率为 `1/40=0.025`，不用于宣称高精度显著性。

原型实例提取采用每模板、每方向的局部峰值与非极大值抑制，最多保留两个半峰值以上的候选中心；不同模板允许共同出现。图库优先选不同城市、不同全景，保存：

- 原图预览、原生分辨率裁剪、清晰的贡献热图和二值遮挡图；
- 模板组成的 SAE 维度、响应位置、空间范围；
- 删除该位置的一次模板响应后 SSE 的有符号变化。负值表示删除反而改善重建，不应把响应高直接等同于有效解释；
- 两个训练种子在同尺度内通过 Hungarian 匹配后的完整模板 cosine、Top20 维度与代表全景重叠。

Qwen3-VL-2B 在原型训练及实例图完成后，对每原型最多 8 个不同全景的裁剪给出共同组成、布局、异类与不确定性。它只看图像，不把名称作为模型输入。不明确、混合、光照或相机伪影可以原样记录。候选名称没有独立语义准确率。

```text
outputs/
  corpus.json, panoramas.parquet     数据、原图映射、空间块与旧/新 split
  activations/<city>/<chunk>/        完整稀疏激活，原子完成标记
  positive_rms.npy                   仅训练块校准
  models/<run>/                     last/best、训练曲线原始记录、每轮验证
  evaluation/                       完整验证、NMF、置乱与跨种子分析
  gallery/index.html                每原型真实实例和热图
  interpretation/index.html         原型级 Qwen 候选解释
  pipeline.log, pipeline_status.json 后台流程与当前阶段
```

## 运行、查看和恢复

在仓库根目录运行。按当前要求使用 GPU 0、2，每卡同时运行一组原型实验，两批完成四组；其他已有任务继续运行。

2026-10-16 修复了稀疏重建损失中 `masked_select` 在 CUDA 上生成大规模临时索引的问题，改为保留形状的掩码归约。目标函数、正/零激活归一化和梯度保持一致，已用数值与梯度对照验证。实际 GPU 0 上 batch=8 的六步检查覆盖三种数据和遮挡阶段，显存峰值为 820 MiB（PyTorch reserved，不包含 CUDA 上下文）。关闭 cuDNN 自动调优，并将每进程 PyTorch 缓存上限设为 1280 MiB，保留原 batch=8 和全部模型设置。

启动前需有至少 1600 MiB 空闲显存；显存不足时排队。训练期间若发生 CUDA OOM，等待后从最近的检查点重试。阈值分别通过 `UVG_PROTOTYPE_MIN_FREE_MIB`、`UVG_PROTOTYPE_MEMORY_MIB` 调整。全量缓存通过身份、分块大小和标准化校验后直接复用，恢复流程不会重新加载冻结 SAE。原型 Qwen 阶段需 7000 MiB 空闲显存，会单独等待，不影响已完成的训练和评估。

这次兼容迁移的旧代码、配置、检查点和身份变更记录保存在本地 `outputs/recovery_20261016/`；只更新经过数值/梯度验证的损失实现和显存运行设置，保留模型权重、优化器状态、采样和步数。

```bash
# 当前后台进程存在时，不要重复启动；flock 会阻止同目录重复流水线
python -u -m experiments.vitl30_multiscale_prototypes.code.run --stage all --devices 0,2

# 只读状态
python -m experiments.vitl30_multiscale_prototypes.code.run --stage status

# 单独恢复分析或原型解释
python -m experiments.vitl30_multiscale_prototypes.code.run --stage evaluate --device cuda:0
python -m experiments.vitl30_multiscale_prototypes.code.run --stage interpret --device cuda:2

python -m pytest -q experiments/vitl30_multiscale_prototypes/tests
```

全流程依次执行准备→双 GPU 冻结 SAE 推理→训练块标准化→四组原型训练→完整验证/图库/对照→原型 Qwen。稀疏激活按 chunk 恢复，训练每 100 步及每轮保存，置乱逐次保存，VLM 逐原型保存。错误在日志和 `pipeline_status.json` 明示；修复后同命令恢复。配置、数据或训练代码身份改变时拒绝混用旧 checkpoint。

预计稀疏缓存约 35 GB；各大文件留在本地 ignored 的 outputs。依赖沿用项目的 PyTorch、pandas/pyarrow、SciPy、scikit-learn、Pillow、matplotlib、transformers、joblib、threadpoolctl。这里没有全盘扫描 NAS，也没有迁移或删除旧分析。
