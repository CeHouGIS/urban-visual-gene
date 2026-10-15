# SAE 维度的语义 UMAP：Top 100 激活区域

分析单位是 SAE 维度。逐维检索同一批图像中激活最高的 100 张不同方向图像，裁剪它们的激活区域，使用 SigLIP 图像编码器得到语义 embedding，再汇总成每维向量进行余弦 UMAP。空间位置、城市编号、decoder 向量和原有 64 组聚类不参与投影。语义类别仍需结合图像证据标注。

## 数据来源

当前文章选择的是 30 城市、W=1024、BatchTopK k=8。**该版本的 checkpoint 和逐图激活映射尚未在当前服务器上找到**，不能拿 307 城市 checkpoint 冒充。论文示例图的 provenance 另指向 `/workplace/urban_visual_gene/outputs/experiments/dinov3_multicity/patch_sae_topk32_w512_n30x12800`，记录为 W=512、TopK=32，同样不应视为上述模型的输出。该路径及 `/global/scratch/users/cehou/urban-visual-gene/` 在此环境不存在。

流程要求显式指定 checkpoint、数据集名称和 JSONL 映射，默认验证城市数 30、字典宽度 1024、k=8。每条映射表示一张方向图像：

```json
{"city":"HongKong","pano":"真实panoid","heading":0,"image":"/真实路径/panoid_0.jpg","feature_path":"/真实路径/context_features.f16.npy","row_start":0}
```

`feature_path` 是已经按实际模型预处理方案产生的 contextualized patch 特征，形状 `[patch_count, 1024]`；`row_start` 是这一张图的 784 个 patch 的起始行号。28×28 网格须与图像预处理对应。代码不搜索全量 NAS、不从图像文件顺序猜映射、不再次提取 DINOv3 特征。无已有激活时要求每个 pano 四方向齐全。需要切分的百度全景可提供原图坐标中的 `view_bbox: [left,top,right,bottom]`。

若存在原始稀疏激活 NPZ，可指定 `--sparse-activations`；须包含 `idx`、`val`，形状 `[images,784,slots]`，图像顺序与 JSONL 一致。有 city/pano/heading 元数据时逐项校验。必须包含实际要分析的稀疏激活，不能把固定槽位截断导出误当作完整 BatchTopK 输出。激活 archive 必须与指定模型对应；仅靠其形状不能验证来源。

## 操作定义

- **图像评分**：`max_patch sparse_activation[image,patch,dimension]`。同一图像只占一个名额，取正激活最高的 Top 100；不施加城市或 pano 配额。排名范围限于显式 manifest，不能称为 NAS 全量 Top 100。
- **推理批次**：如果没有旧激活缓存，用完整四视角 pano 的 3,136 个 patch 作为一个 BatchTopK 预算单位。多个 pano 可在 GPU 同时运算，但预算彼此独立。这样吞吐批次大小变化不改变定义；该定义可能与历史批次推理不同，记录在 `run.json`。有原始激活时原样保留其批次效应。
- **裁剪**：以最大激活为峰值，阈值为峰值的 50%，取包含峰值的 8 连通区域，外扩一个 patch，至少 3×3 patch；按比例映射到原始方向图像坐标裁剪。分离的激活区域不被巨大的联合 bbox 合并，数量记录在裁剪清单。它是 patch 级候选区域，不是物体分割 mask。
- **语义 encoder**：`google/siglip-base-patch16-224` 的 `get_image_features`，使用原生 processor，逐 crop 做 L2 归一化。不输入热图颜色、文本类别或位置坐标。保留模型配置、processor、版本和原始裁剪图。
- **维度表征**：每维 Top 100 的 unit embedding 等权平均，再 L2 归一化。平均前向量的模长作为一致性诊断；它不是语义准确率。暂不使用正负原型差，以严格执行本次 Top 100 方案。
- **UMAP**：1024 个维度向量、余弦距离、`n_neighbors=30`、`min_dist=0.1`、seed=42。默认不足 20 个有效实例的维度不投影，明确标记；有 20–99 个实例仍报告实际数量，不补零或重复图像凑数。均值可能隐藏复合语义，因此保留逐图 embedding 和所有实例。

## 运行

依赖：`torch transformers Pillow numpy scipy matplotlib umap-learn`。用户此前授权的 GPU 1/2 可用于推理，无需重新训练 SAE。

```bash
python -m formal.analysis.semantic_feature_umap \
  --manifest /真实路径/30city_images.jsonl \
  --checkpoint /真实路径/30city_batchtopk_w1024_k8.pt \
  --dataset '30-city BatchTopK W1024 k8' \
  --outdir /真实路径/semantic_umap_top100 \
  --device cuda:1 \
  --vlm-model /真实路径/siglip-base-patch16-224
```

可用 `--stage rank|crop|embed|reduce` 分步执行，保持来源/设置一致；完成的阶段复用，embedding 可从完成的 batch 继续。变更 checkpoint、数据、Top 数量、裁剪参数或 VLM 模型须使用新输出目录。仅改变 UMAP 参数可重跑 `--stage reduce`。

结果目录包含：来源及配置 `run.json`，`top_images.jsonl`、`crops.jsonl`、逐图 native crop 和 heatmap PNG、`crop_embeddings.f32.npy`、`feature_semantic_embeddings.f32.npy`、逐维坐标 CSV/NPZ、科学图 PNG，以及可直接打开的 `index.html`。HTML 支持缩放、平移、维度编号检索、点击维度浏览全部裁剪实例。未自动赋予类别标签或发布到网站。

## 已完成的验证（2026-10-12）

7 项测试通过，覆盖逐图 Top K 去重/排序、连通区域裁剪及原图坐标、稀疏槽位重建、推理吞吐 batch 不改变每个 pano 的预算、语义原型汇总和重复映射拒绝。

另用 **307 城市模型的一小批真实图像**验证全部阶段：8 个城市、8 个 pano、32 张方向图像，每维取 Top 5（仅用于流程测试），产生 2,496 个有效裁剪区域；SigLIP 输出 `[2496,768]` embedding，521 个至少有两例的维度成功进入 UMAP。裁剪错误为 0，所有 embedding 和入选坐标均为有限数。此结果不能作为 30 城市 Top 100 的研究结果。

测试目录：`/tmp/semantic_umap_smoke/results/`。已校验 SHA256 的 SigLIP 模型位于 `/root/.cache/urban_scene_models/siglip-base-patch16-224/`。正式 30 城市分析仍待原版 checkpoint 和映射到达后运行。

## 按用户要求测试最新模型（2026-10-12）

用户随后指定使用最新模型测试。选择最后完成的 W=4096、BatchTopK k=64 checkpoint：`/host/root/mnt/nas/huangyj/GoogleSV/urban_visual_gene/formal_batchtopk_307city_w4096_k64/sae_448_k4096.pt`。候选数据集为 307 城市，实际训练缓存保留 304 城市、3,000 万 patch；本次语义证据使用已有 QC 核验及 SAE 训练排除点位：295 城市 × 4 pano × 4 方向 = 4,720 张图。上海百度全景通过训练时的原生四视角裁剪坐标接入。

没有重新提取 DINOv3。`prepare_semantic_umap_manifest.py` 仅读取已有点位、图片清单及 QC 元数据，核验旧缓存的 pano 顺序、完整四视角、每城特征行数及新版 QC 排除项。逐图使用完整 784 patch，不限于以前尺度实验每 pano 的 128 个 anchor。

运行命令：`bash formal/analysis/run_latest_semantic_umap.sh all`；输出目录 `reports/data/semantic_umap_w4096_k64_top100/`。每维仍取 Top100；使用 8 个 CPU worker 保存裁剪、4 个线程预处理图像，GPU1 上以 BF16 autocast 执行 SigLIP，归一化及最终 embedding 均为 FP32。精度核验单独记录于 `precision_validation.json`。大量可再生图像及 embedding 目录已列入 `.gitignore`，避免意外上传数十万个文件。

9 项测试及最新模型真实图像完整流程验证通过。新增测试覆盖百度原生四视角坐标、QC 排除和输入 manifest 位于结果目录时的续跑校验；输入文件不会再被规范化写回改变 SHA256。实际完成情况以结果目录的 `test_summary.json`、`embedding_run.json`、`reduction_run.json` 为准。
