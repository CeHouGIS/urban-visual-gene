# 全球基线：W1024 / K8

## 数据与参数

307 个候选城市：300 个 GoogleSV 城市和上海、杭州、南京、北京、天津、广州、深圳 7 个百度城市，每城目标 3,000 个完整全景。实际缓存 **304 城**；Houston、Dallas、Indianapolis 缺少完整四方向数据。质量剔除后不补抽。

冻结 DINOv3 ViT-L/16，448×448，28×28×1024；四方向环形 context mixing，weight=0.25。城市均衡抽取，共 3,000 万 FP16 patch。SAE 输入 / 字典 / 输出均 1024，BatchTopK K=8；cosine loss、Adam lr=1e-3、60 epochs。实际训练 batch=8192；YAML 及 balanced 守护脚本中的 16384 是早期默认值，以训练产物为准。

百度全景读取时切四个水平视角。共用质量门在 formal/quality/，过滤全黑及隧道，不删除原图。

## 文件与入口

- configs/：城市清单、Google / 百度路径与配额。300_sample 配置是加百度前的过渡记录。
- code/gpu_run.py：缓存、训练与编码；内置默认城市来自 37 城方案，本版本必须指定城市配置。
- code/watchdog_batchtopk_train_after_cache.sh：检查 304 城缓存，用 batch=8192 训练。
- code/watchdog_batchtopk_balanced.sh、watchdog_batchtopk_cache_worker.sh：缓存阶段与早期运行流程。
- code/encode_batchtopk_w1024_k8.py：共享编码入口；旧37城Slurm位于city37_plan。
- runtime/logs/：本版本历史日志；旧混合日志按任务拆分，兼容位置为相对链接。

```bash
python -m formal.gpu_run --help
bash experiments/global307_baseline_w1024_k8/code/watchdog_batchtopk_train_after_cache.sh
```

脚本会启动计算，运行前检查 GPU、NAS 与配置。模型和缓存位于 /host/root/mnt/nas/huangyj/GoogleSV/urban_visual_gene/formal_batchtopk_307city_balanced_3000/；后续 QC 参考同父目录 formal_batchtopk_307city_qc_filtered/。NAS 文件未复制进 Git。尺度、K 消融及 K64 语义结果分列在相邻版本目录。
