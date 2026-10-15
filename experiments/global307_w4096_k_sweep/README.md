# K 消融：W4096，K=16/32/64

固定全球系列 304 城市、同一 FP16 3,000 万 × 1024 缓存、字典 W4096，仅改变 K16/32/64。默认 60 epochs，batch8192，复用 formal.gpu_run.train_sae 的 cosine loss 流程。这里的 K32 与当前 30 城主实验的数据和目标不同。

- code/train_expanded_sae.py：检查缓存 / QC 身份并训练，支持 --width、--k。
- code/run_expanded_sae_ksweep.sh：可恢复多任务脚本，运行前检查物理 GPU 编号。
- code/build_expanded_sae_gallery.py：四方向图与激活图库。
- reports/gallery/：本机完整 K16/K32/K64 图库；原 reports/sae_4096/ 为链接。
- results/：已存在的结果摘要；完成状态按各自产物记录。

```bash
python -m formal.analysis.train_expanded_sae --help
python -m formal.analysis.build_expanded_sae_gallery --help
```

模型位于 NAS 的 formal_batchtopk_307city_w4096_k16/、..._k32/、..._k64/。K64 checkpoint 的 Top100 语义分析单列在 [global307_w4096_k64](../global307_w4096_k64/README.md)。
