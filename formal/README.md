# Formal BatchTopK experiment

`gpu_run.py` 是当前正式实验入口。它对每个采样点的四张街景图像分别提取冻结的 DINOv3 patch 特征，再以环形全景 context mixer 融合空间邻接关系，训练共享的 W=1024、K=8 BatchTopK SAE。正式设计为 37 城市、每城 12,800 个完整四向采样点；城市清单位于 `configs/formal_city_manifest.json`。

## 主要文件

| 路径 | 用途 |
| --- | --- |
| `gpu_run.py` | 可恢复的特征采样、BatchTopK 训练及编码入口 |
| `watchdog_batchtopk.sh` | 当前服务器训练监控与重启脚本 |
| `quality/tiny_qc.py` | 训练前过滤全黑和隧道图像 |
| `quality/qc_model.joblib` | 小型质量分类器权重 |
| `slurm/batchtopk_w1024_k8.sbatch` | 集群训练作业 |
| `slurm/encode_batchtopk_w1024_k8.sbatch` | 集群编码作业 |
| `batchtopk_w1024_k8/` | 早期 12 城市试运行审计结果 |

完整四向街景位于 NAS 的 `/host/root/mnt/nas/huangyj/GoogleSV/`。`gpu_run.py` 默认从该目录读取，可通过 `GOOGLE_SV_ROOT` 覆盖。默认输出目录 `formal/formal_out_panorama_context/` 含训练缓存、恢复点和最终 checkpoint，不提交到 Git。正式训练命令见仓库根目录的 `README.md`。

质量门在 DINOv3 提取前检查四张视图。如果任意视图近乎全黑，或至少两张视图呈隧道特征，就剔除整个采样点。可运行 `python -m formal.quality.tiny_qc --help` 查看训练与校准参数；无可用分类器时会使用确定性规则。

旧的 Feature-MAE、基因层次分析、网站和非 BatchTopK 实验已移至 `archive/non_batchtopk/`；道路 MRLU 流水线位于 `archive/road_mrlu/`。
