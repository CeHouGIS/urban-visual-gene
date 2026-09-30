# Urban Visual Gene

本仓库当前维护的实验是 **DINOv3 + 四向全景 patch context + BatchTopK SAE**。其他研究线已归档，见下文。

## 当前实验

- 37 个代表性城市，配置见 [`configs/formal_city_manifest.json`](configs/formal_city_manifest.json)；每城目标为 12,800 个采样点，每点采集 0°、90°、180°、270° 四张图像。
- 冻结 DINOv3 ViT-L/16，分别处理四张 448×448 图像，取得各自的 28×28 patch 特征。
- 将四个 patch 网格按水平方向组成环形全景，并用局部 context mixer 处理相邻视角及 270°→0° 的接缝；SAE 接收 1024 维 patch 特征。
- 共享 BatchTopK SAE：字典宽度 W=1024，batch 平均激活数 K=8。正式训练使用 60 epochs。

主入口是 [`formal/gpu_run.py`](formal/gpu_run.py)，模型和特征组件在 [`sae_experiments/`](sae_experiments/)，质量筛选器在 [`formal/quality/`](formal/quality/)。`formal/batchtopk_w1024_k8/` 保存早期 12 城市试运行的审计结果，不能当作正式 37 城市结果。

## 数据与运行

街景图像和元数据默认读取 NAS 的 `/host/root/mnt/nas/huangyj/GoogleSV/`，可用 `GOOGLE_SV_ROOT` 覆盖。DINOv3 权重可通过 `DINO_MODEL_PATH` 指定。当前服务器直接运行时可让 GPU 1、2 同时参与特征提取：

```bash
export CUDA_VISIBLE_DEVICES=1,2
export PYTHONPATH="$PWD"
python -m formal.gpu_run \
  --cities $(python -c 'import json; print(" ".join(json.load(open("configs/formal_city_manifest.json"))))') \
  --dict-panos 12800 --K-list 1024 --topk 8 --epochs 60 \
  --context-weight 0.25 --sample-json configs/formal_city_manifest.json \
  --skip-infer
```

训练输出和恢复点默认位于 `formal/formal_out_panorama_context/`，该目录不提交到 Git。更多参数与质量筛选说明见 [`formal/README.md`](formal/README.md)。

## 目录

| 路径 | 内容 |
| --- | --- |
| `formal/` | BatchTopK 正式训练入口、质量筛选、Slurm 作业和试运行审计 |
| `sae_experiments/` | DINOv3 特征提取、BatchTopK 模型、训练和编码 |
| `configs/` | BatchTopK 配置与正式城市清单 |
| `tests/` | 当前 BatchTopK 接口与全景 context 测试 |
| `archive/non_batchtopk/` | Feature-MAE、图视觉词汇、图原型、旧城市分析、网站及历史结果 |
| `archive/road_mrlu/` | 道路 MRLU 历史流水线 |

旧实验仍保留原有相对目录结构，方便追溯；归档中的脚本不再作为当前实验入口。`configs/dinov3_batchtopk_k16_width16_seed42.yaml` 是早期 BatchTopK 草案配置，正式运行以 `formal/gpu_run.py` 参数和城市清单为准。

## 验证

```bash
python -m pytest tests/test_panorama_context.py tests/test_dinov3_sae_interfaces.py -q
```
