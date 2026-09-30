# Urban Visual Gene

城市街景视觉分析代码库。当前论文主方法是 **DINOv3 patch features + BatchTopK sparse autoencoder (SAE)**：冻结 DINOv3 ViT-L/16，在 448×448 图像上得到 28×28 patch grid，再用共享的 W=1024、K=8 BatchTopK SAE 学习可复用的 visual genes。

## 主实验

正式实验固定为 37 个具有代表性的城市。每个城市抽取 12,800 个街景采样点，并获取每个采样点的四个方向（0°、90°、180°、270°）图像，即每城最多 51,200 张方向图像。城市和配额写在 [`configs/formal_city_manifest.json`](configs/formal_city_manifest.json)；当前可复现的 BatchTopK 配置是 [`configs/dinov3_patch_batchtopk_w1024_k8.yaml`](configs/dinov3_patch_batchtopk_w1024_k8.yaml)。

- DINOv3 ViT-L/16，输入 448×448，冻结 backbone，移除 prefix/register tokens；
- BatchTopK SAE，字典宽度 1024，batch-level K=8，decoder columns 做 unit normalization；
- cosine reconstruction loss，Adam，learning rate 1e-3（无 weight decay），batch size 16,384，60 epochs；
- `formal/batchtopk_w1024_k8/summary.json` 是现有 12 城市、14,400 张方向图像的试运行审计结果（不是正式世界城市样本的最终结果）：final train loss 0.154989，reconstruction cosine 0.845011；
- 该试运行中 896 个 latent 被使用、128 个未使用；79 个 universal genes、125 个 single-city genes、121 个 two-city genes 只适用于这次试运行，不能直接外推到正式 30--40 城市结果。

模型实现位于 [`sae_experiments/models/base_sae.py`](sae_experiments/models/base_sae.py)，训练入口位于 [`sae_experiments/training/train_sae_from_features.py`](sae_experiments/training/train_sae_from_features.py)，编码入口位于 [`sae_experiments/evaluation/encode_batchtopk_w1024_k8.py`](sae_experiments/evaluation/encode_batchtopk_w1024_k8.py)。

## 数据位置

街景图像在 NAS：

```text
/host/root/mnt/nas/huangyj/GoogleSV/images/
```

元数据在：

```text
/host/root/mnt/nas/huangyj/GoogleSV/metadata/
```

formal 代码默认使用 `GOOGLE_SV_ROOT=/host/root/mnt/nas/huangyj/GoogleSV`，也可以覆盖：

```bash
export GOOGLE_SV_ROOT=/host/root/mnt/nas/huangyj/GoogleSV
export CUDA_VISIBLE_DEVICES=1,2 # 当前服务器可并行使用 GPU1 和 GPU2
export PYTHONPATH="$PWD"
```

NAS 上已核对 manifest 中 37 个城市都至少有 12,800 个完整四向全景（每个 pano 同时存在 `0/90/180/270` 四张图像）。城市路径映射在 [`formal/gpu_run.py`](formal/gpu_run.py) 中，配额 manifest 是唯一的正式城市来源。

## 运行 BatchTopK SAE

从仓库根目录运行 manifest 驱动的正式 pipeline：

```bash
CUDA_VISIBLE_DEVICES=1,2 python -m formal.gpu_run \
  --cities $(python -c 'import json; print(" ".join(json.load(open("configs/formal_city_manifest.json"))))') \
  --dict-panos 12800 --K-list 1024 --topk 8 --epochs 60 \
  --context-weight 0.25 --sample-json configs/formal_city_manifest.json \
  --skip-infer
```

formal GPU runner 使用共享的 BatchTopK 实现；旧的 per-row Top-K checkpoint 只保留兼容读取能力，不再作为论文主方法。它先分别提取四个方向的 DINOv3 patch，再把四个 28×28 patch 网格按水平环形排列，用局部邻域 context mixer 建模图像边界和 270°→0° 的环绕邻接关系。SAE 输入仍是 1024 维 patch 向量，不使用 4096 维直接拼接。

## 目录

| 路径 | 用途 |
|---|---|
| `sae_experiments/` | DINOv3 特征、BatchTopK SAE 训练和编码 |
| `formal/` | visual gene 统计、层次结构、城市 prevalence、共现和网站资产 |
| `configs/` | 当前实验配置 |
| `results/`、`paper/data/` | 已导出的分析结果 |
| `tests/` | 接口和合成数据测试 |
| `archive/road_mrlu/` | 已归档的道路 MRLU 历史代码，不属于当前论文主流程 |

`scripts/multicity/` 下的早期分析脚本和旧结果仅供历史追溯，不属于当前论文主流程，也不应作为正式结果来源。正式采样时，`formal/gpu_run.py` 的 `--dict-panos` 应设为 `12800`，并保持四个 `HEADINGS` 全部采集；城市列表使用 `configs/formal_city_manifest.json`。

## 测试

若环境已安装 pytest：

```bash
OMP_NUM_THREADS=1 python -m pytest tests/test_dinov3_sae_interfaces.py -q
```

完整街景数据、DINOv3 权重和大规模缓存不提交到 Git。训练和编码需要 PyTorch、Transformers、Pillow、NumPy；具体模块的额外依赖以实际 import 为准。
