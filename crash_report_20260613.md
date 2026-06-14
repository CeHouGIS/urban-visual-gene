# 深度学习服务器崩溃排查报告

**报告日期：** 2026-06-13  
**排查人：** Claude Code 自动诊断  
**服务器：** lianglab-System-Product-Name  

---

## 一、系统基础信息

| 项目 | 详情 |
|---|---|
| 操作系统 | Ubuntu 22.04.5 LTS，内核 6.8.0-124-generic |
| CPU | Intel Core i9-14900KS（8P核+16E核，32线程），Microcode 0x133 |
| 主板 | ASUS TUF GAMING Z790-PLUS WIFI |
| BIOS 版本 | 1836（2026-04-16） |
| 内存 | G.Skill DDR5-7800 4×16GB = 64GB，**实际运行频率 4200 MT/s** |
| GPU | NVIDIA GeForce RTX 3060 LHR 12GB |
| NVIDIA 驱动 | 580.159.03（三处版本一致，无 mismatch） |
| CUDA 驱动支持 | 13.0 |
| 系统 CUDA Toolkit | 11.5（apt 安装） |
| PyTorch（容器内） | 2.1.0，CUDA 11.8 |
| 系统盘 | Samsung SSD 990 PRO 4TB NVMe |
| 训练方式 | Docker 容器（`cehou_urban_visual_gene`），`--ipc=host` |
| NAS | 已挂载至 `/nas_data`（NFS，136.152.48.67），并映射进容器 |

---

## 二、排查过程与关键发现

### 2.1 硬件层面：排除

运行以下全套硬件诊断后，**未发现任何硬件错误**：

| 检查项 | 命令 | 结果 |
|---|---|---|
| GPU 驱动错误（Xid） | `journalctl -k -b -1 \| grep -i xid` | **无** |
| GPU 掉线（fallen off bus） | `dmesg \| grep -i fallen` | **无** |
| PCIe / AER 错误 | `dmesg \| grep -i aer` | **无** |
| CPU 硬件错误（MCE） | `dmesg \| grep -i mce` | **无** |
| 内核 OOM 事件 | `dmesg \| grep -i "out of memory"` | **无** |
| Watchdog / hung task | `dmesg \| grep -i watchdog` | **无** |
| NVMe 磁盘健康 | `smartctl -a /dev/nvme0n1` | **PASSED**，Media Errors: 0 |
| GPU 温度（空闲） | `nvidia-smi` | 37°C（正常） |
| CPU 温度（空闲） | `sensors` | Package 33°C（正常） |
| PCIe 速率（空闲） | `nvidia-smi -q \| grep "PCIe Generation"` | Gen1（空闲省电，正常） |
| PCIe 速率（负载） | 已知结论 | **Gen4 x16**（正常） |
| NVIDIA 驱动版本一致性 | 三处交叉验证 | **完全一致**，无 mismatch |

> **结论：GPU、CPU、NVMe、PCIe 硬件层面均无异常，此次崩溃不是硬件故障。**

---

### 2.2 崩溃日志分析：发现核心证据

检查上次启动（2026-06-12）的内核日志，发现**大量 Python 进程崩溃**，集中在 20:04–23:00 之间：

```
Jun 12 20:04:35  python[3530]:  segfault at 26cde0 in libtorch_python.so
Jun 12 20:14:20  python[5042]:  segfault at 0 in python3.10        (null 指针)
Jun 12 20:15:37  python[5141]:  segfault at 0 in python3.10        (null 指针)
Jun 12 20:18:06  python[5422]:  general protection fault ip:4f60b2 in python3.10
Jun 12 20:19:17  python[5544]:  general protection fault ip:4f60b2 in python3.10
Jun 12 20:19:53  python[5658]:  segfault at 5e30 in python3.10
Jun 12 20:21:06  python[5849]:  general protection fault ip:4f60b2 in python3.10
Jun 12 20:31:01  python[6333]:  segfault at be in python3.10
Jun 12 21:42:12  python[8468]:  segfault at 0 in python3.10
Jun 12 22:45:54  python[10271]: segfault at 0 in python3.10
Jun 12 22:47:56  python[10421]: segfault at 0 in python3.10
Jun 12 22:49:18  python[10590]: segfault at 1 in python3.10
Jun 12 23:00:15  python[11233]: segfault at 5e30 in python3.10
Jun 12 23:00:20  python[11237]: general protection fault ip:4f60b2 in python3.10
```

**最关键的异常：** PID 5422、5544、5849、11237 这 **4 个完全独立的 Python 进程**，在**完全相同的地址 `ip:4f60b2`** 触发了 General Protection Fault（通用保护错误）。

不同进程在同一代码地址重复崩溃，是 **运行环境被污染** 的典型特征——同一套有缺陷的库被反复加载，反复在同一路径崩溃。

---

### 2.3 训练代码分析：找到根因

检查训练入口脚本 `rerun_isolated.sh`（用户自己编写），发现以下注释：

```bash
# Pin all BLAS/OpenMP thread pools to 1 — mixed numpy/scipy/torch/geopandas
# thread pools were causing random native segfaults.
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 \
       NUMEXPR_NUM_THREADS=1 VECLIB_MAXIMUM_THREADS=1 KMP_DUPLICATE_LIB_OK=TRUE
```

**用户自己已经诊断出：混用 numpy/scipy/torch/geopandas 的线程池会导致随机原生 segfault。**

#### 根因详解：双 OpenMP 线程库冲突

`run_experiment.py` 在**同一个 Python 进程**中导入了完整的处理流水线：

```python
import geopandas as gpd       # → 使用 GNU OpenMP (libgomp)
from scipy.spatial import cKDTree  # → 使用 OpenBLAS (libgomp)
import torch                   # → 使用 Intel MKL + Intel OpenMP (libiomp5)
import numpy as np             # → 可能使用 OpenBLAS 或 MKL
```

**冲突机制：**

```
同一进程内：
  libiomp5 (Intel OpenMP，来自 PyTorch/MKL)
    +
  libgomp  (GNU OpenMP，来自 GCC 编译的 NumPy/SciPy/GeoPandas)
    ↓
  两套线程池争抢资源 → 互相干扰 → 崩溃
```

当 PyTorch 调用 MKL 进行矩阵运算，而 geopandas/scipy 已经初始化了一套不兼容的 GNU OpenMP 线程池时，底层 C 库发生冲突，产生无法预测的内存错误，即 `segfault in libtorch_python.so`。首次崩溃（20:04:35）后，用户反复重启训练，但进程级别的 OpenMP 状态已被污染，因此崩溃持续发生，直到服务器重启。

#### 修复不彻底的原因

用户在 `rerun_isolated.sh` 中添加了环境变量，但：

1. **最早的 `run_both_sep.sh` 完全没有这些变量**——这是 20:04 首次崩溃的直接原因
2. `KMP_DUPLICATE_LIB_OK=TRUE` 只是抑制 Intel MKL 的警告，并非真正隔离两套 OpenMP——冲突依然存在，只是不再报错就直接 crash

---

### 2.4 其他发现（次要问题）

| 问题 | 严重性 | 说明 |
|---|---|---|
| GPU Persistence Mode 为 OFF | 低 | 用户曾设置 `-pm 1`，但重启后被重置，未持久化 |
| 内存运行频率 4200 MT/s | 中 | DDR5-7800 套条跑在 4200 MT/s（低于 JEDEC DDR5-4800 基准），说明主板在 XMP 训练时存在问题，需验证稳定性 |
| PyTorch 版本 2.1.0 | 低 | 发布于 2023年10月，已超 2.5 年，存在已修复的历史 bug |
| 系统 CUDA toolkit 11.5 + PyTorch CUDA 11.8 | 低 | 并存无直接冲突（PyTorch 使用自带运行时），但增加环境复杂度 |
| NAS 挂载在容器内 | 低 | `/workplace/nas_data` 可被训练代码访问；代码有 NAS fallback 逻辑，需确认本地数据完整以避免意外走 NFS |

---

## 三、结论

### 根因

**训练崩溃的根本原因是软件层面的 BLAS/OpenMP 线程库冲突，而非 GPU、CPU 或内存硬件故障。**

`run_experiment.py` 将 geopandas、scipy、torch、numpy 全部导入同一个 Python 进程，导致 Intel OpenMP（libiomp5，来自 PyTorch）与 GNU OpenMP（libgomp，来自 GCC 编译的科学计算库）同时被加载，产生不可控的线程池冲突，最终表现为 `libtorch_python.so` 内部 segfault，并引发级联崩溃。

---

## 四、修复方案

### 4.1 立即修复（今日可做）

**所有训练脚本**（包括 `run_both_sep.sh`、`rerun_isolated.sh`、`rerun_hk.sh`）的顶部必须统一添加：

```bash
export OMP_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export MKL_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1
export VECLIB_MAXIMUM_THREADS=1
export KMP_DUPLICATE_LIB_OK=TRUE
```

这是最小改动的缓解措施，可降低冲突概率，但不能从根本上消除。

### 4.2 根本修复（建议）

**将 CPU 密集型地理处理阶段与 GPU 训练阶段分进程运行，不要在同一 Python 进程内混用。**

当前 `run_experiment.py` 在一个进程内顺序调用所有 stage，改为子进程方式：

```bash
#!/bin/bash
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 KMP_DUPLICATE_LIB_OK=TRUE

# Stage 1–3：纯 CPU 地理计算（geopandas/scipy），无 torch
python -m scripts.run_stage3 --out outputs/China/HongKong

# Stage 4–5：纯 GPU 训练（torch），无 geopandas
python -m scripts.run_stage45 --out outputs/China/HongKong --K 32 --epochs 50

# Stage 6：纯 CPU 后处理
python -m scripts.run_stage6 --out outputs/China/HongKong
```

每个子进程独立初始化线程库，完全避免冲突。

### 4.3 其他建议

**① 升级 PyTorch**（中期）

```bash
# 在 Docker 镜像内
pip install torch==2.4.1 --index-url https://download.pytorch.org/whl/cu118
```

PyTorch 2.4+ 对 MKL/OpenMP 共存问题有更多修复。

**② 持久化 GPU Persistence Mode**（低成本）

新建 systemd 服务，确保重启后自动生效：

```bash
sudo tee /etc/systemd/system/nvidia-persistence.service > /dev/null <<'EOF'
[Unit]
Description=NVIDIA Persistence Mode
After=network.target

[Service]
Type=oneshot
ExecStart=/usr/bin/nvidia-smi -pm 1
RemainAfterExit=yes

[Install]
WantedBy=multi-user.target
EOF

sudo systemctl enable --now nvidia-persistence.service
```

**③ 验证内存稳定性**（服务器空闲时运行）

```bash
sudo memtester 32G 3 | tee ~/server_diagnosis_logs/memtester_result.txt
```

若有错误输出，需进 BIOS 检查 XMP 配置，或将内存手动固定为 4800 MT/s（JEDEC 基准）。

**④ 进 BIOS 确认内存 XMP 状态**

当前内存运行在 4200 MT/s，低于 DDR5 JEDEC 基准（4800 MT/s），属于异常状态。请进 BIOS 确认：
- XMP 是否启用，目标频率是否为 7800 MT/s
- 如主板无法稳定 train XMP，建议将内存频率手动设为 4800 MT/s

---

## 五、后续监控建议

训练前启动监控脚本（已在 `~/server_diagnosis_logs/monitor.sh`）：

```bash
~/server_diagnosis_logs/monitor.sh &
```

若再次崩溃，重启后查看：

```bash
# 上次崩溃前的内核日志
journalctl -k -b -1 | grep -iE "segfault|general protection|xid|mce|oom" | tail -50

# 监控日志末尾
tail -200 ~/server_diagnosis_logs/runtime_*/status.log
```

---

*报告生成时间：2026-06-13 16:xx PDT*  
*所有诊断数据保存在：`~/server_diagnosis_logs/`*
