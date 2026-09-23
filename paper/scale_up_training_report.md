# Feature-MAE Scale-up 训练报告

**版本日期：** 2026-09-23

**报告状态：** 正式训练已启动

**关联计划：** `paper/scale_up_experiment_plan.md`

## 1. 当前基线

当前正式模型使用质控后的30城数据训练：

```text
source images: 384,000
retained images: 378,818
dropped images: 5,182
latent width: 512
mask ratio: 75%
encoder depth: 4
decoder width/depth: 256 / 2
best epoch: 10
training loss: 0.103563
validation loss: 0.103343
recorded peak GPU memory: 3.27 GiB
```

当前checkpoint：

```text
outputs/experiments/dinov3_multicity/
feature_mae_n30x12800_qc/mae/model_best.pt
```

该模型可作为现有论文结果的外部锚点，但严格的scale-up比较使用重新划分且排除公共评价集的A10000/B512模型作为受控参考。

## 2. 服务器资源快照

启动前检测结果：

| 资源 | 当前配置 | 启动前状态 |
|---|---|---|
| GPU | NVIDIA GeForce RTX 3060, 12,288 MiB | 约11,780 MiB空闲 |
| RAM | 62 GiB | 约59 GiB available |
| Swap | 2 GiB | 未使用 |
| `/workplace` | 3.6 TiB | 约3.0 TiB可用 |
| CPU约束 | 使用0--7、10--15 | 必须避开CPU 8和9 |

主要风险不是磁盘，而是W1024的GPU显存峰值、30城memmap随机读取造成的host page cache增长，以及长任务中断后丢失当前epoch进度。

### 2.1 启动记录

```text
started_utc: 2026-09-23T15:15:06Z
runner status: running
formal tasks: 21
current task: scale_n10000_w0512_s42
execution mode: sequential, one GPU task at a time
```

按 `panoid` 隔离的数据划分已经生成并通过审计：30个城市均包含500、1,000、2,000、4,000、8,000和10,000张嵌套训练子集；训练、验证和公共评价集之间的全景重叠为0。W128、W256、W512和W1024均已完成前向、反向、验证、checkpoint写入和断点恢复smoke test。

smoke test记录的峰值显存分别约为：

```text
W128:  2.31 GiB
W256:  2.60 GiB
W512:  3.27 GiB
W1024: 2.08 GiB (micro-batch = 60)
```

## 3. 正式训练任务清单

计划共21个不重复正式任务：

| 模块 | 设置 | 数量 | 当前状态 |
|---|---|---:|---|
| 样本量单种子曲线 | W512; n=500, 2k, 8k; seed42 | 3 | pending |
| 样本量三种子节点 | W512; n=1k, 4k, 10k; seeds42--44 | 9 | pending |
| 宽度实验 | n=10k; W128, W256, W1024; seeds42--44 | 9 | pending |
| 共用参考 | n=10k; W512; seeds42--44 | 已包含在上面 | pending |

所有任务必须拥有唯一 `experiment_id`，例如：

```text
scale_n01000_w0512_s42
scale_n10000_w0128_s43
scale_n10000_w1024_s44
```

## 4. 防止服务器崩溃的训练策略

### 4.1 串行调度

- 同一时间只允许一个Feature-MAE训练任务使用GPU；
- 使用文件锁防止重复启动；
- seed 42阶段完成并通过审计后，才启动seed 43和44；
- 不同时运行UMAP、全量热图渲染或其他高RAM任务。

### 4.2 CPU和内存约束

所有命令使用：

```bash
taskset -c 0-7,10-15 <command>
```

并限制数据加载线程，避免底层BLAS或OpenMP误用CPU 8和9。训练继续使用只读memmap、排序批内索引和周期性 `MADV_DONTNEED`，防止30个城市的token文件将page cache持续推高。

资源警戒线：

```text
RAM available < 8 GiB: warning and stop scheduling next run
RAM available < 4 GiB: save checkpoint and terminate current run cleanly
swap usage > 1 GiB: investigate before continuing
disk free < 500 GiB: stop creating new checkpoints
```

### 4.3 GPU显存控制

先使用以下保守起始micro-batch进行smoke test：

| Latent width | 起始 micro-batch | 策略 |
|---:|---:|---|
| 128 | 240 | 若稳定可保持 |
| 256 | 240 | 若稳定可保持 |
| 512 | 120--240 | 以当前3.27 GiB记录为参考 |
| 1024 | 60 | 通过梯度累积保持有效batch |

最终micro-batch由实测显存决定，不把micro-batch大小当作科学变量。所有宽度保持相同有效batch和optimizer update数。

```text
target peak GPU memory: <= 10.5 GiB
temperature warning: >= 80 C
temperature stop threshold: >= 84 C
CUDA OOM: do not blind retry; halve micro-batch and restart from checkpoint
```

W1024必须先运行短程测试，确认前向、反向、验证和checkpoint写入全部成功后才能进入正式训练。

### 4.4 Checkpoint与恢复

现有训练程序已具备：

- `checkpoint_last.pt` 自动恢复；
- `model_best.pt` 最佳验证模型；
- 临时文件写完后原子替换；
- epoch级历史记录；
- validation early stopping；
- memmap page-cache释放。

正式scale-up运行前建议增加：

1. 每250个batch保存一次轻量step checkpoint；
2. 保存epoch内batch位置和随机数状态；
3. checkpoint写入后进行可读性校验；
4. 每个run保存独立日志，禁止多个配置共用输出目录；
5. 保留 `last` 和 `best`，旧step checkpoint最多保留两个。

如果暂不实现epoch内恢复，则每个epoch必须控制在可接受时长，并明确记录崩溃后最多重算一个epoch。

## 5. 启动前检查

每个任务启动前必须完成：

```text
[ ] filtered manifest和split文件存在
[ ] 30个城市均达到目标样本量
[ ] train/validation/evaluation按panoid无重叠
[ ] heading分布审计通过
[ ] GPU无其他计算任务
[ ] available RAM >= 8 GiB
[ ] disk free >= 500 GiB
[ ] 输出目录不存在有效完成标记
[ ] 若存在checkpoint，其config与当前run完全一致
[ ] experiment registry写入planned/running状态
```

配置不一致时不得自动加载旧checkpoint。

## 6. Smoke test协议

每个宽度依次执行：

1. 2个训练batch和1个验证batch；
2. 保存并重新加载checkpoint；
3. 运行一次恢复训练；
4. 检查loss为有限值、梯度无NaN；
5. 记录GPU峰值、RAM峰值和batch耗时；
6. 再运行一个完整epoch；
7. 通过后锁定该宽度的micro-batch配置。

128、256、512、1024四个宽度必须分别测试，不能根据512维结果直接推断1024维安全。

## 7. 监控与状态记录

每个run至少记录：

```text
experiment_id
data split hash
sample size per city
latent width
random seed
effective batch size
micro-batch size
gradient accumulation
optimizer steps
train and validation loss
best epoch
wall-clock time
peak GPU memory
peak RAM or minimum available RAM
checkpoint path
exit code
resume count
software and CUDA versions
```

状态只能取：

```text
planned -> smoke_passed -> running -> completed
                              |-> interrupted -> resumed
                              |-> failed
```

“进程结束”不等于“训练完成”。只有最佳checkpoint可加载、历史行数正确、评价任务成功且完成标记写入后，状态才能设为 `completed`。

## 8. 崩溃恢复流程

### CUDA OOM

1. 保存错误日志和显存峰值；
2. 确认没有第二个GPU进程；
3. micro-batch减半；
4. 调整梯度累积以保持有效batch；
5. 从最后一个有效checkpoint恢复；
6. 将调整记录到registry，不覆盖原始配置记录。

### Host OOM或服务器卡顿

1. 停止启动新任务；
2. 检查memmap page cache和其他进程；
3. 降低每次跨城市读入量；
4. 提高cache释放频率；
5. 必要时将任务拆为更短epoch段；
6. 不使用强制删除checkpoint的方式“清理”。

### 断电、SSH断开或进程意外退出

训练必须运行在可恢复的后台会话中。重新连接后先验证checkpoint，再恢复同一 `experiment_id`；不得新建目录冒充继续运行。

## 9. 训练结果表模板

| experiment_id | n/city | width | seed | best epoch | train loss | val loss | GPU GiB | hours | resumes | status |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|
| scale_n01000_w0512_s42 | 1,000 | 512 | 42 | -- | -- | -- | -- | -- | 0 | pending |
| scale_n10000_w0128_s42 | 10,000 | 128 | 42 | -- | -- | -- | -- | -- | 0 | pending |
| scale_n10000_w0256_s42 | 10,000 | 256 | 42 | -- | -- | -- | -- | -- | 0 | pending |
| scale_n10000_w0512_s42 | 10,000 | 512 | 42 | -- | -- | -- | -- | -- | 0 | pending |
| scale_n10000_w1024_s42 | 10,000 | 1,024 | 42 | -- | -- | -- | -- | -- | 0 | pending |

正式运行后，该表由registry自动生成，不手工填写训练数值。

## 10. 完成标准

训练阶段完成需同时满足：

```text
21个正式任务均completed，或失败任务有明确排除理由
所有最佳checkpoint可以独立加载
公共评价集上的激活已生成
E+D+P层级已生成
32/64类匹配已完成
样本量与宽度稳定性表已导出
服务器异常和所有恢复事件已记录
大型模型与缓存未进入paper目录或GitHub
```

在上述条件满足前，报告不得将计划值或部分运行结果表述为最终科学结论。
