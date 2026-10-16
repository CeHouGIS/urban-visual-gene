# 从景观要素到关系原型

本实验在冻结的主 SAE（DINOv3 ViT-L、W4096/K32）之上，寻找跨图像重复的**多要素组成与布局**。较早的 3/7/15 patch 模板继续作为重建参考，不能直接解释为景观类型。

## 当前完成：第一步要素表

复用同一 SAE 的 masked Top10 Qwen 解释，并仅为此前没有解释的维度补入扩样解释。未遮罩 Top100 解释用于发现父类冲突，不作为真值，不覆盖 masked 解释。没有重提 DINO 特征、重训 SAE 或运行新 VLM。

| 候选状态 | 维度数 |
| --- | ---: |
| 可进入探索性组合分析的暂定候选 | 1,561 |
| 背景、材料或上下文 | 1,048 |
| 待核查 | 503 |
| 实例证据不足（少于 5 例） | 522 |
| 当前仍无正激活证据 | 462 |
| 合计 | 4,096 |

已有解释 3,634 维：原 masked 解释 3,178 维，扩样新增 456 维。426 维存在遮罩与未遮罩解释的父类分歧，优先排入复核；分歧不等于 masked 解释错误。当前全部维度均未通过独立人工语义验证，VLM 自述信心不作为准确率。

`taxonomy.json` 保留细标签，以明确名称归并对象父类：窗户、门及建筑墙面归入建筑，树冠和树干归入乔木，车窗和车轮归入车辆。依附对象不明的墙面、地面和泛指植被不强行归入明确对象；不从解释中偶然提及的对象猜测类别。暂定候选涉及 15 类要素。这是输入组织，不是发现了 15 种景观原型，也不是实例分割。

输出：

- `results/hierarchy.json`：4,096 维完整映射、原始解释、质量标记及来源身份。
- `results/dimension_elements.csv`：完整要素表。
- `results/review_queue.csv`：优先处理冲突的复核表。
- `results/summary.json`：状态计数和主模型身份。
- 网站 `/elements/`：逐维检索与实际 masked 输入拼图；冲突维度展示两套图像证据。

## 小试数据与第一轮训练

沿用 `vitl30_prototype_coverage` 的 1-km 空间块划分与训练集 RMS，不重新划分。每城固定选 32 个训练与 8 个开发全景；29 城共 928 训练、232 开发全景，即 4,640 张方向图。同一全景四方向不拆组。Moscow 原图映射未核实，暂不纳入。此步只读取名单和状态，没有载入激活或打开测试图像。

`outputs/protocol.json` 与 `outputs/pilot_panoramas.parquet` 保存服务器侧固定名单，公开摘要为 `results/pilot_manifest_summary.json`。代码和输入哈希参与身份计算，身份变化时须使用新输出目录，避免覆盖旧结果。

从原有完整 SAE 稀疏激活构建每方向 14×14、21×21、28×28 区域的 4×4 要素强度布局，四方向以本方向、右转90°、对向、左转90°顺序提供全景上下文。每个候选父类的激活先除以新的训练块 RMS、截断，再以该父类的候选维度数平方根调整；只使用暂定合格的 1,561 维。这样得到 **64,960 个区域**，训练前仅保留至少两个不同父类都有明显响应的区域：**24,113 训练、5,817 开发**。其余区域仍计入分母，不可把该小试的成绩外推到全体 patch。

在 GPU 1 上以相同划分、32个离散槽位、seed42、10轮分别训练：仅看组成强度、看完整区域布局、不额外使用图消息的模型，以及完整布局加要素质心邻近关系图消息的模型。三组都使用四方向上下文，共用同一目标（候选父类 4×4 布局重建）。模型和开发评分在 `outputs/models/`；来源、槽位使用情况及空间块配对区间见 `results/relational_pilot_summary.json`。

| 训练臂 | 开发MSE，越低越好 |
| --- | ---: |
| 仅组成强度 | 0.2311 |
| 完整区域布局 | **0.2121** |
| 区域布局＋邻近关系图 | 0.2166 |

关系图减完整区域模型的配对开发MSE为 **+0.00455**，按218个空间块自助重采样的95%区间为 **[+0.00299, +0.00619]**。这轮图结构没有改善重建；完整布局相对只看组成有收益。32个槽位在开发集都有响应，但仅表示被使用，不表示32种不同且有意义的景观。**原定的同预算局部模板重训尚未完成**；旧模型仅作描述性参考，不能与新三组直接比较优劣。

## 后续按顺序实现

1. 核查候选父类及混合维度；保留原始 4,096 维，不删除不确定项。
2. 已完成：从已有完整稀疏激活构建每方向 14×14、21×21、28×28 区域；四方向按朝向提供上下文，不跨透视图接缝做局部卷积。
3. 已完成：32槽位离散字典的组成、布局、关系图三组探索小试；关系图未超过不用图消息的布局模型。
4. 待完成：核查各槽位真实图像、要素删除与布局置乱敏感性，排查错误或混合父类；再做跨种子比较，并以同预算重训局部模板。
5. 待完成：比较 16/32/64 槽位，跨城检索和独立语义验证；容量不是景观数量。

原有“每张图至少 80% 有效 patch 达到 NMSE≤0.5”的重建覆盖目标继续单独报告。景观组合覆盖须另行定义并独立核查，不能因为一个大区域匹配成功，就把框内全部 patch 算作已解释。

## 复现第一步

```bash
python -m experiments.vitl30_relational_prototypes.code.hierarchy
python -m experiments.vitl30_relational_prototypes.code.prepare
python -m experiments.vitl30_relational_prototypes.code.regions
python -m experiments.vitl30_relational_prototypes.code.train --arm composition --device cuda:1
python -m experiments.vitl30_relational_prototypes.code.train --arm context --device cuda:1
python -m experiments.vitl30_relational_prototypes.code.train --arm relation --device cuda:1
python -m experiments.vitl30_relational_prototypes.code.evaluate --device cuda:1
python -m pytest -q experiments/vitl30_relational_prototypes/tests
python docs/site/element_hierarchy.py --site-dir /path/to/prebuilt-main-site
```

前两条命令做元数据整理和名单准备；`regions`只从已有 CSR 缓存整理输入，三个训练臂和评估必须使用 GPU，不启用 CPU 回退。服务器侧权重、完整缓存与原图不会上传 Git。

## 解释范围

本层训练/开发/测试虽然按空间块隔离，冻结 SAE 已见过其中部分图像。更早生成的 VLM 语义资料也包含这些图像，因此语义先验不独立于封存测试；本路线首先是探索性分析。本步骤没有新开测试激活，但这不消除已有先验的重叠。最终需要独立图像与人工审查，才能检验可推广的景观语义。
