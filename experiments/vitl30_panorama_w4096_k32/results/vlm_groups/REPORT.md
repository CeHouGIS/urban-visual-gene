# 30-city panorama-aware BatchTopK SAE: VLM interpretation

本报告使用已完成的 ViT-L/16 panorama-aware SAE（输入 1,024 维，字典 4,096 维，BatchTopK K=32，epoch-10 checkpoint）。复用了特征缓存与模型，没有重新提取 DINO 特征或重新训练。

64 个 decoder group 全部完成 Qwen3-VL 结构化初稿，并通过 Codex 图像工具核对了全部 379 张局部裁剪。原始 Qwen 标签、confidence 和回答独立保留，复核描述在 reviewed_* 字段中；两者不能混为同一个模型的输出。

在这批检索实例中，7 组显示重复出现的物体家族或外观模式，8 组有主要类别但包含异类实例，48 组保留 mixed 描述。这是对小规模证据的定性检查，不能当作全量语义纯度或准确率。

只有一个实例的组为 [51]；其描述只针对单张裁剪，不构成组内语义一致性的证据。

证据检索覆盖 29 个映射已核对的城市、3,698 个候选全景点位，选出 379 个组别实例。候选包含各城市、各组 dominant-patch share 最高的 2 个点位、均匀分布的 16 个点位，以及之前的实例；这不是全量图像检索。Moscow 的图像映射未核对，因此不用于视觉解释。

针对每个候选点位，将四方向的 3,136 个 contextualized patch 一起送入冻结 SAE，按组内所有维度的最大正稀疏激活检索实例，保留非 dominant 响应。每组优先选择不同城市和不同维度，最终每个点位只贡献一个实例。

Qwen3-VL-2B-Instruct 接收干净的 96×96 局部裁剪（在 448×448 方向图上裁剪，再放大到 192×192 输入）。原始四方向全景、192×192 上下文裁剪和激活叠图另行保存供检查。颜色叠图、位置框、城市名称与地理标签均不输入 VLM。

每个回答包含 label、semantic_type、recurring_cues、ambiguities 和 confidence。程序检查 JSON、字段取值、列表长度和生成是否截断；不合格回答自动重试并保留历次原始输出。旧的 35 份草稿单独保留。

复核发现局部模型有时会用单个实例来命名整组：例如 G00 的 water cooler 只对应其中一张裁剪，G06 的 trampoline 也只出现一次，G12 的 bench 忽略了另外五张植被裁剪。复核据此保留异质组，避免将一个物体外推成整组的含义。

较明确的实例包括 G01 的交通工具家族、G30 的道路标线和边界、G35 的天空与云层。G35 的六张裁剪只覆盖两个成员维度；即使实例看起来一致，也不能证明全组的所有维度语义相同。

这些结果只描述检索到的实例，不能证明一个组内的全部 SAE 维度都有相同语义。VLM 的 confidence 是模型自述，不能当作准确率；Codex 复核同样是自动化图像判断，尚未进行独立人工标注验证。decoder 分组、城市统计与共现分析均不使用这些描述。

激活图已使用灰度底图和 inferno 色谱（紫→橙→黄），青色框标出检索 patch；每张全景按自己的最大响应归一化，并保留原始 peak 数值。颜色适合检查激活位置，跨全景比较强度要看原始数值。

- 完整结果表：`group_interpretations.csv`
- 原始回答、提示词与图像路径：`group_interpretations.json`
- 本地交互证据页面：`/root/code/urban-visual-gene/outputs/experiments/nas_30city_vitl_panorama_w4096_k32/analysis/semantics_v2/index.html`
- 语义证据插图：`Fig_ViTL30_VLM_Interpretation.png` / `.pdf`
- 激活对照插图：`Fig_ViTL30_Activation_Examples.png` / `.pdf`

## 组别解释

| Group | 维度数 | Qwen 原始标签 | 图像复核描述 | 实例模式 | 实例/城市 |
|---|---:|---|---|---|---:|
| G00 | 22 | water cooler | mixed | mixed | 6/6 |
| G01 | 87 | vehicle | vehicles and transport objects | recurring | 6/6 |
| G02 | 62 | mixed | mixed | mixed | 6/6 |
| G03 | 14 | mixed | ground and vegetation boundaries | recurring | 6/6 |
| G04 | 51 | road | mixed | mixed | 6/6 |
| G05 | 41 | mixed | mixed | mixed | 6/6 |
| G06 | 831 | trampoline | mixed | mixed | 6/6 |
| G07 | 31 | mixed | mixed | mixed | 6/6 |
| G08 | 178 | road | mixed | mixed | 6/6 |
| G09 | 219 | texture | mixed | mixed | 6/6 |
| G10 | 54 | mixed | mixed | mixed | 6/6 |
| G11 | 25 | textured fabric | mixed | mixed | 6/6 |
| G12 | 251 | bench | vegetation with outlier | predominant_with_outliers | 6/6 |
| G13 | 32 | mixed | mixed | mixed | 6/6 |
| G14 | 69 | rail | mixed | mixed | 6/6 |
| G15 | 65 | light post | mixed | mixed | 6/6 |
| G16 | 80 | mixed | low-detail surfaces and sky | recurring | 6/6 |
| G17 | 23 | road | mixed | mixed | 6/6 |
| G18 | 88 | car | mixed | mixed | 6/6 |
| G19 | 18 | mixed | mixed | mixed | 6/6 |
| G20 | 69 | table | mixed | mixed | 6/6 |
| G21 | 75 | road | mixed | mixed | 6/6 |
| G22 | 48 | car | mixed | mixed | 6/6 |
| G23 | 14 | mixed | mixed | mixed | 6/6 |
| G24 | 20 | mixed | mixed | mixed | 6/6 |
| G25 | 203 | mixed | mixed | mixed | 6/6 |
| G26 | 79 | grass | vegetation and ground cover | recurring | 6/6 |
| G27 | 13 | Earth | mixed | mixed | 6/6 |
| G28 | 28 | car | mixed | mixed | 6/6 |
| G29 | 34 | car | mixed | mixed | 6/6 |
| G30 | 20 | road marking | road markings and boundaries | recurring | 6/6 |
| G31 | 28 | road | mixed | mixed | 6/6 |
| G32 | 20 | mixed | mixed | mixed | 6/6 |
| G33 | 22 | window | mixed | mixed | 6/6 |
| G34 | 232 | garage door | building facades and openings | predominant_with_outliers | 6/6 |
| G35 | 29 | mixed | sky and clouds | recurring | 6/6 |
| G36 | 17 | mixed | mixed | mixed | 6/6 |
| G37 | 15 | mixed | mixed | mixed | 6/6 |
| G38 | 28 | textured fabric | mixed | mixed | 6/6 |
| G39 | 10 | mixed | mixed | mixed | 6/6 |
| G40 | 12 | roll-up shutter | roller shutters with outliers | predominant_with_outliers | 6/6 |
| G41 | 9 | textured surface | mixed | mixed | 6/6 |
| G42 | 9 | mixed | people and human portraits | predominant_with_outliers | 6/6 |
| G43 | 63 | building | pale low-detail surfaces | recurring | 6/6 |
| G44 | 9 | mixed | mixed | mixed | 6/6 |
| G45 | 23 | mixed | mixed | mixed | 6/6 |
| G46 | 45 | car | mixed | mixed | 6/6 |
| G47 | 34 | mixed | wall surfaces with outliers | predominant_with_outliers | 6/6 |
| G48 | 58 | tree | trees and branches with outlier | predominant_with_outliers | 6/6 |
| G49 | 59 | fence | mixed | mixed | 6/6 |
| G50 | 65 | mixed | mixed | mixed | 6/6 |
| G51 | 8 | textured surface | single ground-surface example | single_example | 1/1 |
| G52 | 63 | mixed | vegetation with outliers | predominant_with_outliers | 6/6 |
| G53 | 26 | mixed | mixed | mixed | 6/6 |
| G54 | 16 | mixed | mixed | mixed | 6/6 |
| G55 | 57 | trash can | vegetation and leaf litter with outliers | predominant_with_outliers | 6/6 |
| G56 | 28 | mixed | mixed | mixed | 6/6 |
| G57 | 33 | car | mixed | mixed | 6/6 |
| G58 | 122 | cobblestone | mixed | mixed | 6/6 |
| G59 | 31 | mixed | mixed | mixed | 6/6 |
| G60 | 22 | mixed | mixed | mixed | 6/6 |
| G61 | 23 | mixed | mixed | mixed | 6/6 |
| G62 | 18 | car | mixed | mixed | 6/6 |
| G63 | 18 | plant | mixed | mixed | 6/6 |
