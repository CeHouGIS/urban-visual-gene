# 主实验网站

网站只发布 30 城 DINOv3 ViT-L/16、panorama-aware BatchTopK SAE（W4096/K32）主实验，以及当前局部要素组合探索。导航包括主实验、语义地图、图像实例、维度解释、要素标签、标签结构、维度复核和要素关系。旧 NMF、固定槽位及开放式全景原型页面已从发布目录移除，历史实验资料仍保留于服务器实验目录。

标签结构页使用现有 Qwen 细标签和显式整理规则区分 `is_a`（如松树属于树木）和 `part_of`（如树冠是树木的一部分）；只有共享同一上级且关系类型相同，或明确列入同层组的标签才展示为并列。392 个可展示细标签中，200 个进入至少一种明确关系，192 个保留为未判定；这些均为待人工复核的文字关系，不是图像语义验证。每个标签可跳到要素页查看对应 SAE 维度与全景激活。规则及来源见 `docs/site/label_structure_rules.json`、`docs/site/label_structure.py`。

维度复核页列出 63 个 Top100 峰值位置集中的候选，其中 5 个符合较高支持度的优先复核条件；另列出 33 个疑似模糊响应候选（29 个旧 Qwen 名称含模糊词，4 个少量目视抽查发现低清晰度实例）。位置集中不等于位置编码，旧名称或低清晰度实例也不能证明该维度检测模糊。审阅者可核对遮罩拼图、原图和热图，并在浏览器本地标记待剔除项、导出 CSV；构建过程不删除或屏蔽任何维度。

要素标签页包含 Qwen 类别图谱：按暂定父类和 Qwen 细标签汇总全部 4,096 个 SAE 维度。每个细标签最多展示 5 个四方向全景，分别显示原图和 SAE 激活高亮；优先选不同维度、不同点位的高峰值实例。实例先从固定验证样本取，稀缺标签补充 86,038 个全景的扩大样本记录。若仍不足 5 个正激活实例，就显示真实数量。图谱数据由 `experiments/vitl30_relational_prototypes/results/hierarchy.json` 生成到 `elements/category-graph.json`；维度数量不是地物出现频率，类别仍需人工验证。要素关系结果单独发布在 `relations/`。

## 当前探索

复用现有 SAE 激活，在每个方向 28×28 patch 网格中形成自适应区域，再以完整四方向全景为统计单位检索两个不同暂定要素父类的重复关系。四张方向图按 0°、90°、180°、270° 顺序展示，不进行球面无缝拼接。父类来自事后 VLM 解释，窗户和墙面同属建筑。

29 城样本包含 1,392 个训练全景、464 个开发全景，合计 7,424 张方向图。新的全景级检索观察到 58,595 个关系 key，提出 1,583 个三节点训练候选，网页展示其中 16 个关系布局（每个布局 6 个四方向实例）。1,543 个候选的两个探索性置换检验均达到 BH-FDR q<0.05；这只是排序证据，不能直接视作已确认原型。源码位于 experiments/vitl30_local_compositions/panorama_motifs.py，发布数据位于 relations/local-compositions/results.json。旧的单方向二元关系快照不再发布。

这是探索性关系检索，尚未训练连续软匹配原型模型、搜索三元关系或验证逐图 80% 语义覆盖。要素父类、mask 和对象完整性需要人工核查。现有 VLM 语义先验可能包含后续测试图像，因此置换显著性并非独立语义确认；封存测试激活未在本试验中使用。

## 更新网站

    python docs/site/build_main_site.py --output /tmp/usp-current-site --workers 6
    # 或在已构建的专用 gh-pages 工作树刷新当前路线
    python docs/site/element_hierarchy.py --site-dir /path/to/pages-checkout
    python -m pytest docs/site/tests -q

构建核对主 SAE checkpoint、VLM 解释及空间划分数据身份。右侧研究看板由 docs/site/roadmap.json 与 docs/site/project_progress.py 生成，区分计算完成、人工语义验证及不同的覆盖目标。后台 VLM 产出更新后需重新构建网站。
