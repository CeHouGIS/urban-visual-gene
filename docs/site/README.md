# 主实验网站

网站只发布 30 城 DINOv3 ViT-L/16、panorama-aware BatchTopK SAE（W4096/K32）主实验，以及当前局部要素组合探索。导航包括主实验、语义地图、图像实例、维度解释、要素标签和要素关系。旧 NMF、固定槽位及开放式全景原型页面已从发布目录移除，历史实验资料仍保留于服务器实验目录。

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
