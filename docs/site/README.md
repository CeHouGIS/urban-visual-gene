# 主实验网站

网站只发布 30 城 DINOv3 ViT-L/16、panorama-aware BatchTopK SAE（W4096/K32）主实验，以及当前局部要素组合探索。导航包括主实验、语义地图、图像实例、维度解释和要素组合。旧 NMF、固定槽位及开放式全景原型页面已从发布目录移除，历史实验资料仍保留于服务器实验目录。

## 当前探索

复用现有 SAE 激活，在每个方向 28×28 patch 网格中形成自适应区域，再以完整四方向全景为统计单位检索两个不同暂定要素父类的重复关系。四张方向图按 0°、90°、180°、270° 顺序展示，不进行球面无缝拼接。父类来自事后 VLM 解释，窗户和墙面同属建筑。

29 城样本包含 1,392 个训练全景、464 个开发全景，合计 7,424 张方向图。训练提出 298 个原始关系候选，其中 173 个在开发集两种置换检验中均达到 BH-FDR q<0.05。网页仅在展示层合并方向等价关系，呈现 5 个布局与 30 张四方向遮罩实例；显著性仍按原始 298 个候选计算。源码位于 experiments/vitl30_local_compositions/run.py，发布数据位于 elements/local-compositions/results.json。

这是探索性关系检索，尚未训练连续软匹配原型模型、搜索三元关系或验证逐图 80% 语义覆盖。要素父类、mask 和对象完整性需要人工核查。现有 VLM 语义先验可能包含后续测试图像，因此置换显著性并非独立语义确认；封存测试激活未在本试验中使用。

## 更新网站

    python docs/site/build_main_site.py --output /tmp/usp-current-site --workers 6
    # 或在已构建的专用 gh-pages 工作树刷新当前路线
    python docs/site/element_hierarchy.py --site-dir /path/to/pages-checkout
    python -m pytest docs/site/tests -q

构建核对主 SAE checkpoint、VLM 解释及空间划分数据身份。右侧研究看板由 docs/site/roadmap.json 与 docs/site/project_progress.py 生成，区分计算完成、人工语义验证及不同的覆盖目标。后台 VLM 产出更新后需重新构建网站。
