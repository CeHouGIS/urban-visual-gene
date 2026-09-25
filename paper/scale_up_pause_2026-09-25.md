# Feature-MAE Scale-up 暂停记录

- 状态：已按用户要求暂停
- 已完成：6/21 个任务
- 当前任务：`scale_n10000_w0128_s42`
- 恢复方式：保留本地 `checkpoint_last.pt`，runner 会跳过已有报告并从 checkpoint 继续
- 本次提交不包含 checkpoint、模型、图像、parquet 或大型生成特征

## Graph 实验

Graph 原型实验已完成：378,818 张图像、2,080 维 descriptor、K=8，结果摘要和复现代码位于 `paper/data/image_graph_archetypes/` 与 `scripts/image_graph_archetypes/`。
