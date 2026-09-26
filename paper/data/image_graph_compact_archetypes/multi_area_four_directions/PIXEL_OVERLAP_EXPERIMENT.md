# Pixel-overlap DINOv3 + Feature-MAE 对照实验

上一组 10 个地点的四方向图片被构造成 8 个像素级窗口。四个新增窗口分别跨越 0°/90°、90°/180°、180°/270°、270°/0° 接缝；每个窗口完整经过 DINOv3 和冻结 Feature-MAE，然后在 512D 层融合。

- 独立方向 512D 接缝余弦相似度：`0.6775`
- 像素重叠 512D 接缝余弦相似度：`0.9258`
- 独立方向 winner 接缝一致率：`11.96%`
- 像素重叠 winner 接缝一致率：`50.36%`
- 独立方向 F 接缝一致率：`13.04%`
- token-overlap F 接缝一致率：`35.36%`
- pixel-overlap F 接缝一致率：`54.82%`
- 峰值 GPU 显存：`0.400 GiB`

注意：本原型用相邻方向图各一半组成跨缝窗口，并非从原始 equirectangular panorama 重新渲染真实 45° 透视图。pixel-overlap 的接缝一致率略高于普通内部邻接，说明方法有效，但也可能存在跨缝过度一致化；正式替换前应再用真实 45° 视图验证。

对照图：`paper/figures/supplementary/multi_area_four_directions/Fig_Pixel_Overlap_Panorama_Comparison.png`

逐地点指标：`pixel_overlap_method_comparison.csv`
