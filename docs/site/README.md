# 主实验网站

网站只发布 `vitl30_panorama_w4096_k32`：30 城市、DINOv3 ViT-L/16、28×28 patch、panorama context mixing、W4096/K32。构建时核对 checkpoint SHA256，避免将不同版本的同编号维度混在一起。

首页汇总训练和城市结果；导航保留维度语义、Top10 实例、masked Top10 Qwen 解释与同一模型的组合探索。旧 Feature-MAE、ViT-B、全球版本、尺度和参数消融页面不再发布。服务器上的研究文件由各实验目录继续管理。

```bash
python docs/site/build_main_site.py --output /tmp/usp-current-site --workers 6
```

需要服务器现有主实验的 outputs、reports、原图映射、NumPy 与 Pillow。脚本只读取这些资料，写入一个全新的目录，不修改实验数据或后台任务。VLM 发布内容是构建开始时的固定快照，后台新完成的结果须重新构建才能上线。

Top100 完整证据和悬停预览按保存的裁剪框从共享原图在浏览器绘制，避免复制约 27 万张重复裁剪；排名、UMAP 坐标、分组及激活图不重算。逐维 JSON 通过 gzip 无损压缩，浏览器用 DecompressionStream 解压，需要支持该标准的现代浏览器。原始方向 JPEG 原样复制；Qwen 实际输入拼图使用像素一致的无损 WebP。每个维度可查看最多 Top100、原图、激活和四方向全景，渐进加载。

四方向显示按已核对的同一 `city/pano`、0° / 90° / 180° / 270° 顺序，保留原图长宽比。证据卡片使用不收缩的独立图片格，支持横向滚动和放大，避免窄卡片内图片重叠。`panorama_display.py` 从已有 NMF 实例名单、组合报告的原有选择规则及峰值缓存重绘组合展示图；峰值标记按每张图实际显示的宽高映射。`combinations/panorama_layout.json` 保存逐图的全景索引供核查。四张透视图并排展示不等于无缝球面投影拼接。

发布前检查静态资源引用，并用浏览器验证 UMAP 筛选、悬停、逐维证据、热图、四方向图和手机布局。生成目录替换 `gh-pages` 的全部站点内容，保留 Pages workflow；不覆盖 `main` 中其他未提交工作。公开首页为 https://cehougis.github.io/urban-visual-gene/ 。
