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

## 右侧项目进度

五个主页面共用右侧项目看板，跟踪整条研究线：数据与全景表征 → SAE 候选单元 → 单元语义核查 → 局部景观原型 → 原型有效性验证 → 城市与空间规律 → 论文与成果发布。每阶段可展开查看研究问题、现有证据、缺口与完成判据；顶部显示当前关键路径和下一步。计算任务明细默认收起，手机上整个看板可展开。

`roadmap.json` 管理研究目标、完成判据、写作进度和未核查事项，须随实际研究结论更新。语义核查、城市规律和论文阶段的 `review_status` 由核查结果维护，支持 `partial`、`pending`、`running`、`complete`；变更时同时更新证据、剩余工作和 `reviewed_on`。`project_progress.py` 仅读取主实验与原型层的少量状态 JSON，生成 `data/project-progress.json`；不扫描原图、不导入训练模型。计算完成不会自动把语义验证或论文阶段标成完成，也不把训练步数当作整个项目的百分比。

```bash
# 给已有站点 bundle 安装看板；完整构建也自动调用
python docs/site/project_progress.py --site-dir /tmp/usp-current-site --install

# 仅刷新本地进度 JSON
python docs/site/project_progress.py --site-dir /tmp/usp-current-site

# 已发布且 HEAD 与 origin/gh-pages 一致的专用 Pages checkout：
# 每 10 分钟检查后台进度，只在内容变化时单独提交并推送进度 JSON。
# 流水线完成、失败或停止时发布最后状态后退出。不会合并或覆盖其他发布。
python -u docs/site/project_progress.py --site-dir /path/to/pages-checkout --publish --watch --interval 600
```

浏览器每分钟检查已发布快照，可手动刷新；标明北京时间与快照更新时间。GitHub Pages 的部署延迟仍然存在。后台 watcher 应在持久终端中运行；静态网页不会直接连接服务器或读取 GPU。逐维 Qwen 的图像和解释正文仍需重新构建，不能只更新进度 JSON。
