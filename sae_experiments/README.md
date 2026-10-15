# 共用实验组件

版本专用流程在 [experiments/](../experiments/README.md)。这里保留 models/（BatchTopK和context mixer）、backbones/、feature_extraction/、training/、semantic_tools/ 及共享IO和绘图。

sae_experiments.panorama_vitl30_* 通过包搜索路径指向主实验code/；evaluation.encode_batchtopk_w1024_k8 指向全球基线，未保留重复源文件。
