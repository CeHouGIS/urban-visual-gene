# 共用质量筛查与历史入口

实验代码已分版本放入 [experiments/](../experiments/README.md)。

- quality/：全黑 / 隧道筛查、特征QC映射与城市筛查；qc_model.joblib 是小型弱标签分类器，需抽检。
- slurm/qc_*.sbatch：共用筛查作业。
- formal.gpu_run → experiments/global307_baseline_w1024_k8/code/gpu_run.py。
- formal.analysis.* → 各版本 code/ 或 sae_experiments/semantic_tools/，通过包路径兼容。
- 守护脚本、非QC Slurm 和logs保留相对链接，实体在对应版本目录。

[全球基线说明](../experiments/global307_baseline_w1024_k8/README.md)；[当前主实验](../experiments/vitl30_panorama_w4096_k32/README.md)。
