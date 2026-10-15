# 早期 37 城市采样计划

configs/formal_city_manifest.json 记录37城、每城目标12800个完整四方向全景。code/watchdog_batchtopk.sh 是当时的守护脚本，调用共享 formal.gpu_run。

目标配额不等于完成量。本目录不含可独立核验的完整结果；后续扩展为307城候选，又定义了当前30城ViT-L主实验。旧配置和脚本路径保留链接，运行时请先检查默认设置。

code/slurm/ 是明确引用37城配置的历史训练 / 编码集群作业；runtime/logs/ 保留该方案的旧运行日志。
