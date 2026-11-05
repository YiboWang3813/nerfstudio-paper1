# Paper1 原生实验协议

本仓库是 paper1 当前运行配置、方法接入和协议工具的入口。正式 1M/30k 实验尚未启动；已有原生工程测试不能替代逐方法历史指标参照验收。

| 内容 | 当前入口 |
| --- | --- |
| 八方法原生配置生成 | `nerfstudio/configs/ultrasound_protocol.py` |
| 冻结算法配置快照 | `nerfstudio/configs/ultrasound_protocols/` |
| 配置转换来源 | `scripts/ultrasound/convert_protocols.py` |
| 数据准备与包校验 | `scripts/ultrasound/{prepare_manifest,freeze_datasets,package_dataset,publish_dataset}.py` |
| 原生工程短程验证 | `scripts/ultrasound/smoke_matrix.py` |
| 代表 sweep 的实际指标对照 | [八方法结果](../../../docs/experiments/native-5k-liver-l1/report.md) |
| 历史指标参照探针 | `scripts/ultrasound/probes/results_probe.py` |
| 工程验收证据 | `../../../migration/preformal-acceptance.json` |
| 当前研究说明 | `../../../docs/current/README.md` |
| 历史数值与图表 | `../../../docs/experiments/README.md` |

## 当前边界

保留 ours、ultragauss、ultragray、gaussianpile、slice_adapted_3dgs、ultrags_baseline、nerfus、ultranerf。冻结数据版本为 `p100k-native-v2`，覆盖四个数据集的 78 个 sweep。数据正文只放 `/home/yibo/shared/datasets`；运行时共用 `/home/yibo/shared/envs/torch251-cu124-py311`，本项目 `.venv` 仅路由本项目源码。

逐方法探针只运行新代码，使用历史 results 指标作为质量参照；不要求旧算子、梯度和参数更新严格等价。当前使用 synthetic_liver/l1 的 5k 前缀，高斯方法保持 1M 初始化，比较同一完整测试帧集合上的六项指标，并保存代表帧重建图。部分历史参考为较早参数版本，差异在探针回执中注明；检查有无明显退化，不将这种参照冒称完全受控等价。未启动正式 1M/30k 矩阵。

历史 checkpoints 仍待有选择地处理；旧环境和旧二进制不进入当前运行入口。

## 历史来源

源端独立 protocol 仓库的冻结快照收入 [provenance/eehpc-20261017](provenance/eehpc-20261017/README.md)，来源提交 `042cc1baa39fedde52130029e61ff6c00f3d1148`，通过 Git subtree 保留来源。这里的旧 workspace lock、Slurm 工具、checkpoint 清单与绝对路径是历史实验的来源记录，不是本机环境声明，也不表示清单中的权重已迁入。

原生配置快照承接有效算法参数；当前实际运行工具在 `scripts/ultrasound`。源端 protocol 的工具和规则保留在历史快照中供审计，不能直接运行旧 Slurm 命令。本机操作以 `/home/yibo/AGENTS.md`、`/home/yibo/paper1/AGENTS.md` 为准。不建立独立的 `/home/yibo/paper1/protocol` 一级项目。

## 代表 sweep 对照结果

八个方法已用新代码完成 synthetic_liver/l1 的 5k 指标参照。七个当前参数结果接近或优于历史参照；UltraGS 当前参数 PSNR 提升但 SSIM 下降较多。仅用新代码按历史参数复跑后，PSNR 差约 −0.005 dB、SSIM 差约 −0.001，支持参数版本变化是首轮差异的主要来源。当前默认参数没有因此被改写，完整两组结果均保留。
