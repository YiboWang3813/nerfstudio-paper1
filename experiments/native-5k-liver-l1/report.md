# 八方法原生代码历史指标对照

仅运行新代码，以已保留的历史 results 指标为参照。synthetic_liver/l1，冻结数据 p100k-native-v2，同一完整测试帧集合，5000 次更新；六个高斯方法保持 1M 初始化。正式 1M/30k 矩阵未启动。

| 方法或参数版本 | 历史 PSNR | 新 PSNR | Δ PSNR | 历史 SSIM | 新 SSIM | Δ SSIM |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| ours | 28.985 | 29.195 | +0.210 | 0.6551 | 0.6500 | -0.0051 |
| ultragauss | 28.639 | 28.365 | -0.275 | 0.6297 | 0.6262 | -0.0035 |
| ultragray | 28.632 | 28.759 | +0.127 | 0.6311 | 0.6288 | -0.0023 |
| gaussianpile | 26.380 | 26.196 | -0.183 | 0.5334 | 0.5385 | +0.0051 |
| slice_adapted_3dgs | 28.513 | 28.021 | -0.492 | 0.6392 | 0.6314 | -0.0078 |
| ultrags_baseline | 25.272 | 25.986 | +0.714 | 0.6183 | 0.5134 | -0.1049 |
| nerfus | 21.508 | 22.518 | +1.010 | 0.6109 | 0.6255 | +0.0146 |
| ultranerf | 21.660 | 21.691 | +0.031 | 0.5579 | 0.5562 | -0.0016 |
| ultrags_baseline_historical_parameters | 25.272 | 25.266 | -0.005 | 0.6183 | 0.6173 | -0.0010 |

完整六项指标见 [metrics.tsv](metrics.tsv)。执行回执、逐帧指标、冻结原生配置和对照图片位于 outputs/probes 对应方法目录（图片左为目标，右为重建）。

七个方法使用当前冻结配置的结果与历史参照接近或更好。UltraGS baseline 当前配置存在明显 SSIM 差异；历史 stage1 的初始化、尺度与 opacity 学习率、密度控制、opacity reset 以及物理切片裁剪行为与当前参数不同。额外一行只用新原生实现重放历史参数，原始当前配置结果保留，不用复跑结果覆盖。

历史参数逐项差异见 [reference-config-differences.json](reference-config-differences.json)。历史参数 UltraGS 的入口是 `results_probe.py --method ultrags_baseline --historical-ultrags`；没有执行旧 Python 训练器或旧算子，未改动生产默认配置。

UltraGS 按历史参数复跑后 PSNR 差约 −0.005 dB、SSIM 差约 −0.001，其余四项指标也接近。这支持首轮差异主要来自参数版本变化。当前默认参数仍保留首轮 SSIM 差异，未据此改动生产参数。

该对照用于用户要求的近似质量验收，不声称逐算子等价、种子统计等价或完整预算复现。历史高斯参照来自较早 stage1 参数版本；神经场参照来自 implicit_stage3_full_sweep 的 5k 中间记录。
