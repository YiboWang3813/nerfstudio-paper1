# 当前研究导航

原生 Nerfstudio 和 gsplat 接入已完成工程测试；八个方法在 synthetic_liver/l1 上的 5k 历史指标参照探针已完成；UltraGS 当前参数与旧参照有 SSIM 差异，按历史参数复跑后接近。历史图表可供追溯，不能替代新代码的结果。

## 保留的方法

| 方法 | 历史说明 |
| --- | --- |
| Ours | [方法与数学](../archive/eehpc-20261017/01_problem_and_methods/ours/) |
| UltraGauss | [说明](../archive/eehpc-20261017/01_problem_and_methods/baselines/ultragauss/) |
| UltraGray | [说明](../archive/eehpc-20261017/01_problem_and_methods/baselines/ultragray/) |
| GaussianPile | [说明](../archive/eehpc-20261017/01_problem_and_methods/baselines/gaussianpile/) |
| Slice Adapted 3DGS | [说明](../archive/eehpc-20261017/01_problem_and_methods/baselines/slice_adapted_3dgs/) |
| UltraGS baseline | [说明](../archive/eehpc-20261017/01_problem_and_methods/baselines/ultrags_baseline/) |
| NeRFUS | [说明](../archive/eehpc-20261017/01_problem_and_methods/baselines/nerfus/) |
| UltraNeRF | [说明](../archive/eehpc-20261017/01_problem_and_methods/baselines/ultranerf/) |

淘汰方法的旧说明仅随历史文档保留，不进入当前方法列表，也不要求迁移其代码或环境。

## 数据和协议

使用冻结的 `p100k-native-v2`：synthetic_liver、spine_phantom、tus_rec_2024_qc、tdsc_abus_2023，共 78 个 sweep。原始 tus_rec_2024 与 natural_llff 不在当前迁移范围。

[原生协议与历史来源](../../nerfstudio/experiments/paper1/README.md)集中在代码仓库；配置快照在 `nerfstudio/configs/ultrasound_protocols`。旧规约中的 Slurm、环境路径和云存储用法不覆盖本机 AGENTS.md。

## 证据和文稿

- [八方法代表 sweep 指标对照](../../results/probes/native-5k-liver-l1/README.md)：六项指标、逐帧结果及重建图，停在正式 1M/30k 之前。
- [当前工程验收记录](../../migration/preformal-acceptance.json)：34 项测试及 32 组真实数据短程案例，不等于新旧效果等价。
- [结果仓库](../../results/README.md)：历史数值、图表输入与来源清单。
- [历史文稿](../archive/eehpc-20261017/05_manuscript/)：保留原样，待原生代码正式复现后再更新论文结论与图表。

探针只用新代码运行各方法一个代表 sweep，以历史 results 的指标和重建质量作参照；不再要求逐算子、逐梯度等价。参数版本差异在回执中注明。通过后再选择值得保留的历史关键 checkpoints，不整包迁移。
