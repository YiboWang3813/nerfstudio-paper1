# Pose-Uncertainty-Aware US-GS 实验报告

日期：2026-05-27  
工作树：`omniultra` / `main`

## 结论摘要

本轮已经按 `docs/methods/ours/pose_uncertainty_prior/usgs_pose_uncertainty_plan.md` 尝试完成 Stage A-D，并补跑了 density-control 时序对 predicted uncertainty weighting 的影响。

核心结论：

1. **FiMA uncertainty 本身是有信号的**：uncertainty-error 相关性较强，Stage-B 导出的校准指标约为 Pearson 0.942、Spearman 0.907、AUROC 0.813、AUPRC 0.364。
2. **但直接把 frame uncertainty 映射成 per-frame loss weight 目前不稳定**：raw predicted weighting、oracle-error weighting、mild0p9 都失败；mild0p7 + `density_control.end_iter=7000` 可以恢复 PSNR/SSIM，但 LPIPS 变差。
3. **oracle proof-of-concept 只在局部 outlier 场景部分成立**：random outlier medium 的 hard filter 可以提高 PSNR，但 LPIPS 变差；系统性 bias/drift 下 filtering 会因为丢掉几乎所有帧而崩。
4. **footprint inflation 目前不是有效主增益**：inflate 0.05/0.10/0.20 相比 uniform corrupted prior 没有提高 PSNR，只是 LPIPS 有很小变化。
5. **MC pose marginalization 当前实现不适合作主线**：K=2/K=4 的短训都明显失败。

建议论文叙事不要把 “uncertainty weighting 一定提升重建质量” 当主结果。更稳妥的主线是：

> FiMA/PEM 输出 pose prior 和 uncertainty；uncertainty 是可靠的质量诊断信号，但在 GS 中直接按帧降权会和 coverage/density-control 强耦合。当前可保留 bounded weighting 作为受约束补充实验，主文中应强调负结果和设计边界。

## 结果文件

- Stage A oracle corruption：
  `/home/boywang25/ultrasound-scene-reconstruction/omniultra/checkpoints/ours/pose_uncertainty_stageA_oracle_250k_dcdecouple_results.tsv`
- Stage B predicted uncertainty：
  `/home/boywang25/ultrasound-scene-reconstruction/omniultra/checkpoints/ours/pose_uncertainty_stageB_predicted_250k_dcdecouple_results.tsv`
- Stage B predicted extra：
  `/home/boywang25/ultrasound-scene-reconstruction/omniultra/checkpoints/ours/pose_uncertainty_stageB_predicted_250k_dcdecouple_extra_results.tsv`
- Stage B dc7000 补跑：
  `/home/boywang25/ultrasound-scene-reconstruction/omniultra/checkpoints/ours/pose_uncertainty_stageB_predicted_dc7000_results.tsv`
- Stage C footprint inflation：
  `/home/boywang25/ultrasound-scene-reconstruction/omniultra/checkpoints/ours/pose_uncertainty_stageC_inflation_250k_dcdecouple_results.tsv`
- Stage D MC marginalization：
  `/home/boywang25/ultrasound-scene-reconstruction/omniultra/checkpoints/ours/pose_uncertainty_stageD_mc_250k_dcdecouple_results.tsv`
- FiMA uncertainty checkpoint：
  `/home/boywang25/ultrasound-scene-reconstruction/omniultra/checkpoints/fima/fima_pooled_fullres_mamba_unc_supervised_liver_6sweep_40000/checkpoint_final.pt`
- FiMA Stage-B 导出目录：
  `/home/boywang25/ultrasound-scene-reconstruction/omniultra/checkpoints/ours/pose_uncertainty_stageB_predicted/r2`

## Stage A：Oracle Corruption

基准 clean GT：

| 实验 | PSNR | LPIPS | L1 | SSIM |
|---|---:|---:|---:|---:|
| clean_gt | 26.8375 | 0.3092 | 0.03279 | 0.5329 |

random outlier：

| 实验 | PSNR | LPIPS | 结论 |
|---|---:|---:|---|
| mild_uniform | 26.8277 | 0.3087 | 几乎不受影响 |
| mild_oracle_weight | 27.0324 | 0.3401 | PSNR/SSIM 提升，LPIPS 变差 |
| mild_oracle_filter | 25.8231 | 0.4248 | coverage 不足，失败 |
| medium_uniform | 26.7178 | 0.3171 | 轻微下降 |
| medium_oracle_weight | 18.1844 | 0.6170 | soft weighting 崩 |
| medium_oracle_filter | 27.1845 | 0.3619 | PSNR 提升，LPIPS 变差 |
| severe_uniform | 26.3263 | 0.3271 | 明显下降 |
| severe_oracle_weight | 13.3296 | 0.8201 | soft weighting 崩 |
| severe_oracle_filter | 25.2701 | 0.4416 | 不如 uniform |

systematic / drift：

| 实验 | uniform PSNR/LPIPS | weight PSNR/LPIPS | filter PSNR/LPIPS | 结论 |
|---|---:|---:|---:|---|
| translation_bias | 26.8478 / 0.3069 | 26.4136 / 0.4832 | 15.0509 / 0.6041 | filter 丢帧过多 |
| rotation_bias | 26.8403 / 0.3087 | 27.4212 / 0.3957 | 14.9960 / 0.6054 | PSNR 可升，LPIPS 变差 |
| smooth_drift | 26.8308 / 0.3079 | 25.8233 / 0.4746 | 18.2404 / 0.7947 | 不适合 filter |
| sinusoidal_drift | 26.8350 / 0.3088 | 12.3229 / 0.8181 | 15.0047 / 0.6070 | weighting/filter 均失败 |
| random_walk | 26.8308 / 0.3087 | 27.3526 / 0.3689 | 15.0269 / 0.6035 | PSNR 可升，LPIPS 变差 |

Stage A 的关键判断：

- 当前 corruption 对 uniform GS 的杀伤力不够强，很多系统性误差下 uniform 仍接近 clean。
- hard filter 只适合真正局部 outlier，不适合全局 bias/drift。
- soft weighting 会改变训练覆盖和 density-control 行为，不能直接当作稳定收益。

## Stage B：FiMA Predicted Uncertainty

FiMA 导出和校准：

- `val_loss=1.23813360`
- `val_edge_trans_mm=0.03936457`
- `val_edge_rot_deg=0.00000000`
- chain/frame mean translation error 约 `3.2449 mm`
- mean rotation error 约 `0.0150 deg`
- uncertainty score：min `0.0`，mean `0.6827`，max `1.0`
- raw frame weights：min `0.2218`，mean `0.3611`，max `1.0`
- Pearson `0.9419`，Spearman `0.9069`，AUROC `0.8125`，AUPRC `0.3645`

planned Stage-B：

| 实验 | PSNR | LPIPS | L1 | SSIM | 结论 |
|---|---:|---:|---:|---:|---|
| fima_fixed | 26.8551 | 0.3160 | 0.03268 | 0.5313 | 作为 FiMA prior baseline 正常 |
| fima_pred_weight raw | 11.0550 | 0.8319 | 0.36523 | 0.3141 | raw 权重过激，失败 |
| fima_oracle_error_weight | 12.7264 | 0.8209 | 0.27895 | 0.3344 | oracle-error top20 权重也失败 |

extra / dc7000 补跑：

| 实验 | PSNR | LPIPS | L1 | SSIM | num_gaussians | 结论 |
|---|---:|---:|---:|---:|---:|---|
| pred_weight_mild0p7, dc9000 | 13.8412 | 0.8235 | 0.21195 | 0.3594 | 144979 | 后期退化 |
| fima_fixed, dc7000 | 26.8616 | 0.3163 | 0.03266 | 0.5315 | 262821 | dc7000 baseline 正常 |
| pred_weight_mild0p9, dc7000 | 12.8088 | 0.8256 | 0.26200 | 0.3400 | 205644 | 轻微权重仍失败 |
| pred_weight_mild0p7, dc7000 | 26.8897 | 0.3758 | 0.03291 | 0.5395 | 229684 | PSNR/SSIM 可恢复，LPIPS 变差 |

Stage B 的关键判断：

- FiMA uncertainty 是可校准、有排序能力的信号。
- GS 训练直接使用 per-frame loss weighting 很敏感；即使 mild0p9 的权重只在 0.922-1.0，也会失败。
- mild0p7 + dc7000 的成功说明“完全不能用”不成立，但 LPIPS 变差且结果非单调，当前不够稳，不能作为主方法结论。

## Stage C：Footprint Inflation

对 `random_outlier_medium_uniform` baseline：PSNR `26.7178`，LPIPS `0.3171`。

| 实验 | PSNR | LPIPS | L1 | SSIM | 结论 |
|---|---:|---:|---:|---:|---|
| inflate_0p05 | 26.7254 | 0.3151 | 0.03322 | 0.5275 | 几乎持平 |
| inflate_0p10 | 26.6876 | 0.3130 | 0.03342 | 0.5265 | PSNR 下降，LPIPS 微降 |
| inflate_0p20 | 26.4273 | 0.3108 | 0.03464 | 0.5217 | 变糊，PSNR 明显下降 |
| oracle_weight + inflate_0p10 | 11.9063 | 0.8269 | 0.32211 | 0.3220 | 被 weighting 拖垮 |

Stage C 的关键判断：

- 当前 diagonal footprint inflation 没有带来可靠 reconstruction 增益。
- inflation 增大后 LPIPS 略降但 PSNR/L1/SSIM 下降，符合“软化/变糊”的预期。
- 如果保留该模块，应定位为 uncertainty-aware smoothing/robustness ablation，而不是主增益。

## Stage D：MC Pose Marginalization

| 实验 | iter | PSNR | LPIPS | L1 | SSIM | 结论 |
|---|---:|---:|---:|---:|---:|---|
| MC K=2, sigma 0.10/0.10 | 2000 | 11.1537 | 0.8322 | 0.35954 | 0.3149 | 失败 |
| MC K=4, sigma 0.10/0.10 | 2000 | 11.2879 | 0.8311 | 0.35070 | 0.3167 | 失败 |

Stage D 的关键判断：

- 当前 MC jitter 训练不适合直接用于 GS 主训练。
- 该路径计算代价高，并且短训已经显示强退化，不建议继续作为主线。

## 最终建议

1. 保留 FiMA uncertainty head 和 calibration 评估，因为这个结果是正的，能够支持 “pose prior has measurable uncertainty”。
2. 不要把 raw uncertainty frame weighting 作为主方法；如果写入论文，应作为受约束设计：bounded weighting、DC 时序限制、并明确覆盖风险。
3. 不要继续加大 MC marginalization 投入，除非有新的理论约束或只作为 appendix 负结果。
4. footprint inflation 可以保留实现，但目前实验不支持它作为核心贡献。
5. 更合理的下一步不是继续调 weighting，而是重新设计 uncertainty 的使用方式，例如：
   - 用 uncertainty 作为训练 curriculum / frame sampling priority；
   - 只对 top-k 明显 outlier 做温和降权；
   - uncertainty 只进入 density-control 或 Gaussian pruning 保护策略；
   - 在论文中强调 uncertainty calibration 和 failure-mode analysis，而不是承诺重建指标稳定提升。

