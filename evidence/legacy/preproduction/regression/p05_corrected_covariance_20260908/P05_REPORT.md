# P0.5 corrected covariance freeze / impact assay

本阶段只授权 covariance correction、tests、smoke 和两个 MP/full/full、1M/30k、seed 42 anchor。其他矩阵没有提交。最终 LaTeX 工程尚未提供，不妨碍本阶段代码与测试；交换版 tex 未修改。

## 1. Root cause 与修改范围

row-major R 的 covariance helper 错将 `R diag(s²)` 乘以 R 的列，得到 `R diag(s²) R`。现统一为 `Sigma = R diag(s²) Rᵀ`。

- `ops/ours_renderer/include/shader/plane_cond2d_device.cuh`：修正乘法索引。
- `ops/ours_renderer/src/shader/plane_cond2d.cu`：重复 covariance 实现改为调用共享 helper。
- 历史 plane 分支 backward：修正 tied off-diagonal 的半梯度分配、2×2 inverse derivative 乘法及对称 Schur adjoint。此处原来的解析梯度也不一致；现在有独立 float64 和有限差分验证。它不是当前五行 LGR 使用的 us_focus renderer family。
- `src/covariance_diagnostic.cu`、`src/bindings.cpp`、`setup.py`：增加直接调用生产 helper 的诊断接口。
- `test_p0_local_modes.py`：移除 covariance xfail，所有旋转样本必须匹配 reference。
- `test_p05_covariance.py`：新增 covariance 与随机旋转 forward/backward 测试。
- `validate_plane_cond2d.py`：统一参考旋转约定、修正旧测试 scalar/FD objective，增加 float64 梯度对照。

CPU us_focus reference 已使用正确公式，无需修改。Global ray 自行使用正确的 inverse covariance/whitening；mapper 的 projected extents 本来也正确。Local cached footprint、local contribution/residual diagnostics 都使用共享 helper，因此随修复统一更新。Initialization 仅构造/采样 positions、scales、quaternions，不调用错误 covariance。A_g 的定义未变，但 geometry、tau、local residual 与 adaptation 轨迹改变后，A_g 和最终 population 也会改变。没有改动任何 baseline/common renderer。

## 2. Default 与 ablation interfaces

默认仍为 adaptive local + full integral global + independent learnable tau + residual-guided/path-aware CGA。
保留 none/fixed/adaptive、gradient/residual_guided、opacity/none/local_aware/path_aware。
Global disabled 或 beta/lambda 为零时，path-aware 退化为 local-aware。
LGR structural 为 Plane、Fixed 1 mm、Adaptive、Simple Global、Full 五行；已删除 acoustic-control 计划，不再声称 attribute decoupling 获得独立消融验证。

## 3. Tests 与 smoke

GPU compute node gpu3，RTX 4090，CUDA 11.8。job 79910 的 CUDA 编译通过；最终测试/smoke job 79914。

- Pytest：55 passed，零 xfail。
- Covariance：128 个随机 quaternion × positive/zero-scale 两类，加 identity cases；检验 symmetry、PSD/PD、特征值与 anisotropic scale spectrum。
- 原始 covariance 最大 CUDA/reference 绝对误差：5.72205e-6，满足设置的绝对/相对容差。
- 随机旋转 renderer forward 最大绝对误差：5.96046e-8。
- 随机旋转 backward 与 autograd reference 最大绝对误差：8.74703e-5，满足测试容差；生产 shape backward 保留原有有限差分实现。
- 历史 plane 分支：forward 与 reference 一致，float64 scale-gradient 对照通过；有限差分相对误差约 0.336%。
- P0 suite 的三模式、梯度、A_g、adaptation event、独立 pruning、no-LGR fallback、PPP 全部通过。
- 六个真实数据 smoke（Liver/Arm × adaptive/fixed/none）各 100 iterations、1M init、两次 adaptation event，均成功、finite output、无 OOM。具体配置和 stats 保存在 `freeze/smoke/`。

前两次测试尝试保留在日志中：第一次发现历史 plane backward 问题；第二次发现该测试很小的 float32 FD 步长产生 cancellation。最终测试增加独立 float64 参考并采用 1e-3 FD 步长，而不是放宽 renderer/reference 判定。

## 4. Source freeze

Base commit：`2834e7ec24a1346b2819fbb27b7efb3981f5c98c`，不 commit 用户原有 dirty changes。

最终 freeze_v2/source.tar.gz SHA-256：

```text
b2525342593c8dd84a1f1ddb8c1ac124c9b1fb179ddcffea3fb902eb2a073ad2
```

`freeze_v2/` 包含 source.tar.gz、全源码 hashes、git status、full current diff、covariance.patch、相对 P0 的完整 patch、P0 infrastructure patch、pre-P0/P0 source archives、exact Hydra anchors、binary 和 hashes、pip/conda 环境、Python/host compiler、完整 CUDA build/test logs 和 smoke configs/stats。
Anchor 启动前检查 critical source/binary hash，并把 source/config/binary SHA 写入自己的 run_manifest。
这恢复了源代码和参数的可重建性；跨 toolchain/hardware 的编译产物逐位一致及 GPU atomic reduction 的逐位确定性不作保证。已额外保留实际使用的 binary。

Liver 使用原 `freeze/`，SHA 为 `56efcb665e0e8e01b3c7c40e73c0b75adff6b7932c8cff258fac16b15f80ebb4`。它完成了训练/评估/checkpoint 保存，但额外的 post-validation 将 GaussParam 当作 Tensor 而退出。直接加载 30k checkpoint 核验所有参数和最终指标有限后，恢复了成功状态并保留原错误 stats；没有重新训练。v2 仅修正该审计包装脚本及恢复/封存脚本，逐文件验证所有 production、config、test 源码和 CUDA binary 与 v1 完全相同。Arm 使用 v2。两次训练的真实源码版本均被保留，不能把 wrapper 文件变化误判为算法变化。

重建：恢复 source archive；为 P0 回归将 pre_p0_source.tar.gz 解压到测试指定的 `outputs/preproduction/regression/p0_semantic_freeze_20260908/pre/tree`；按 conda/pip 环境及 CUDA build logs 重建。`p05_test_gpu.sh` 默认强制编译。训练不得盲目重放 shell：只有两个 anchor 获准，且已有结果目录不可覆盖。

## 5. Anchor impact

采用 MP 而非 PPP，以便直接匹配 Stage-2 full-renderer/file-pose。相关 model/scene/loss/engine/seed 配置经兼容映射后逐项一致，差异文件为空。不是可严格重建 legacy binary 的 A/B 实验，不做显著性结论。

两个 anchor 均已完成，未调参。完整指标和逐项 delta 见 `anchor_impact.json`。

| 指标 | Liver legacy | Liver corrected | Delta | Arm legacy | Arm corrected | Delta |
|---|---:|---:|---:|---:|---:|---:|
| PSNR | 29.757154 | 29.656905 | -0.100248 | 32.762396 | 32.595118 | -0.167279 |
| SSIM | .677750 | .672849 | -.004901 | .865987 | .862374 | -.003612 |
| LPIPS | .233258 | .179989 | -.053269 | .141053 | .120477 | -.020576 |
| L1 | .022820 | .023018 | +.000198 | .012858 | .013010 | +.000152 |
| DISTS | .185306 | .151676 | -.033630 | .158611 | .145150 | -.013461 |
| GMSD | .096205 | .096490 | +.000284 | .056025 | .056300 | +.000275 |
| Final #G | 783211 | 616314 | -166897 | 844428 | 660659 | -183769 |

| Corrected run | Whole-run wall | Peak local tile–Gaussian pairs | Peak global ray–Gaussian pairs | PyTorch peak allocated GiB | Pure-forward FPS |
|---|---:|---:|---:|---:|---:|
| Liver l1 | 425.979 s | 3220422 | 261852 | 2.882 | 249.49 |
| Arm 010_LH_Per_L_DtP_qc_0005_0370 | 326.891 s | 3389200 | 171742 | 5.757 | 357.92 |

Wall 包含 setup/train/evaluation/save，不是纯 optimizer-step 计时；两个 legacy matched rows 都是 614 s shared-worker wall。Corrected 为单 worker/GPU，legacy 为共享 GPU worker，不能直接声称修复带来加速。FPS 按既有 5 warmup、20 holdout frames、3 timed passes、CUDA synchronization 测量，排除数据加载、metric 和 checkpoint I/O；这里只是两个 anchor，不是四数据集 main aggregate FPS。Memory 是 PyTorch allocator 的 peak，不是包含所有 CUDA context/native allocation 的 nvidia-smi 峰值。候选数是 CSR pair count，不是 pixel–Gaussian interaction count。旧 matched runs 没有可用的同口径 candidate statistics，不能给候选数的 legacy delta。

两次训练各有 12 个 adaptation events，未见 NaN/OOM、爆炸性 #G 增长，最终 checkpoint 的全部 raw parameters 已验证有限。PSNR/SSIM 小幅下降，LPIPS/DISTS 降低，属于质量取舍，不是所有指标均改善。没有发现需要暂停修复诊断的明显训练/质量异常，但单 seed、两个代表样本不足以保证其他 76 个 main sweeps 或 Plane 30k 的稳定性。

定性固定选择已保存 holdout cases 的中间索引，而不是按改善幅度挑选：Liver item_23、Arm item_41；matched GT PNG hash 完全相同。已直接检查 legacy/corrected 的 compose 图：宏观解剖结构和扇形/空腔边界保持，局部纹理和细亮界面发生变化，未见明显结构崩坏或大面积错误亮斑。不要把两个检查样本概括为所有帧都改善。

- Liver corrected: `anchors/liver/vis/item_23/compose.png`
- Arm corrected: `anchors/arm/vis/item_41/compose.png`
- Legacy paired paths 和选择规则均在 `anchor_impact.json`。

这两个 corrected anchors 可作为后续 main/full-MP/Joint/internal 默认点的 FINAL-REUSE，不需要再训练。此判断来自相同 production/config/test source 与 binary hashes，而不是把 legacy 结果强行复用。

## 6. Final-paper reuse audit

| 旧证据 | 旧结果状态 | Corrected final target |
|---|---|---|
| Main comparison Ours 四数据集 | LEGACY-REFERENCE | RERUN 78（包括本次 2 anchors） |
| Joint 8 rows × 2 datasets | LEGACY-REFERENCE | RERUN；与 main full MP 去重 |
| PPP MP / PP-S / PP-A / PPP | LEGACY-REFERENCE | RERUN；MP/PPP 与 main/Joint 去重 |
| PPP multi-sweep MP/PPP | LEGACY-REFERENCE | RERUN；MP 取 corrected main，PPP representative 取 Joint |
| LGR internal、CGA internal | LEGACY-REFERENCE | 原有对应行 RERUN；Plane/Fixed/RGD+OP NEW |
| CGA population trajectory | LEGACY-REFERENCE | 从 corrected CGA 五行重新提取，无额外训练 |
| rho sensitivity | LEGACY-REFERENCE | representative curve RERUN，默认点复用 corrected Joint |
| LGR hyperparameters | LEGACY-REFERENCE | 对应点 RERUN；新增 window/gamma 点 NEW |
| CGA hyperparameters（包括 Stage-1 accumulated-A_g） | LEGACY-REFERENCE | RERUN；不能因 A_g 公式未改而复用旧训练 |
| Baselines | FINAL-REUSE | 不重跑 |

已盘点的 538 个旧 ablation runs 全部使用 us_focus；全部受错误 local covariance 路径影响。`legacy_ablation_inventory.json`、`legacy_main_inventory.json` 保存逐行判定。LEGACY-REFERENCE 描述旧文件可做什么，RERUN 描述最终替代证据需要做什么，两者不是矛盾标签。不存在旧 Ours 数值被直接 FINAL-REUSE 的情况。

额外追溯：这些旧 ablations 对应的六个 tracked commits（2834e7ec、4961acd2、4fe88c2a、535f5ab1、a0e2b662、db6ea145）都保留了错误的 `M[i*3+0] * Rg[j]` 乘法索引；78 个 main configs 也全部为 us_focus。这是 tracked-source/config 证据，不冒充对所有历史 dirty binary 的逐位重建。

## 7. Corrected run matrix（未授权执行）

实际 production main protocol：Liver 6、Spine 9、Arm 33、Breast 30，总计 78。交换 Results 的 Spine 8/Arm 48 与已执行 manifests 不一致，已记录待改。

| 独立 training runs | Full core | Minimal |
|---|---:|---:|
| Main comparison MP Ours | 78 | 78 |
| 去重后 structural：Joint/PPP/LGR/CGA | 30 | 30 |
| Multi-sweep PPP 额外行 | 14 | 14 |
| Representative rho 非默认点 | 12 | 12 |
| LGR hyperparameters | 38 | 16 |
| CGA hyperparameters | 82 | 40 |
| 总计（包括本次两个 anchor） | 254 | 190 |
| 两个 anchor 完成后剩余 | 252 | 188 |

Structural 30 = Joint 新增 14 + PP-S/PP-A 4 + LGR Plane/Fixed/Simple 6 + CGA RGD/RGD+OP/RGD+LP 6。CGA population 五行和主比较 qualitative/FPS 从保留的 corrected logs/checkpoints 重建，不增加 training run。

Main、renderer/CGA internal 和其 hyperparameter grids 固定 MP，便于隔离 renderer/adaptation；Joint 跨 MP/PPP，PPP internal/multi-sweep/rho 则明确切换 pose 条件。这是本计划的协议选择，不把 MP anchor 冒充 PPP 结果。

Full optional multi-sweep rho：再加 24 个非默认点，总计 278、anchor 后 276。Full core 和 Full+optional 单独列明，默认不执行 optional。

Minimal 不删除 evidence category：LGR figure 用 beta/lambda/Nq/window 四 panel；Full 增加 Tmin/gamma 共六 panel。CGA 两者都保留 qd×B、qp×qA、tau_alpha、split threshold/population。Minimal grids：qd={.94,.98,.99} × B={.02,.05,.12}；qp={.05,.1,.2} × qA={.6,.8,.9}。是否出现 plateau 由数据判断，不预先保证。

Beta=0/lambda=0 在 corrected tests 下等价于 adaptive local-only + local-aware fallback，作为同一 corrected row 的消费者，不重复训练。所有默认点按 corrected main/Joint/internal rows 去重，旧 Stage-1 默认点不再 FINAL-REUSE。

精确 overrides、main 的完整 scene/metric config、消费者关系、授权标记见 full_run_matrix.json / minimal_run_matrix.json；optional_full_multisweep_rho.json 单独列 24 点。它们是审阅计划，不是已提交任务。

终点审核后的 target 分类：Full = FINAL-REUSE 2（本次 corrected anchors）+ RERUN 228 + NEW 24；Minimal = FINAL-REUSE 2 + RERUN 174 + NEW 14。最终 JSON 已将所有 execution_authorized 设为 false，避免把已完成 anchor 的旧授权当成再次提交授权；原 legacy source 仍明确为 LEGACY-REFERENCE。

### GPU 时间预算（规划区间，不是 corrected 全矩阵实测）

历史实际 78 个 main runs 合计 13.4286 worker-hours。对 corrected main 预留 0–50% 不确定性，估为 13.43–20.14 worker-hours；其他 ablation/grid 按既有 8–12 min/worker-run 估算。两个 corrected anchor 的独占 GPU wall 为 7.10/5.45 min，没有出现明显性能异常，但不能据此精确推断 Spine/Breast 或并发训练耗时。

| 计划（包括本次两 anchors） | Runs | Worker-hours | 6-worker ideal elapsed | 3-GPU / 2-worker-per-GPU GPU-hours |
|---|---:|---:|---:|---:|
| Main comparison only | 78 | 13.43–20.14 | 2.24–3.36 h | 6.71–10.07 |
| Full core | 254 | 36.90–55.34 | 6.15–9.22 h | 18.45–27.67 |
| Minimal | 190 | 28.36–42.54 | 4.73–7.09 h | 14.18–21.27 |
| Full + optional multi-sweep rho | 278 | 40.10–60.14 | 6.68–10.02 h | 20.05–30.07 |

上述不含 queueing、失败重试、最终 figure assembly 和完整 main FPS 后处理；六 worker 是理想吞吐估算，任务负载不均衡、Plane candidate 增长或必须单 worker/GPU 时会变长。0–50% 是明确的规划余量，不是实测的 covariance slowdown。还需提交的 runs 分别是 main 76、Full 252、Minimal 188、Full+optional 276；本次已完成训练占约 .209 GPU-hours（不含 tests/compile/FPS）。

## 8. Main table 与论文输出 gate

两个 representative anchors 不能形成四数据集 main row。其余 76 个 main runs 未获准执行，因此不能现在生成 corrected 四数据集 aggregate，不能把旧 Ours 的 table/FPS/qualitative 换个标签继续使用。后续批准后：保持 baseline 原数据不变；汇总全部 corrected main sweeps；从同一 corrected checkpoints 按既有 5 warmup / 20 frames / 3 passes benchmark 重测 FPS；按固定 case/frame manifest 重建 qualitative；更新训练时间与所有 Ours 数值和描述。

`docs/production/p0_semantic_freeze/paper_alignment.patch` 已通过 `git apply --check`（对交换版内容作只读检查），但未应用。它是 semantic patch；其余旧 numerical Results 必须等 corrected evidence 完整后替换，不能当作已经更新完毕的最终稿。

此外，原 Results 的固定 150-frame Spine 描述与本地导出例如 152/153/151 frames 不完全一致；canonical 工程到位后应依据 `dataset_descriptions.md` 和逐 sweep manifests 一并校对，不能将旧 dataset 描述原封不动带入最终稿。

## 9. 完成状态

Covariance correction、全路径核验、tests/smoke、两 anchors、matched impact comparison、FPS、source freeze 和重跑规划已完成。Liver 原 job 79917 在训练完成后的审计包装脚本退出；checkpoint 独立验证后保留结果。恢复 job 79923 仅训练 Arm，并完成两个 checkpoint 的 FPS。没有重复 Liver 30k，没有提交剩余正式矩阵。
等待用户决定下一批 runs、Full/Minimal grid 密度及 canonical LaTeX 工程位置。Baseline 结果保持不动；旧 Ours 不进入 corrected final tables/figures。
