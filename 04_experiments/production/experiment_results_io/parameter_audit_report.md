# 论文方法参数—代码实现—实验消融完整审计

审计日期：2026-08-31  
审计范围：`docs/production/experiment_results_io/03_method.tex`、`04_experiments_and_results.tex`，Ours 的生产配置、训练/渲染/CUDA renderer、pseudo-pose、Gaussian adaptation、生产脚本及已有结果。  
约束：本审计未修改代码、未启动新训练或昂贵实验；全部结论来自仓库现有源码、配置、日志汇总和结果文件。

## 0. 执行摘要

最重要的结论如下。

1. **论文当前写的 `k_sigma=2` 不是 local renderer 的统一截断参数。**生产配置中：
   - local mapper 使用 `model.slice_render.k_sigma=2.5`；
   - global ray candidate 使用 `model.slice_render.ray_attenuation_candidate_k_sigma=2.0`。
   因此 Experiments 的 `k_sigma=2` 实际对应 global candidate support，最接近 Methods 的 $\kappa_{\mathrm{glo}}$，而不是 $\kappa_{\mathrm{ele}}$ 或 $\kappa_{\mathrm{img}}$。
2. **Methods/Experiments 的面积补偿指数与代码不一致。**论文写 $\gamma=2$，生产代码为 `us_focus_det_gamma=0.25`，实际因子是 $(\det\Sigma_{2D}+10^{-6})^{-0.25}$。
3. **Methods 的 local hard support 与代码不是同一机制。**当前 `us_focus` 路径没有实现 $r^{\mathrm{ele}}\le\kappa_{\mathrm{ele}}^2$ 的硬截断，也没有逐像素执行 $d^\top\Sigma^{-1}d\le\kappa_{\mathrm{img}}^2$；代码用 `k_sigma=2.5` 构造 2D AABB，再用 `us_focus_weight_min=10^{-6}` 去除小权重。
4. **“2-mm integration window”在代码中是半径，不是总宽度。**`ray_attenuation_quadrature_window_mm=2.0` 被当作 `quadrature_window_radius_mm`，积分区间为最接近 Gaussian 的射线位置 $t_c$ 周围 $[t_c-2,t_c+2]$ mm，再裁到 $[0,t_p]$；最大总宽度是 4 mm。Methods 当前却写从 $z_{near}$ 到像素的完整积分。
5. **$N_q=3$ 是三点 Gauss–Legendre quadrature，不是 uniform sampling。**节点为 $0,\pm0.7745966692$，权重为 $8/9,5/9,5/9$。
6. **Methods 的 global-path pruning protection 未由当前生产代码实现。**论文定义 $A_g=\sum(1-T_{igp})$ 并以固定 $\tau_A$ 判断；生产默认实际是 `protect_attenuation_mode=high_tau`，保护 $\tau_g$ 位于第 80 百分位以上的 Gaussian。它既不计算论文的 $A_g$，也没有固定 $\tau_A$，而且高 $\tau_g$ 表示较高 transmission floor，并不等价于“强 attenuation/path effect”。
7. **生产 adaptation 的真实默认值**是 $q_d=0.98$、$\tau_\omega=10^{-6}$、$B=0.05N$、split threshold 0.6 mm（取三轴 `max(scale)`）、$q_p=0.10$、$\tau_\alpha=10^{-4}$；$\tau_A$ **未实现/未找到**，替代配置是 `protect_attenuation_quantile=0.80` 和 `high_tau` mode。
8. **“top 2% + 5% budget”不是两个都紧约束。**代码先在正分数中取 0.98 quantile，再用 5% population top-k threshold 作为额外上限阈值；通常 2% quantile 更严格。由于最终使用 `score >= threshold`，阈值并列时数量理论上可超过预算，因此不是严格 `TopK_B`。
9. **split 的真实拓扑操作也应谨慎描述。**代码保留父 Gaussian，并追加两个子 Gaussian；在 bounded-sigmoid scale 参数化下，代码对 raw scale 减 `log(2)`，并用 `exp(raw)` 生成扰动尺度，这不等价于把物理 scale 精确减半。论文不应声称标准 3DGS 式“替换父点”的 split，除非代码后续统一。
10. **现有 sensitivity 数据足够支持三块主文结构，但 renderer 不是二维 heatmap。**真实已有：
    - pseudo-pose：$\rho$ 一维 sweep；
    - renderer：$\lambda_{ray}$、$\beta$、$T_{min}$、$N_q$、$\kappa_{glo}$ 分别 one-at-a-time sweep；没有 $(\beta,\lambda_{ray})$ 二维笛卡尔网格；
    - adaptation：$(q_d,B)$ 二维 heatmap，另有 $(q_p,q_{protect})$ heatmap 和 $\tau_\alpha$ one-at-a-time sweep。
11. **Results 的主比较数字基本准确；最大问题是 protocol/术语。**主比较 Ours 生产配置使用 `pose_mode=file`，而论文方法主张 sensorless。现有 pseudo-pose ablation 能证明 uniform canonical pose 的代表性结果，但主比较本身不是 sensorless Ours。正文还说“common-pose protocol”，Comparison Methods 小节却没有真正定义该 protocol。
12. **补入的五张最终 LaTeX 表与源 Markdown/TSV 的数值一致，没有发现二次转录错误。**主比较表的加粗也与 caption 所说的“按未四舍五入值判定”一致；但表中把源结果的 `NeRF-US w/o prior` 简写成 `NeRF-US`，容易让读者误以为使用了带 diffusion prior 的完整版本。项目 protocol 明确要求标成 `NeRF-US w/o prior`，所以最终表/正文必须恢复该 variant 名称。

---

## 1. 配置层级与“默认值”的口径

仓库同时存在 dataclass 默认、generic YAML 默认、`ours_us_focus` 默认、preproduction wrapper 和最终 production override。本文后续表格的“默认值”除非特别说明，均指**最终论文生产/消融 anchor 的 resolved value**，主要由下列链路决定：

```text
configs_hydra/experiment/production/ablations/ours/base_1m30k.yaml
  -> configs_hydra/experiment/production/comparisons/
     us_recon_four_dataset_p100k_v1/gs_strict/ours.yaml
  -> configs_hydra/experiment/preproduction/ideas/ours/
     responsibility_density_control/fixed_protocol_main_30k.yaml
  -> configs_hydra/model/ours_us_focus.yaml
```

最终论文 sensitivity/expanded ablation 还由
`scripts/production/ablations/ours/submit_expanded_paper_plan_multiworker_1m30k.sh`
显式覆盖。旧脚本 T02/T05/T06 曾使用 `slice_depth, q=1, beta=0.60, strength=0.30, Tmin=0.55`；这些是历史运行入口，不是当前 T01–T04/T14/T15 论文配置。当前 production anchor 是 `integral, q=3, beta=0.80, strength=0.55, Tmin=0.35`。

---

## 2. Methods 参数总表

### 2.1 Acquisition-guided pseudo-pose

| 论文符号/名称 | 含义 | 代码变量名 | 生产默认值 | 定义/读取位置 | 是否可学习 | 是否应写入 Implementation Details | 是否已做消融 |
|---|---|---|---|---|---|---|---|
| $d_0$ | nominal inter-frame spacing | `scene.loading.canonical_sweep_delta_mm`; 若 $\le0$ 则 `DatasetMeta.delta_elevational_mm` | 默认 `-1`，实际回退到数据集 meta：Liver 0.2685 mm；Spine 0.79 mm；TUS 0.404 mm；TDSC 0.475674 mm | `configs/base/scene.py:59-62`; `scene/container/single_sweep_container.py:258-263`; `scene/meta/dataset_meta/*.py` | 否 | 必须说明其来源，但它是校准量，不是调参超参数 | 未直接 sweep；只 sweep $\rho$ |
| $\rho$ | nominal spacing 的全局比例 | `canonical_sweep_delta_scale` | 1.0 | `configs/base/scene.py:61-62`; `configs_hydra/experiment/production/sensorless/ours_canonical.yaml:22`; `single_sweep_container.py:262-263` | 否 | 是 | 是，0.60–1.50 |
| $\mathbf u$ | canonical scan direction | `canonical_sweep_axis` | `z`，即 $(0,0,1)$ | `single_sweep_container.py:275-290`; sensorless YAML | 否 | 是，但属于坐标约定 | 有 axis/符号 negative controls；主表含 file-axis diagnostic |
| $i_0$ | reference frame | `canonical_sweep_ref_frame` | 0，即当前 retained sequence 的第一帧 | `single_sweep_container.py:265-273` | 否 | 是，但属于 gauge/坐标定义 | 无独立 sweep |
| spacing mode | spacing sequence 来源 | `canonical_sweep_spacing_mode` | `uniform` | `single_sweep_container.py:292-321` | 否 | 是 | 是：uniform / file-step-norm / file-axis-projection |
| frame selection | pseudo-pose 作用的帧序列 | `frame_start/end/stride`, `frame_indices_path` | 全序列、stride 1；具体 production 由数据导出/脚本决定 | `single_sweep_container.py:133-196` | 否 | Supplementary/数据协议 | 无独立 sensitivity |
| pose orientation | 所有 canonical frame 的旋转 | 代码直接用 `torch.eye(4)` | 恒等旋转 | `single_sweep_container.py:323-332` | 否 | 是，属于方法定义 | 与 file pose 在 joint/T04 比较 |
| pose optimization | 是否再优化 pseudo-pose | `model.pose_optimization.enabled` | false | sensorless YAML:26-28；`methods/ours/pose_optimization.py` | 否（生产）；模块本身支持可学习 | 必须明确“未优化” | 当前主实验未作为参数 sweep |

注意：代码只允许 `x/y/z` 及正负号，不支持 Methods 所写的任意连续单位向量 $\mathbf u\in\mathbb R^3$。另外，uniform mode 对 retained array 使用局部 `arange(n)`，即使 `frame_indices_path` 跳过了原始帧，也不会按原始 frame-id gap 放大间距。

### 2.2 Local renderer

| 论文符号/名称 | 含义 | 代码变量名 | 生产默认值 | 定义/读取位置 | 是否可学习 | 是否应写入 Implementation Details | 是否已做消融 |
|---|---|---|---|---|---|---|---|
| $\sigma^{ele}_{ig}$ intercept | depth-adaptive elevational width 的基值 | `us_focus_sigma_ele_0` | 1.0 mm | `configs_hydra/model/ours_us_focus.yaml`; `us_focus_device.cuh:213-226` | 否 | 必须 | 未做正式 sweep |
| $\sigma^{ele}_{ig}$ slope | 随 axial depth 的线性斜率 | `us_focus_sigma_ele_1` | 0.01（mm/mm） | 同上 | 否 | 必须 | 未做正式 sweep |
| $\sigma^{ele}$ lower/upper bound | elevational width 的 clamp | `us_focus_sigma_ele_min/max` | 0.8 / 5.0 mm | 同上 | 否 | 必须 | 未做正式 sweep |
| lateral focus width | 可选 lateral depth focus | `us_focus_sigma_lat_0/1/min/max` | 0.8, 0.002, 0.4, 3.0 mm | `ours_us_focus.yaml`; `us_focus_device.cuh:218-226` | 否 | 当前 `use=false`，可放 Supplementary | 未做主文 sweep |
| lateral focus switch | 是否把 lateral focus precision 加入 kernel | `us_focus_use_lateral_focus` | false | `us_focus_device.cuh:222-235` | 否 | 可省略，但应说明关闭 | 否 |
| $\gamma$ | footprint determinant compensation exponent | `us_focus_det_gamma` | **0.25** | `ours_us_focus.yaml`; `us_focus_device.cuh:99-112,374-381` | 否 | 必须 | 未做正式 sweep；论文误写为 2 |
| $\varepsilon$ | covariance/determinant numeric epsilon | `us_focus_det_eps`; `_PROBE_COV_JITTER` | $10^{-6}$；covariance jitter $10^{-6}$ | `ours_us_focus.yaml`; `us_focus_reference.py:26,147-151` | 否 | Supplementary | 否 |
| local $k_\sigma$ / AABB support | conditioned 2D footprint 的 mapper AABB 半径倍率 | `slice_render.k_sigma` | **2.5** | `ours_us_focus.yaml: k_sigma`; `us_focus_device.cuh:312-315`; `core_tile.cu:122-157` | 否 | 必须 | 未在当前 sensitivity 正式 sweep |
| $\kappa_{ele}$ | Methods 的 normalized elevational hard cutoff | **无直接实现** | 未找到 | `core_tile.cu:86-89` 对 `us_focus` 明确跳过传统 elevational hard cull；由 slice response + weight threshold 软抑制 | — | 论文应删除或按代码重写 | 否 |
| $\kappa_{img}$ | Methods 的 2D Mahalanobis hard cutoff | **无一一对应实现**；最接近 local `k_sigma=2.5` | 2.5 仅用于 AABB，不是 `quad <= k^2` | `us_focus_device.cuh:359-383` | 否 | 需要解释映射，不能直接写成硬截断 | 否 |
| local response cutoff | 小 local weight 丢弃阈值 | `us_focus_weight_min` | $10^{-6}$ | `ours_us_focus.yaml`; `us_focus_device.cuh:374-383` | 否 | 建议 Supplementary；比未实现的 $\kappa_{ele}$ 更真实 | 否 |
| footprint inflation | 对 2D covariance 额外加宽 | `us_focus_inflate_lat_mm`, `us_focus_inflate_ax_mm` | 0 / 0 mm | `ours_us_focus.yaml`; `us_focus_device.cuh:256-260` | 否 | 可省略/附录 | 否 |
| calibrated corridor half-width | global ray candidate 的 elevational corridor；local mapper 仅在非 `us_focus` weight mode 执行该 hard cull | `elev_corridor_half_mm` | Liver/Spine 0.395 mm；TUS 0.202 mm；TDSC 0.238 mm | inherited fixed protocol + comparison/ablation runner overrides；`core_tile.cu:86-89`；`ray_attenuation.cu:468-473` | 否 | 必须，尤其存在 dataset-specific 值 | 有历史 corridor diagnostics，不在当前主 sensitivity 图 |
| slab extra half-thickness | corridor 额外厚度 | `slab_half_thickness_mm` | 0 | `ours_us_focus.yaml` | 否 | Supplementary | 否 |
| tile size | local/global sparse mapper tile | `tile_size_px` | 8 px | `ours_us_focus.yaml`; `core_tile.cu` | 否 | Supplementary | 否 |
| composition mode | local accumulation | `compositing_mode` | additive | `ours_us_focus.yaml`; renderer type checks | 否 | 必须写“additive” | renderer component ablation间接验证，非参数 sweep |
| background | fixed output background | `fixed_bg_color` | 0 | `ours_us_focus.yaml` | 否 | 可省略 | 否 |

### 2.3 Global renderer

| 论文符号/名称 | 含义 | 代码变量名 | 生产默认值 | 定义/读取位置 | 是否可学习 | 是否应写入 Implementation Details | 是否已做消融 |
|---|---|---|---|---|---|---|---|
| $z_{near}$ | 射线起始 axial depth | `DatasetMeta.near_mm` → `axial_near_mm` | 四个当前 dataset 均为 0 mm | `methods/ours/render.py:22`; dataset meta files | 否，数据校准量 | 说明来自 calibration/meta；不作为 hyperparameter | 否 |
| $\kappa_{glo}$ | global ray candidate lateral/elevational support | `ray_attenuation_candidate_k_sigma` | **2.0** | production `ours.yaml`; `wrapper_shader.cu:346-371`; `ray_attenuation.cu:468-494` | 否 | 必须 | 是：-1, 1.5, 2, 3；-1 表示回退 local k=2.5 |
| $N_q$ | ray integral quadrature 点数 | `ray_attenuation_quadrature_points` | 3 | production YAML; `ray_attenuation.cu:137-167` | 否 | 必须 | 是：1,2,3 |
| integration window | closest-point 局部积分窗口的**半径** | `ray_attenuation_quadrature_window_mm` | 2.0 mm radius（最大 4-mm full width） | `types.py:66`; `ray_attenuation.cu:123-136` | 否 | 必须，且应改正“2-mm window”措辞 | 当前主 sensitivity 未 sweep |
| $\lambda_{ray}$ | Gaussian overlap attenuation strength | `ray_attenuation_strength` | 0.55 | production YAML; `ray_attenuation.cu:691-700` | 否 | 必须 | 是：0, .20, .35, .55, .75, 1.0 |
| $T_{min}$ | product transmission floor | `ray_attenuation_min_transmission` | 0.35 | production YAML; `ray_attenuation.cu:704-705` | 否 | 必须 | 是：.20,.35,.50,.65,.80 |
| $\beta$ | local-global modulation strength | `ray_attenuation_beta` | 0.80 | production YAML; `ray_attenuation.cu:704-705` | 否 | 必须 | 是：0,.4,.6,.8,1 |
| $\tau_g$ | learnable per-Gaussian transmission floor | `gauss_container.attenuations` | init Uniform[0.016,0.024]；范围 $(0,1)$ | `ours_us_focus.yaml`; `gaussian_container.py:74-79,104-107` | 是 | 写参数化/范围和初始化；不是固定 hyperparameter | adaptation protection ablation间接涉及 |
| ray response cutoff | 忽略小 `strength*integral` | `ray_attenuation_weight_min` | $10^{-4}$ | `ours_us_focus.yaml`; `ray_attenuation.cu:694-697` | 否 | Supplementary | 否 |
| pre-pixel margin | 允许 Gaussian center 超过 pixel 的 axial margin | `ray_attenuation_axial_margin_mm` | 0 mm | `ours_us_focus.yaml`; `ray_attenuation.cu:657,675` | 否 | Supplementary | 否 |
| ray mapper | global candidate / sorting path | `ray_attenuation_mapper_kind` | `ray_csr`（由 inherited fixed protocol） | fixed protocol YAML; `wrapper_shader.cu` | 否 | Supplementary，但复现时应记录 | renderer design ablation间接涉及 |
| integral formula | global response family | `ray_attenuation_formula` | `integral` | production YAML | 否 | 必须 | 是：local-only、simple `slice_depth`、integral q=1/q=3 |
| ray attribute source | 使用独立 $\tau_g$ 还是 opacity | `ray_attenuation_source` | `attenuation` | inherited fixed protocol YAML; `methods/ours/render.py:89` | 否 | 必须 | 旧 diagnostics 中有，当前主表固定 |
| ray bin width | lateral candidate bin 粒度 | 共享 `tile_size_px` | 8 px | `ray_attenuation.cu:419-495` | 否 | Supplementary | 否 |

### 2.4 Contribution-aware Gaussian adaptation

| 论文符号/名称 | 含义 | 代码变量名 | 生产默认值 | 定义/读取位置 | 是否可学习 | 是否应写入 Implementation Details | 是否已做消融 |
|---|---|---|---|---|---|---|---|
| $q_d$ | positive responsibility score 的高分位 | `responsibility.score_quantile` | 0.98 | `methods/ours/configs.py`; production/expanded script | 否 | 必须 | 是：.94,.96,.98,.99 |
| $\tau_\omega$ | 最小 responsibility score | `responsibility.min_score` | $10^{-6}$ | `configs.py`; `density_control.py:277-286` | 否 | 必须或 Supplementary | 否 |
| $B$ | 单次 densify population fraction | `responsibility.max_fraction` | 0.05（当前 population 的 5%） | `density_control.py:279-284` | 否 | 必须 | 是：.02,.05,.08,.12，与 $q_d$ 二维扫 |
| residual quantile | 高 residual hit diagnostic 阈值 | `high_residual_quantile` | 0.90 | `dc_diagnostics.py:109-127` | 否 | Supplementary | 未正式 sweep；而且 `max_residual_hits` 不参与当前 score/selection |
| split threshold | split/clone 的物理尺度门槛 | `thresholds.split_scale` | 0.6 mm | `configs.py`; `density_control.py:394-396` | 否 | 必须 | design ablation间接；无 threshold sweep |
| split scale statistic | 判断大/小 Gaussian 的尺度 | `torch.max(scales,dim=1).values` | 三轴物理 scale 的最大值 | `density_control.py:394-396` | — | 必须澄清 | 否 |
| clone lower scale | clone eligibility 下限 | hard-coded `min_scale_limit=0.01` mm | 0.01 mm；但 bounded scale min=0.05，生产中恒满足 | `density_control.py:360-396` | 否 | 可省略/说明无实际约束 | 否 |
| $q_p$ | low local contribution quantile | `low_contribution_quantile` | 0.10 | production YAML; `density_control.py:463-466` | 否 | 必须 | 是：.02,.05,.10,.15,.20 |
| $\tau_\alpha$ | weak local echo amplitude threshold | `thresholds.prune_opacity`；`opacities` alias `echo_alphas` | $10^{-4}$ | production YAML; `gaussian_container.py:112-118`; `density_control.py:443-455` | 否 | 必须 | 是：$10^{-5},5\cdot10^{-5},10^{-4},5\cdot10^{-4},10^{-3}$ |
| $\tau_A$ | Methods 的 fixed global path contribution threshold | **无对应变量** | **未找到/未实现** | 当前 protection 使用 quantile，不计算 $A_g$ | — | 不能按固定值写 | 否 |
| global protection quantile | 当前真实 protection threshold | `protect_attenuation_quantile` | 0.80 | production YAML; `density_control.py:468-488` | 否 | 必须按真实逻辑写 | 是：.60,.70,.80,.90，与 $q_p$ 二维扫 |
| global protection mode | protection score 的定义 | `protect_attenuation_mode` | `high_tau` | dataclass default；sensorless YAML 显式 `high_tau` | 否 | **必须**；它决定语义 | 有 on/off design ablation；`path_effect` 在后续 sensorless 脚本出现，但不是当前主表默认 |
| prune intensity | legacy dark threshold | `thresholds.prune_intensity` | 0.015 | `ours_us_focus.yaml`; `density_control.py:447-455` | 否 | 不应写为有效参数：Ours `intensities` 恒为 1，因此生产中无效 | 否 |
| max scale prune | 大 Gaussian 直接删除 | `thresholds.max_scale_mm` | 0（关闭） | `configs.py`; `density_control.py:500-504` | 否 | Supplementary/说明关闭 | 否 |
| adaptation schedule | start/end/frequency | `timing.start_iter/end_iter/freq` | 1000 / 25000 / 2000 | production `ours.yaml`; `density_control.py:161-173` | 否 | 必须 | 有 schedule 历史实验，当前主文固定 |
| density strategy | contribution vs gradient | `density_control.strategy` | `contribution_residual` | production YAML | 否 | 必须 | 是：gradient vs responsibility variants |
| densify/prune switches | 是否执行两个阶段 | `enable.densify/prune` | true / true | model YAML | 否 | 必须/表中说明 | 是：densify-only、no-protection、full |
| mix mode | 是否并入 gradient candidates | `responsibility.mix_mode` | `responsibility` | fixed protocol / dataclass | 否 | Supplementary | 当前主表固定；union 分支未使用 |

### 2.5 Optimization、初始化和数据采样

| 论文符号/名称 | 含义 | 代码变量名 | 生产默认值 | 定义/读取位置 | 是否可学习 | 是否应写入 Implementation Details | 是否已做消融 |
|---|---|---|---|---|---|---|---|
| initial Gaussian count | 初始点数 | `model.gauss_init.num_points`; 同步写入 `scene.sampler.num_init` | $10^6$ | production YAML; `gs_only_callbacks.py:259-267` | 否 | 必须 | 固定 protocol；无主文 sweep |
| iterations | 训练步数 | `engine.iterations` | 30,000 | production YAML | 否 | 必须 | 固定 |
| optimizer | 所有 Gaussian 参数 optimizer | `torch.optim.Adam` | Adam，PyTorch default betas=(0.9,0.999)，`eps=1e-15`，无 weight decay | `gaussian/gauss_solver.py:33-47` | — | 必须写 Adam；eps 可附录 | 否 |
| center LR | center learning rate | `gauss_container.lr.means` | $1.6\times10^{-4}$ → $1.6\times10^{-5}$ | `ours_us_focus.yaml`; `gaussian_solver.py:18-27` | — | 必须 | 有前期 tuning，非当前主 ablation |
| scale LR | scale raw parameter LR | `lr.scales` | $5\times10^{-3}$ constant | `ours_us_focus.yaml` | — | 必须 | 否 |
| rotation LR | quaternion raw parameter LR | `lr.quats` | $10^{-3}$ constant | 同上 | — | 必须 | 否 |
| $\alpha_g$ LR | echo amplitude raw parameter LR | `lr.echo_alphas` | **$8\times10^{-2}$** constant | 同上 | — | 必须 | 否 |
| $\tau_g$ LR | transmission raw parameter LR | `lr.attenuations` | $10^{-2}$ constant | 同上 | — | 必须 | 否 |
| scale bounds | activated physical scales | `scale_activation=bounded_sigmoid`; `scale_min/max_mm` | [0.05,3.0] mm（sigmoid 实际开区间） | `gaussian_container.py:35-59`; `ours_us_focus.yaml` | scale 可学习 | 必须 | bounded-scale diagnostics 已做；主表固定 |
| $\alpha_g$ range/init | activated echo amplitude | `echo_alpha_max`, `init_val.echo_alphas` | range $(0,0.2)$；init Uniform[0.08,0.12] | `gaussian_container.py:65-76,94-103` | 是 | 必须 | 否 |
| $\tau_g$ range/init | activated transmission floor | `attenuation_max`, `init_val.attenuations` | range $(0,1)$；init Uniform[0.016,0.024] | 同上 | 是 | 必须 | 否 |
| initial scale | 所有初始 Gaussian 尺度 | `scene.sampler.scales_init` | [0.3,0.3,0.3] mm | `configs_hydra/scene/*.yaml`; `scene/scene.py:209-216` | 初始化后可学习 | 必须或 Supplementary | 否 |
| initial rotation | 初始 quaternion | `quats_init` | [1,0,0,0] (wxyz) | scene YAML; `scene/scene.py:179-181` | 初始化后可学习 | Supplementary | 否 |
| initial sampler mix | uniform / intensity importance 比例 | `uniform_ratio` | 0.30 / 0.70 | `configs/base/scene.py:82`; `unified_sampler.py:47-63` | 否 | 重要但当前论文未写；至少 Supplementary | 否 |
| init intensity threshold | importance candidates | `intensity_threshold` | 0.05（输入已归一化到 [0,1]） | `unified_sampler.py:87-105`; `SliceMeta:44-55` | 否 | Supplementary | 否 |
| init jitter | sampled center 的均匀位置抖动 | `jitter_radius` | 0.2 mm/axis | `unified_sampler.py:74-77` | 否 | Supplementary | 否 |
| frame sampling | 每 iteration 选一完整 frame | offline scheduler / holdout train indices | 无放回 shuffled cycle，seed=42；每轮结束 reshuffle | `scene/scheduler/offline_scheduler.py:13-50`; `scene/scene.py:151-159` | 否 | 必须 | 否 |
| batch / accumulation | 每步 frame 数和梯度累积 | `accum_steps` | 1 | engine YAML | 否 | 必须 | 否 |
| loss $\lambda$ | $0.8L_1+0.2(1-SSIM)$ | `loss.lambdas.l1/ssim` | 0.8 / 0.2，即论文 $\lambda=0.2$ | production YAML | 否 | 必须 | loss tuning 历史有，主文固定 |
| SSIM implementation | structural loss window | `SSIMLoss` defaults | window 11，sigma 1.5 | `utils/loss/image_loss.py` | 否 | Supplementary | 否 |
| holdout protocol | evaluation split | `test_mode`, `holdout_test_ratio`, `holdout_trim_each` | `holdout_interval`, 20%，两端各 trim 10 | production runner | 否 | 必须 | 固定 |
| image resolution | approximately $10^5$ px | dataset runtime meta / export | Liver 448×224；Spine 896×112；TUS 288×384；TDSC 280×356 | dataset meta files | 否 | 必须 | 固定 |

---

## 3. Methods 与代码一致性重点核查

### 3.1 $\sigma^{ele}_{ig}$ 如何随 depth 变化

实际 CUDA/reference 公式一致：

```text
depth_ig = max(mu_probe_axial_ig - z_near, 0)
sigma_ele_ig = clamp(
    us_focus_sigma_ele_0 + us_focus_sigma_ele_1 * depth_ig,
    us_focus_sigma_ele_min,
    us_focus_sigma_ele_max
)
```

生产值：

```text
sigma_ele_ig = clamp(1.0 mm + 0.01 * depth_ig, 0.8 mm, 5.0 mm)
```

证据：`ops/ours_renderer/include/shader/us_focus_device.cuh:213-226` 和
`ops/ours_renderer/ours_renderer/us_focus_reference.py:117-129`。由于 intercept 1.0 已高于 min 0.8，当前 $z_{near}=0$ 且 depth 非负时 lower bound 实际不会触发；upper bound 在 depth 约 400 mm 才触发，当前成像深度也通常不会触发。因此现有数据上它基本是 $1+0.01z$ mm。

### 3.2 `k_sigma=2` 到底对应哪个符号

明确结论：Experiments 的 `k_sigma=2` 对应
`ray_attenuation_candidate_k_sigma=2.0`，最接近 $\kappa_{glo}$。

- Local `slice_render.k_sigma` 来自 `ours_us_focus.yaml`，值为 2.5。
- Global wrapper 若 `ray_candidate_k_sigma>0` 则使用该值，否则回退 local k；见 `wrapper_shader.cu:346-371`。
- 当前 renderer sensitivity 中 `candidate_k_sigma=-1` 就是回退 local 2.5，而不是“无限 support”。
- $\kappa_{ele}$ 没有直接实现；$\kappa_{img}$ 也没有代码中的逐像素 Mahalanobis hard cutoff。

建议统一方式：论文分别写 `local footprint AABB multiplier k_loc=2.5` 和
`global ray-candidate multiplier κ_glo=2.0`，不要继续用未定义且混淆分支的 `k_sigma=2`。

### 3.3 $\kappa_{ele}$、$\kappa_{img}$、$\kappa_{glo}$ 的真实值

| Methods 名称 | 代码实际 | 结论 |
|---|---|---|
| $\kappa_{ele}$ | `us_focus` local mapper 跳过传统 `fabs(mu_ele)>=corridor+...` hard cull；使用 Gaussian slice response 和 `weight_min` | 无真实默认值；Methods hard cutoff 未实现 |
| $\kappa_{img}$ | 最接近 `slice_render.k_sigma=2.5`，但只生成 AABB；不检查 `quad<=2.5^2` | 不能严格称 $\kappa_{img}$；若保留符号需改写定义 |
| $\kappa_{glo}$ | `ray_attenuation_candidate_k_sigma=2.0` | 真实默认 2.0；已 sensitivity |

### 3.4 2-mm integration window 的真实作用

对 pixel 的 pre-pixel ray 长度 $t_p=\max(0,z_p-z_{near})$，代码先在 Gaussian 标准化局部坐标中求射线离 Gaussian 中心最近的参数：

$$
t_c=-\frac{o_g^\top d_g}{d_g^\top d_g+10^{-8}}.
$$

随后实际积分范围是：

$$
t_s=\max(0,t_c-w),\qquad t_e=\min(t_p,t_c+w),\qquad w=2\ \mathrm{mm}.
$$

若 $t_e\le t_s$，贡献为 0。故配置名虽是 `quadrature_window_mm=2.0`，CUDA 参数名和使用方式都证明它是 **radius/half-window**，最大范围为 4 mm。它不是从 $z_{near}$ 到目标 pixel 的完整 integral，Methods Eq. 当前需要按 localized/clipped integral 改写。

### 3.5 $N_q=3$ 的实现

不是 uniform samples，而是映射到 $[t_s,t_e]$ 的 Gauss–Legendre：

- $N_q=1$：中心点，权重 2；
- $N_q=2$：节点 $\pm1/\sqrt3$，等权；
- $N_q=3$：节点 $0,\pm\sqrt{3/5}$，权重 $8/9,5/9,5/9$。

证据：`ray_attenuation.cu:135-167`。值得注意的是现有 sensitivity 中 $N_q=2$ 比 1/3 差约 0.53–0.61 dB，表现异常但确为现有真实结果，不能改写成单调收敛。

### 3.6 adaptation thresholds 的真实值

```text
q_d                  = 0.98
tau_omega            = 1e-6
B / current N        = 0.05
split max-scale      = 0.6 mm
q_p                  = 0.10
tau_alpha            = 1e-4
tau_A                = 未实现
protection substitute= high_tau, quantile 0.80
schedule             = iter 1000..25000, every 2000
```

### 3.7 split threshold 是否真实存在、基于什么尺度

真实存在。判断为：

```text
max_scale = torch.max(container.scales, dim=1).values
split: selected & (max_scale > 0.6)
clone: selected & (max_scale <= 0.6) & (max_scale > 0.01)
```

所以是三轴 activated physical scale 的**最大值**，不是 mean scale、volume 或 projected scale。生产 scale lower bound 0.05 mm 使 clone 的 0.01 mm lower check 实际冗余。

额外实现事实：split 不删除 parent，而是追加两个 children；bounded sigmoid 下 child physical scale 也不是严格 parent/2。论文应避免比代码更强的标准 split 描述。

### 3.8 “top 2%”和“5% budget”的实现

1. score 为 `residual_mean * log1p(contribution)`；`residual_mean=R/(C.clamp_min(1e-8))`。
2. quantile 只在 `score>0` 的集合上计算。
3. 取 $Q_{0.98}$。
4. 计算 `k=max(1,int(N*0.05))`；若 positive count 大于 k，再取全 score 的第 k 大值。
5. 最终 threshold 是 `max(q-threshold, topk-threshold, 1e-6)`；选择 `score >= threshold`。

因此通常是正分数 top 2%，5% 是宽松 cap。并列 score 可能令最终 count 超出 k。Methods 的严格 `TopK_B{...}` 只近似成立。

另外 responsibility statistics 只在 adaptation event iteration 的当前一帧上计算（`should_collect_responsibility_stats` 等于 `should_densify_prune`），不是在前 2000 iterations/所有 frames 累积。论文的 $\sum_p$ 对当前 sampled frame 成立，但若文字暗示多帧累计则不成立。

### 3.9 pruning 的真实 “global-path protection”

生产默认 pruning mask：

```text
weak_alpha = echo_alpha < 1e-4
low_local  = C <= quantile(C, 0.10)
protected  = tau_g >= quantile(tau, 0.80)  # mode=high_tau
prune      = weak_alpha & low_local & ~protected
```

`intensity < 0.015` 也被 OR 入 weak mask，但 Ours 的 `intensities` property 恒返回 1，所以无效。

Methods 写法则是：

```text
alpha_g < tau_alpha & C_g < Q_qp(C) & A_g < tau_A,
A_g = sum(1 - T_igp)
```

两者不一致：当前主生产没有 $A_g$ accumulation，也没有 $\tau_A$。代码虽然支持 `path_effect` mode，用近似 `(1-tau)*support` 做 quantile protection，但当前主表默认仍是 `high_tau`。后续 `sensorless_stage2` 脚本使用 `path_effect`，不能倒推为当前 T01–T15 的默认。

建议统一方式有两种，需作者选择而非本审计代改：

- 按当前代码改 Methods：用 transmission-floor quantile protection，并避免称为 accumulated global path contribution；或
- 将代码统一到 Methods 的 $A_g/\tau_A$ 后重跑受影响结果。

### 3.10 center LR decay

实际 schedule 为 log-linear/exponential：

$$
\mathrm{lr}_{\mu}(t)=1.6\times10^{-4}\cdot 10^{-t/30000},
\quad 0\le t\le30000,
$$

即 step 0 为 $1.6\times10^{-4}$，step 30k 为 $1.6\times10^{-5}$，无 warmup/delay。其他 Gaussian parameter LR 恒定。证据：`methods/ours/gaussian_solver.py:18-27`、`gaussian/lr_scheduler_manager.py:23-54`。

### 3.11 Methods 写了但当前生产代码未按文义使用的参数/逻辑

| 论文写法 | 代码实际 | 建议统一方式 |
|---|---|---|
| arbitrary unit scan direction $\mathbf u$ | 只支持 signed x/y/z | Methods 写成预定义 canonical coordinate axis，或扩展代码后再保留任意单位向量表述 |
| $r^{ele}\le\kappa_{ele}^2$ | `us_focus` 无该 hard cutoff | 删除 $\kappa_{ele}$ hard set；按 slice response + weight threshold 描述 |
| $d^T\Sigma^{-1}d\le\kappa_{img}^2$ | local 只做 k-scaled AABB 和 weight threshold | 将 Methods 改为 sparse AABB candidate + continuous Gaussian weight cutoff |
| $\gamma=2$ | `det_gamma=0.25` | 论文改为 0.25；如果坚持 2 必须重跑 |
| $\int_{z_{near}}^{z_p}$ 完整积分 | closest-point ±2 mm radius 的 clipped integral | 改 Methods 积分上下限/文字 |
| exact ellipse–ray intersection candidate | lateral ray-bin range + elev corridor + center-depth break | 按 ray-CSR approximation 描述，或弱化“exact”集合定义 |
| fixed $\tau_A$ and accumulated $A_g$ | high-$\tau$ quantile protection | 按 3.9 二选一统一 |
| strict per-event TopK budget | threshold + possible ties | 写“approximately capped by fraction B”或实现 strict top-k |
| standard split | parent 保留，追加两个 children；raw-scale 操作不等于物理减半 | 准确描述 append-two refinement，或修代码后再称 split |
| $\rho_{igp}$ 乘 plane-intersection factor $\chi_{ig}$ | global integral直接在完整 3D anisotropic Gaussian 上算，不显式计算 Methods 的 $\chi_{ig}$/2D conditional | 统一 renderer derivation |
| high-residual diagnostic 可能暗示参与 score | `max_residual_hits` 被累计但不进入 `_responsibility_scores` | 不把它列为有效筛选项；可删除死配置或在 Supplementary 说明 diagnostic-only |

### 3.12 代码重要但 Methods/Implementation Details 未写的参数

优先级最高：local `k_sigma=2.5`、真实 $\gamma=0.25$、$\sigma_{ele}$ 四个系数、dataset-specific corridor、`ray_weight_min=1e-4`、`high_tau` protection mode/quantile、$\tau_\omega=10^{-6}$、`prune_opacity=1e-4`、初始化 scale/alpha/tau、sampler 30% uniform + 70% intensity、0.05 intensity threshold、0.2-mm center jitter、holdout 20%/trim 10、frame shuffled-cycle sampling。

---

## 4. Implementation Details 参数分级

### 4.1 必须在 Implementation Details 写明

1. 训练：1M init、30k iterations、Adam、完整 frame/iteration、shuffled-cycle sampling、$0.8L_1+0.2(1-SSIM)$、五组 LR、center 的 10× exponential decay、scale bounds、$\alpha_g/\tau_g$ ranges。
2. pseudo-pose：$d_0$ 来自 dataset metadata/calibration，$\rho=1$，axis=`+z`，reference retained frame 0，uniform spacing，无 pose optimization。
3. local：additive composition；$\sigma_{ele}=\mathrm{clamp}(1+0.01z,0.8,5)$ mm；$\gamma=0.25$；local AABB multiplier 2.5；至少说明 small-weight threshold $10^{-6}$。
4. global：independent learnable $\tau_g$；integral formula；$\kappa_{glo}=2$；dataset-specific candidate corridor half-width；$N_q=3$ Gauss–Legendre；2-mm **radius** clipped window；$\lambda_{ray}=0.55$；$T_{min}=0.35$；$\beta=0.8$。
5. adaptation：$q_d=.98$、$\tau_\omega=10^{-6}$、$B=.05N$、max-scale split threshold .6 mm、$q_p=.10$、$\tau_\alpha=10^{-4}$、真实 protection mode=`high_tau` + quantile .80、schedule 1k–25k every 2k。

### 4.2 可以省略或放 Supplementary

- tile size 8、ray CSR binning、`ray_weight_min=1e-4`、axial margin 0、slab half-thickness 0；
- determinant epsilon/covariance jitter $10^{-6}$；
- lateral focus 参数（因为 switch=false）、2D inflate=0；
- Adam eps $10^{-15}$、SSIM window 11/sigma 1.5；
- initial quaternion、importance sampler实现细节、safe multinomial；
- inactive union-gradient、structure weighting、max-scale prune、prune-intensity legacy fields；
- renderer environment capacity variables，除非最终运行环境设置过它们；现有生产脚本未找到设置证据。

### 4.3 不应作为可调超参数写

- $d_0$、$z_{near}$、lateral/axial pixel spacing、image-plane axes：数据 calibration/meta；
- $\mathbf u$、$i_0$：canonical coordinate/gauge 定义（需要写，但不应包装成 tuned hyperparameter）；
- Gaussian center/scale/quaternion、$\alpha_g$、$\tau_g$：可学习变量；应写 parameterization/range/init，而不是固定超参数；
- TUS/TDSC corridor 若其数值被定义为 acquisition spacing ratio，应称 data-derived calibration setting；但当前代码是直接 override 的固定数值，必须交代来源。

### 4.4 用户点名参数逐项判断

| 参数 | 主文 Implementation Details？ | 原因/真实值 |
|---|---|---|
| $\rho$ | 是 | sensorless geometry 核心；1.0 |
| $\gamma$ | 是 | 直接改变 local amplitude；真实 0.25，不是 2 |
| $\kappa_{ele}$ | 否，当前不应按此符号写 | hard cutoff 未实现 |
| $\kappa_{img}$ | 改名后写 | 写 local AABB multiplier 2.5，不要声称 exact Mahalanobis cutoff |
| $\kappa_{glo}$ | 是 | 2.0 |
| $N_q$ | 是 | 3-point Gauss–Legendre |
| integration window | 是 | radius 2 mm、最多 4-mm span |
| $\lambda_{ray}$ | 是 | 0.55 |
| $T_{min}$ | 是 | 0.35 |
| $\beta$ | 是 | 0.8 |
| $q_d$ | 是 | 0.98 |
| $\tau_\omega$ | 是或补充材料 | $10^{-6}$；用于 reproducing exact selection |
| $B$ | 是 | 5% current population |
| split threshold | 是 | max activated scale >0.6 mm |
| $q_p$ | 是 | 0.10 |
| $\tau_\alpha$ | 是 | $10^{-4}$ |
| $\tau_A$ | 不能写固定值 | 未实现；应换成 real `high_tau` protection quantile 0.80 |

### 4.5 建议写入主文 Implementation Details 的最小参数集合

若版面极紧，至少保留：

```text
1M Gaussians; 30k Adam iterations; one shuffled full frame/iteration;
loss 0.8 L1 + 0.2 (1-SSIM);
LR(mu,scale,quat,alpha,tau)=(1.6e-4,5e-3,1e-3,8e-2,1e-2),
mu LR exponentially to 1.6e-5; scales 0.05–3 mm;
alpha in (0,0.2), tau in (0,1).

Pseudo pose: d0 from dataset metadata, rho=1, +z axis, i0=0, uniform spacing.

Local: sigma_ele=clamp(1+0.01z,0.8,5) mm, gamma=0.25,
local AABB multiplier=2.5, additive composition.

Global: kappa_glo=2, Nq=3 Gauss–Legendre,
dataset-specific elevational candidate corridor,
closest-point integration radius=2 mm,
lambda_ray=0.55, Tmin=0.35, beta=0.8.

Adaptation: qd=0.98, tau_omega=1e-6, B=5%,
split by max scale>0.6 mm, qp=0.10, tau_alpha=1e-4,
high-tau protection quantile=0.80,
events every 2k from 1k through 25k.
```

---

## 5. 现有 Ablation / Parameter Sensitivity 审计

所有当前纸面主 ablation 使用：

- Synthetic：`synthetic_liver/l1`；
- Real：`tus_rec_2024_qc/010_LH_Per_L_DtP_qc_0005_0370`；
- 30k final checkpoint，1M init，20% interval holdout，trim 10；
- 指标：PSNR/SSIM/LPIPS；底层 summary 同时存 L1/DISTS/GMSD；
- 主结果 root：`outputs/production/ablations/ours/expanded_paper_plan_20260702/`；
- 历史/default source rows：`outputs/production/ablations/ours/summary.tsv`。

### 5.1 Joint component ablation（T01）

| 行 | LG | DC | PP/EP | 实际配置变化 | Syn P/S/L | Real P/S/L |
|---|---:|---:|---:|---|---|---|
| a | off | gradient | file | local-only + gradient DC + file pose | 29.849/.682/.270 | 32.516/.864/.164 |
| b | off | gradient | canonical | a + uniform canonical pose | 29.836/.681/.279 | 32.725/.866/.166 |
| c | off | responsibility | file | local-only + contribution DC + file | 29.854/.682/.268 | 32.525/.864/.164 |
| d | off | responsibility | canonical | c + uniform canonical | 29.842/.681/.279 | 32.730/.866/.166 |
| e | on | gradient | file | integral q3 renderer + gradient DC + file | 29.749/.678/.234 | 32.768/.866/.141 |
| f | on | gradient | canonical | e + uniform canonical | 29.753/.677/.244 | **33.004**/.868/.142 |
| g | on | responsibility | file | full renderer + contribution DC + file | 29.746/.677/.233 | 32.764/.866/.141 |
| h | on | responsibility | canonical | full stack | 29.755/.677/.244 | 32.998/.868/.142 |

结果文件：

- `outputs/production/ablations/ours/expanded_paper_plan_20260702/tables/t01_three_module_joint.md`
- `.../summary.tsv`
- 论文对应：`04_experiments_and_results.tex` 的 `tab_ablation_joint_components` 和 joint visual figure。

### 5.2 Local-global renderer ablation（T02）

固定 contribution DC + file pose，只改 renderer：

| 行 | 实际 renderer | 关键 override | Syn P/S/L | Real P/S/L |
|---|---|---|---|---|
| a | local-only | ray disabled | 29.854/.682/.268 | 32.525/.864/.164 |
| b | simple global | `slice_depth`, q=1 | 29.782/.678/.232 | 32.660/.866/.141 |
| c | ray integral | integral, q=1 | 29.741/.678/.235 | 32.759/.866/.142 |
| d | ray integral | integral, q=3 | 29.746/.677/.233 | **32.764**/.866/.141 |

其余默认固定为 $\kappa_{glo}=2,\beta=.8,\lambda=.55,Tmin=.35$。结果：
`outputs/production/ablations/ours/expanded_paper_plan_20260702/tables/t02_renderer_internal.md`。
论文对应 `tab_ablation_local_global_renderer`。

### 5.3 Gaussian adaptation ablation（T03）

固定 full integral q3 renderer + file pose：

| 行 | 实际配置 | Syn P/S/L | Real P/S/L |
|---|---|---|---|
| a | gradient DC | 29.749/.678/.234 | **32.768**/.866/.141 |
| b | responsibility densify，prune=false | 29.742/.678/.234 | 32.730/.866/.142 |
| c | responsibility densify+prune，protection=false | **29.752**/.678/.232 | 32.761/.866/.141 |
| d | full responsibility，`high_tau` q=.80 protection | 29.746/.677/.233 | 32.764/.866/.141 |

结果：`outputs/production/ablations/ours/expanded_paper_plan_20260702/tables/t03_dc_internal.md`；论文对应 `tab_ablation_gaussian_adaptation`。

重要解释限制：这一表证明的是当前 `high_tau` protection on/off，不是 Methods 定义的 $A_g/\tau_A$ global-path protection。

### 5.4 Pseudo-pose ablation（T04）

固定 full renderer + responsibility DC：

| 行 | pose input | 是否依赖 file poses | Syn P/S/L | Real P/S/L |
|---|---|---:|---|---|
| a | full file pose | 是，完整 6DoF | 29.748/.678/.234 | 32.750/.866/.142 |
| b | identity orientation + cumulative file step norm | 是 | 29.745/.677/.245 | 32.817/.866/.145 |
| c | identity orientation + cumulative file translation projected to +z | 是 | **29.774**/.678/.246 | 32.565/.863/.155 |
| d | identity orientation + uniform $d_0\rho$ along +z | 只读取 pose file 以确认 frame count，不用于 geometry | 29.755/.677/.244 | **32.998**/.868/.142 |

结果：`outputs/production/ablations/ours/expanded_paper_plan_20260702/tables/t04_easy_pose_internal.md`；论文对应 `tab_ablation_pseudo_pose`。

广域验证还包括：

- T06 file vs canonical：6 Liver + 5 TUS + 5 TDSC sweeps，结果 `outputs/production/ablations/ours/tables/t06_easy_pose_pairs.md`；
- T08 Spine negative control：`.../tables/t08_negative_control.md`，证明 canonical uniform 不适合该非规则 topology；
- multi-sweep mean：`outputs/production/ablations/ours/multisweep_mean_validation_20260702/tables/`。其中 5-sweep TUS mean：Base 33.450、+LG 33.889、+LG+DC 33.886、+LG+DC+EP 34.235；5-sweep Synthetic 中 local-only PSNR 更高但 LG 改善 LPIPS。

### 5.5 Parameter sensitivity 实际范围

| 组 | 参数 | sweep values | default | 实验形式 | 数据 |
|---|---|---|---|---|---|
| pseudo-pose | $\rho$ | .60,.75,.90,1,1.10,1.25,1.50 | 1 | 一维 curve | 主图取 Liver/l1 + TUS representative；T07 还含 Liver/r2、另一 TUS、2 TDSC |
| renderer | $\lambda_{ray}$ | 0,.20,.35,.55,.75,1 | .55 | one-at-a-time | Liver/l1 + TUS rep |
| renderer | $\beta$ | 0,.40,.60,.80,1 | .80 | one-at-a-time | 同上 |
| renderer | $T_{min}$ | .20,.35,.50,.65,.80 | .35 | one-at-a-time | 同上 |
| renderer | $N_q$ | 1,2,3 | 3 | one-at-a-time | 同上 |
| renderer | $\kappa_{glo}$ | -1,1.5,2,3 | 2 | one-at-a-time；-1=回退 local 2.5 | 同上 |
| adaptation heatmap A | $(q_d,B)$ | q=.94,.96,.98,.99 × B=.02,.05,.08,.12 | (.98,.05) | 4×4 full grid | 同上 |
| adaptation heatmap B | $(q_p,q_{protect})$ | q_p=.02,.05,.10,.15,.20 × protect=.60,.70,.80,.90 | (.10,.80) | 5×4 full grid | 同上 |
| adaptation dot | $\tau_\alpha$ | 1e-5,5e-5,1e-4,5e-4,1e-3 | 1e-4 | one-at-a-time | 同上 |

没有找到当前 production paper sensitivity 对 $\gamma$、integration radius、$\tau_\omega$、split 0.6 mm threshold 的正式 sweep。

结果/figure 路径：

- raw metrics：`outputs/production/ablations/ours/expanded_paper_plan_20260702/summary.tsv`；
- renderer：`.../tables/t14_renderer_sensitivity_delta_psnr.md`、`t14_renderer_sensitivity_summary.md`；
- adaptation：`.../tables/t15_dc_sensitivity_delta_psnr.md`、`t15_dc_sensitivity_summary.md`；
- rho：`outputs/production/ablations/ours/summary.tsv`、`.../tables/t07_spacing_tolerance.md`；
- manifests：`docs/production/ablation_final_outputs/manifests/`；
- 当前论文组合图：`paper_ablation_parameter_sensitivity_single_column.pdf/png`。

当前论文组合图中的 adaptation panel 只有 $(q_d,B)$ heatmap。独立 `paper_dc_parameter_sensitivity` 才还包含 $(q_p,q_{protect})$ heatmap 和 prune-opacity dot panel。因此 Results 当前写“adaptation heatmaps”是复数，但 cited combined figure 实际只显示一个 adaptation heatmap。

---

## 6. Parameter Sensitivity 真实结果

以下 P/S/L 分别为 PSNR/SSIM/LPIPS；均是 final 30k 指标。

### 6.1 Pseudo-pose $\rho$

| $\rho$ | Synthetic Liver P/S/L | TUS-REC P/S/L |
|---:|---|---|
| .60 | 30.0221 / .6833 / .2734 | 32.7710 / .8652 / .1528 |
| .75 | 29.9359 / .6814 / .2596 | 32.8837 / .8666 / .1473 |
| .90 | 29.8260 / .6789 / .2499 | 32.9496 / .8677 / .1434 |
| **1.00** | 29.7471 / .6772 / .2443 | 33.0014 / .8682 / .1414 |
| 1.10 | 29.6771 / .6753 / .2407 | 33.0367 / .8687 / .1406 |
| 1.25 | 29.5542 / .6722 / .2335 | **33.0684** / .8690 / .1399 |
| 1.50 | 29.3038 / .6668 / **.2297** | 33.0579 / .8689 / .1410 |

结论：synthetic PSNR 随 $\rho$ 增大单调下降，但 LPIPS 反而改善；real PSNR 在 1.25 最佳、仅比 default 高 0.067 dB。正文“opposite sides of rho=1”准确，但必须说明选择 default 是跨域折中，不是两个数据都在 1 附近达到各自 optimum。

### 6.2 Renderer：$\lambda_{ray}$ 与 $\beta$

这些是**分别 OAT sweep**，不是 $(\beta,\lambda)$ 二维 heatmap。

| 参数值 | Synthetic Liver P/S/L | TUS-REC P/S/L |
|---|---|---|
| $\lambda=0$ | 29.8559/.6825/.2692 | 32.5292/.8639/.1634 |
| $\lambda=.20$ | 29.7517/.6776/.2338 | 32.6343/.8656/.1412 |
| $\lambda=.35$ | 29.7422/.6773/.2346 | 32.7103/.8658/.1411 |
| **$\lambda=.55$** | 29.7457/.6775/.2334 | 32.7641/.8661/.1415 |
| $\lambda=.75$ | 29.7522/.6777/.2331 | **32.7651**/.8660/.1418 |
| $\lambda=1$ | 29.7635/.6778/.2335 | 32.7621/.8660/.1415 |
| $\beta=0$ | 29.8476/.6823/.2674 | 32.5209/.8638/.1638 |
| $\beta=.40$ | 29.8255/.6808/.2589 | 32.6630/.8653/.1553 |
| $\beta=.60$ | 29.7849/.6795/.2483 | 32.7179/.8656/.1493 |
| **$\beta=.80$** | 29.7457/.6775/.2334 | **32.7641**/.8661/.1415 |
| $\beta=1$ | 29.7305/.6753/**.2119** | 32.7345/.8657/**.1313** |

Synthetic PSNR 偏好关闭/减弱 global modulation，但 perceptual metric 偏好更强 global；real PSNR 支持默认附近。不能仅用 PSNR 宣称 default 对所有域最优。

其他 renderer sweep 的 PSNR：

| 参数 | values → Synthetic PSNR | values → Real PSNR |
|---|---|---|
| $T_{min}$ | .20→29.714, .35→29.746, .50→29.780, .65→29.820, .80→29.830 | .20→32.705, .35→32.764, .50→32.724, .65→32.665, .80→32.606 |
| $N_q$ | 1→29.737, 2→29.220, 3→29.746 | 1→32.753, 2→32.157, 3→32.764 |
| $\kappa_{glo}$ | -1→29.738, 1.5→29.749, 2→29.746, 3→29.749 | -1→32.778, 1.5→32.684, 2→32.764, 3→32.778 |

### 6.3 Adaptation $(q_d,B)$ heatmap

| $q_d$ | $B$ | Synthetic P/S/L | Real P/S/L |
|---:|---:|---|---|
| .94 | .02 | 29.7495/.6778/.2338 | 32.7452/.8659/.1416 |
| .94 | .05 | 29.7492/.6777/.2346 | 32.7583/.8660/.1413 |
| .94 | .08 | 29.7469/.6776/.2328 | 32.7590/.8659/.1413 |
| .94 | .12 | 29.7515/.6777/.2337 | 32.7501/.8658/.1412 |
| .96 | .02 | 29.7474/.6776/.2345 | 32.7501/.8659/.1415 |
| .96 | .05 | 29.7478/.6778/.2336 | 32.7558/.8660/.1410 |
| .96 | .08 | 29.7483/.6776/.2333 | 32.7546/.8659/.1413 |
| .96 | .12 | **29.7536**/.6778/.2326 | 32.7592/.8661/.1414 |
| .98 | .02 | 29.7495/.6777/.2330 | 32.7518/.8659/.1412 |
| **.98** | **.05** | 29.7457/.6775/.2334 | **32.7641**/.8661/.1415 |
| .98 | .08 | 29.7434/.6776/.2346 | 32.7490/.8660/.1413 |
| .98 | .12 | 29.7421/.6773/.2331 | 32.7437/.8659/.1418 |
| .99 | .02 | 29.7412/.6776/.2331 | 32.7552/.8660/.1412 |
| .99 | .05 | 29.7394/.6777/.2348 | 32.7561/.8660/.1414 |
| .99 | .08 | 29.7408/.6774/.2324 | 32.7456/.8658/.1413 |
| .99 | .12 | 29.7453/.6778/.2332 | 32.7583/.8660/.1414 |

PSNR range 只有 Synthetic 0.014 dB、Real 0.020 dB，确实形成 plateau。默认在 real 最优，但不是 synthetic 最优；差异小到不应做强因果解释。

### 6.4 其他 adaptation sensitivity

- $(q_p,q_{protect})$ 全 5×4 grid 已做。PSNR range：Synthetic 0.014 dB，Real 0.023 dB；默认 (.10,.80) 是 real 最优，synthetic 最优是 (.02,.90)。完整每点数据：`t15_dc_sensitivity_delta_psnr.md`。
- $\tau_\alpha$（代码 `prune_opacity`）已扫：$10^{-5},5\cdot10^{-5},10^{-4},5\cdot10^{-4},10^{-3}$。Synthetic PSNR 分别 29.755/29.743/29.746/29.742/29.744；Real 32.743/32.748/32.764/32.760/32.749。
- $\gamma$、integration radius、$\tau_\omega$、split threshold 未找到当前 production paper sweep。

---

## 7. 哪些参数值得在主文做消融

### 7.1 对用户提出方案的判断

| 建议方案 | 是否与现有实验一致 | 数据是否足够 | 准确写法 |
|---|---|---|---|
| Pseudo-pose $\rho$ | 是 | 足够；一维 7 点且还有多 sweep T07 | 可直接作为 spacing sensitivity curve |
| Renderer $(\beta,\lambda_{ray})$ | 部分一致 | 两个参数分别有完整 OAT sweep，但**没有二维 grid/heatmap** | 写成 two one-at-a-time sensitivity profiles，不要写 joint 2D sensitivity |
| Adaptation $(q_d,B)$ | 完全一致 | 足够；4×4 heatmap，synthetic+real | 可直接作为 joint sensitivity heatmap |

### 7.2 三类参数

#### 主文必须做 sensitivity

- $\rho$：sensorless geometry 唯一真正自由的全局 spacing scale，且跨域 optimum 不同；现有 curve 很有解释力。
- renderer 的 $\beta$、$\lambda_{ray}$：直接控制 global modulation，real 端影响约 0.24 dB；现有 OAT 数据足够。若版面只留一行，可把两者放同一 renderer sensitivity panel，但不能称二维 heatmap。
- $(q_d,B)$：对应 selection aggressiveness 与 event budget，现有 4×4 plateaus 支持“稳定而不脆弱”的论点。

#### 主文只需要 component/design ablation，不需要额外 parameter sweep

- local/global/simple/integral/$N_q$ 设计：T02 已经回答，且 $N_q$ 另有 sensitivity；
- contribution densify、prune、protection：T03 已回答；
- file/step/axis/uniform pseudo-pose：T04 已回答；
- split vs clone threshold：当前更需要先把真实 split 语义与代码统一，而不是展示 threshold sweep；
- $T_{min}$、$\kappa_{glo}$、$q_p$、protection quantile、$\tau_\alpha$：已有 supplemental sensitivity，可不占主文。

#### 只需固定值，不值得做主文消融

- $d_0,z_{near},\mathbf u,i_0$（校准/坐标定义）；
- optimizer LR、scale bounds、init values、tile size、numeric eps、weight cutoffs；
- $\tau_\omega$；
- inactive lateral focus/inflation/scale-prune/union-gradient options。

### 7.3 主文现有 ablation 是否足够

在不新增实验的前提下，T01 joint + T02 renderer + T03 adaptation + T04 pseudo-pose + 组合 sensitivity 图已经足够完整。建议保持结构，不再扩张实验数量。必须做的是**修正文义和符号**，尤其是 gamma、k-sigma、window radius 和 protection logic；否则更多实验也不能解决方法—代码不一致。

---

## 8. Experiments and Results 逐段核对

### 8.1 Experimental Settings / Implementation Details

| 当前文字 | 核对 | 问题/建议 |
|---|---|---|
| 30k Adam、1M init、one full frame/iter | 准确 | 补充 shuffled-cycle sampling、Adam eps 可放附录 |
| scale [0.05,3], alpha [0,.2], tau [0,1] | 基本准确 | sigmoid 为开区间；可按论文习惯写 bounded range |
| LR 1.6e-4,5e-3,1e-3,8e-2,1e-2 | 准确 | center final 1.6e-5 和公式应补充 |
| $\rho=1,i_0$ first retained,$\mathbf u$ predefined,$d_0$ metadata | 准确 | `\Cref{sec:METHOD-PSEUDOPOSE}` label 不存在；Methods 真正 label 是 `sec:calibration-guided-geometry` |
| local `k_sigma=2` | **不准确/含混** | global candidate=2；local k=2.5；$\kappa_{ele/img}$ 无一一对应 |
| $\gamma=2$ | **错误** | 代码 0.25 |
| 3 samples within 2-mm window | **一半错误** | 3 点正确；2 mm 是 radius，最大总宽 4 mm；且是 Gauss–Legendre |
| lambda=.55,Tmin=.35,beta=.8 | 准确 | 建议补 `ray_weight_min=1e-4` 到附录 |
| adaptation every 2k, 1k–25k | 准确 | 实际 event 为 iteration divisible by 2000 且在范围内，即事件发生在 2k,4k,...,24k；**不会在 1k 或 25k 发生**。文字 “from 1k to 25k” 表示 active interval，尚可但最好列事件规则 |
| qd=.98, B=5%, split .6 | 准确但不完整 | split 取 max scale；budget 不是 strict TopK；应补 tauomega |
| pruning uses tau_alpha, qp, tau_A | **缺真实数值且逻辑错误** | tau_alpha=1e-4，qp=.10；无 tauA；实际 high_tau quantile=.80 |
| same hyperparameters all sequences | **不完全准确** | corridor half-width 有 dataset-specific overrides：.395/.202/.238 |
| single RTX 4090-class GPU | 与脚本一致 | 当前 expanded job 用多个 GPU 并行跑不同单卡任务，但每个 run 单 GPU；措辞可保留 |

### 8.2 Main comparison 数值与 best baseline

源表：`docs/production/comparison_final_outputs/quantitative_tabs.md`；最终 LaTeX 表：`docs/production/experiment_results_io/tab_comparison_four_datasets.tex`。两者已逐行核对，所有显示数值一致。

核对结论：

- Ours 在四数据集 PSNR/L1/SSIM/GMSD 均第一：准确。
- Liver best baseline PSNR 28.0158（UltraGS）→29.3480：正文 28.02→29.35，准确。
- Spine best baseline PSNR 30.6262（UltraGray）→31.0188：30.63→31.02，准确。
- Liver SSIM .5971（NeRF-US）→.6552：正文 .597→.655，准确。
- Spine SSIM .7551（UltraGray）→.7898：.755→.790，准确。
- TUS best baseline PSNR 33.9551（Slice-adapted 3DGS）→34.2494：33.96→34.25，准确。
- TDSC best baseline PSNR 27.7615（UltraGauss）→28.4809，margin .7194 dB：正文 .72 dB，准确。
- TDSC best baseline SSIM .7926（Slice-adapted 3DGS）→.8013：.793→.801，准确。
- perceptual winners：UltraGray 在两个 simulated 的 LPIPS/DISTS 最低；Slice-adapted 3DGS 在 TUS 两者最低；UltraGauss 在 TDSC 两者最低（DISTS .1802 仅比 Ours .1803 低 .0001）：准确。
- UltraGauss train ~171 s、FPS 1269/1321：正文“about 171 s and exceeding 1269 FPS”准确。
- Ours train 355.3/338.4 s、FPS 240.1/352.8：正文 338–355 和 240–353，准确。
- 最终表 caption 声明按未四舍五入数值加粗，实际加粗正确。例如 TDSC DISTS 中 UltraGauss/Ours 都显示 .180，但原值 .1802/.1803，因此只加粗 UltraGauss；TUS GMSD 中 3DGS/Ours 都显示 .045，但原值 .0454/.0451，因此只加粗 Ours。
- 四张最终 ablation LaTeX 表与 T01–T04 Markdown 的三位小数逐行一致，行顺序和 checkmark 配置也一致，没有发现转录错误。

需要修改/降调的地方：

1. **主比较 protocol 不是真正 sensorless。**`gs_strict/ours.yaml` 设 `scene.loading.pose_mode=file`；所有 main comparison Ours rows 使用 file poses。应明确“main comparison uses a common measured/file-pose protocol for renderer/reconstruction fairness; sensorless replacement is evaluated separately”。否则读者会把主表理解为完整 sensorless 方法。
2. Comparison Methods 小节没有真正定义正文所说的“common-pose protocol”。需补 pose source、holdout、final checkpoint 和每数据集 sweep aggregation。
3. “because 3DGS already provides a strong measured-pose reconstruction”是合理解释但不是表格直接证据；应写成 observation/inference，并明确所指是 Slice-adapted 3DGS。
4. “reduces training time on most dataset groups”需限定比较对象：它比 UltraGS/Slice-3DGS/GaussianPile 快，但不比 UltraGauss/UltraGray 快；对 implicit methods 则 simulated/real 情况不完全相同。当前下一句已有部分限定，仍建议更精确。
5. **NeRF-US variant 命名不完整。**结果源文件明确写 `NeRF-US w/o prior`，最终 LaTeX 表只写 `NeRF-US`，而 Comparison Methods 又介绍 NeRF-US 的 diffusion-learned geometric prior。这会让读者误判被评估的是带 prior 的完整方法。项目记录 `docs/methods/baselines/nerfus/no_prior_synthetic_liver_report.md` 说明：released repository 没有可用的 ultrasound fine-tuned diffusion checkpoint，复训 prior 还需要额外数据准备和计算，因此生产 benchmark 有意关闭 prior，并采用 UltraNeRF-matched no-prior protocol；`docs/production/multi_comparisons.md` 还明确要求表行标为 `NeRF-US w/o prior`。最终论文应照此标注并给 footnote，不能只写 `NeRF-US`。

### 8.3 Qualitative description

已查看 `paper_all_datasets_2d_rendering.png`；它包含每个 dataset 一个 selected frame/crop，图中可观察到：NeRF-US Liver 明显平滑，UltraGauss Spine 明显碎片化，TUS 多方法 texture 差异，TDSC UltraGray 有明显横向 banding。正文总体与图像相容。

但下列措辞是主观解释，不能由单张 selected crop 单独证明：

- “more faithfully”“remains close to reference”“handles different texture statistics”；
- “across simulated, phantom, freehand, and ABUS data”若指普遍 qualitative behavior，证据只有每数据集一个展示帧；定量均值可支持总体 ranking，但不支持每个局部视觉机制。

建议保留为定性观察，避免把 boundary/speckle 的视觉差异写成已验证的物理因果。

### 8.4 Joint ablation 文字

- Base 32.516→+LG 32.768、LPIPS .164→.141：准确。
- CGA near-neutral：准确；T01 g 比 e real PSNR -0.004 dB。
- Full 32.998/.868：准确。
- “full model remains in best group”合理，但必须注意 real PSNR 数值最高是 row f 33.004，不是 full row h；差 0.006 dB。
- synthetic 上 LG 降 PSNR约 .10 dB而改善 LPIPS，当前段落主要强调 real，建议补充该 tradeoff。

### 8.5 Renderer ablation 文字

- local-only synthetic PSNR highest：准确。
- local-only real LPIPS .164，global 后 .141：准确。
- q3 real PSNR 32.764 best：准确，仅比 q1 高 .005 dB。
- “supporting selected finite-corridor attenuation model”略过度：T02 改了 local/simple/integral/q，而 corridor 本身没有在表中变化。更准确是支持 integral q3 相对其他 renderer formulations。

### 8.6 Adaptation ablation 文字

- 全部 PSNR/SSIM narrow bands：准确。
- adding prune recovers densify-only loss：real 32.730→32.761，准确。
- full protection within .004 dB of gradient：按三位小数 32.768 vs 32.764，准确。
- “global-path protection”术语不准确：表中真实是 high-$\tau$ quantile protection，不是 Methods $A_g$ path accumulation。
- “confirms stable budget reallocation without artifact”中“stable metrics”有证据；“without artifact”主要依赖 selected visual，建议降为 “no visible degradation in selected crops”。

### 8.7 Pseudo-pose ablation 文字

- step/axis 依赖 file pose：准确。
- axis variant real 较差：准确（32.565）。
- uniform removes geometry dependency，real PSNR/SSIM best：准确。
- synthetic close to file：准确（+0.007 dB PSNR，LPIPS +.010）。
- 需要明确 canonical uniform 仍加载 `poses.npy` 以取得/校验 frame count；代码日志明确它不用于 SliceMeta geometry。若声称完全不读取 measured pose file，应与数据 loader 实现区分。

### 8.8 Parameter sensitivity 文字

- synthetic/real spacing favor opposite sides：准确。
- renderer response range 大于 $(q_d,B)$ adaptation：准确，尤其 $N_q$。
- “adaptation heatmaps”对当前 cited combined figure 不准确：只有 $(q_d,B)$ 一个 heatmap；完整独立 T15 才有两个 heatmaps。
- “selected quantiles and budget”准确对应 qd/B；若只展示组合图，不应暗示主图同时展示 q_p/protection。
- 需要说明图只画 delta PSNR；SSIM/LPIPS 结果存在 summary，但没有进当前 sensitivity figure。

### 8.9 LaTeX/引用完整性问题

1. `04_experiments_and_results.tex` 引用 `\Cref{sec:METHOD-PSEUDOPOSE}`，仓库无此 label；正确 Methods label 是 `sec:calibration-guided-geometry`。
2. Methods global candidate equation 中 `\mathcal E^{glo}_{ig}` 重复了一次：`\mathcal E ... \mathcal E ... \cap`，是排版错误。
3. 五个补入的 LaTeX 表文件与源结果一致；不过 `04_experiments_and_results.tex` 的实际输入路径是 `tables/main/...`，而本审计收到的副本位于 `docs/production/experiment_results_io/`。合入论文工程时应确认 `tables/main/` 中使用的正是这五个版本。
4. 主表和 Figure 中方法名为 `ULGSplat`，正文使用 `\methodname{}`；需确认 macro 最终也展开为相同名称。
5. 主比较表将源结果的 `NeRF-US w/o prior` 改写为 `NeRF-US`；这违反项目已锁定的 required label，是实质性的 variant 标注缺失，不只是排版缩写。
6. 四张 ablation 表对若干显示值并列加粗，但 caption 没有说明是按三位小数并列、未四舍五入值，还是预定义 best group；建议统一声明，避免把 rounding tie 误解为精确相等。

---

## 9. 最终明确结论

### A. Implementation Details 最终建议参数清单

按必须写的优先顺序：

1. **训练 protocol**：1M init；30k；Adam；1 full shuffled frame/iteration；loss=.8 L1+.2 SSIM；final-30k evaluation；20% interval holdout/trim 10。
2. **可学习 Gaussian 参数**：scale 0.05–3 mm；$\alpha\in(0,.2)$、init .08–.12；$\tau\in(0,1)$、init .016–.024。
3. **LR**：center $1.6e-4\to1.6e-5$ exponential over 30k；scale 5e-3；quat 1e-3；alpha 8e-2；tau 1e-2。
4. **pseudo-pose**：$d_0$ dataset-meta；$\rho=1$；axis +z；$i_0=0$；uniform；pose optimization off。
5. **local**：$\sigma_{ele}=clamp(1+0.01z,.8,5)$ mm；gamma=.25；local AABB k=2.5；additive；weight cutoff 1e-6。
6. **global**：independent tau; integral；global candidate k=2；candidate corridor .395 mm (Liver/Spine), .202 (TUS), .238 (TDSC)；3-point Gauss–Legendre；closest-point radius 2 mm；lambda=.55；Tmin=.35；beta=.8；ray cutoff 1e-4。
7. **adaptation**：qd=.98；tauomega=1e-6；B=.05N；split on max scale>.6 mm；qp=.10；taualpha=1e-4；真实 protection=`high_tau`, q=.80；events at 2k,4k,...,24k（active interval 1k–25k）。

### B. 主文 Ablation 最终建议结构

1. T01 Joint：保留，说明 LG 是 real reconstruction 主要增益、DC metric-neutral、PP/EP 替代 file pose。
2. T02 Renderer：保留 local/simple/integral-q1/integral-q3；不要把它称为 corridor ablation。
3. T03 Adaptation：保留，但将“global-path protection”改成代码真实的 high-tau protection，或先统一实现后再用旧术语。
4. T04 Pseudo-pose：保留，明确 step/axis 是 oracle/file-dependent diagnostics，uniform 才是 sensorless geometry。
5. Sensitivity：保留 $\rho$ curve、renderer 的 beta/lambda OAT profiles、adaptation qd×B heatmap。现有数据足够，不建议重设计大批实验。
6. $Tmin,N_q,kappa_glo$ 和 qp/protection/taualpha 可放 supplement；当前已有数据，无需新跑。

### C. 当前缺失信息

1. 没有找到 $\tau_A$ 的代码变量或固定值；Methods $A_g$ accumulation 也未实现。
2. 没有当前 production sweep 支持 $\gamma=2$；相反代码明确是 .25。
3. 没有 $\kappa_{ele}$ 和 $\kappa_{img}$ 的 exact hard-cutoff implementation/default。
4. 没有 integration radius、$\tau_\omega$、split threshold、gamma 的正式 paper sensitivity。
5. dataset-specific corridor 数值的最终科学来源没有在当前两份 LaTeX 中解释；代码只显示 fixed override，TUS .202 与 nominal spacing .404/2 对应，TDSC .238 与 .475674/2 对应，Liver/Spine .395 并非都等于各自 spacing/2。
6. 无法从 selected qualitative figure 单独验证所有机制性语言，只能确认视觉描述大体相容。
7. 当前主比较使用 file pose；若论文希望主表代表完整 sensorless pipeline，现有 main-comparison table 不能直接支持这一 claim，需改 protocol 声明或另行提供 sensorless comparison（本审计不建议凭空把现有表改写成 sensorless）。

### D. 需要修改论文的符号/术语不一致

| 当前术语/符号 | 问题 | 建议 |
|---|---|---|
| `k_sigma=2` | 混淆 local 2.5 与 global 2.0 | 写 `local AABB multiplier=2.5; κ_glo=2.0` |
| $\gamma=2$ | 与代码 .25 冲突 | 改 .25 |
| $\kappa_{ele}$ | hard cutoff 未实现 | 删除/改为 soft slice response + weight cutoff |
| $\kappa_{img}$ | exact Mahalanobis cutoff 未实现 | 改为 k-scaled AABB candidate support |
| “2-mm integration window” | 实际 ±2-mm radius | 改为 “2-mm half-window/radius (up to 4 mm span)” |
| “three quadrature samples” | 信息不够 | 写 three-point Gauss–Legendre |
| $A_g,\tau_A$ / global-path protection | 当前生产用 high-tau quantile | 按真实代码重写或统一代码后重跑 |
| per-event `TopK_B` | ties 下不是 strict budget | 写 approximately capped fraction，或修 strict top-k |
| “split” | parent 未删除，bounded raw-scale 操作非精确 halving | 精确描述 append-two children；不要声称标准 replacement split |
| arbitrary $\mathbf u$ | 代码只支持 signed axes | 写 predefined canonical axis |
| full pre-pixel integral | 实际 closest-point localized integral | 改积分上下限 |
| `sec:METHOD-PSEUDOPOSE` | label 不存在 | 改 `sec:calibration-guided-geometry` |
| “common-pose protocol” | 小节未定义，且主表 Ours=file pose | 明确定义 file/measured-pose fairness protocol |
| `NeRF-US`（主表） | 原始结果实际是 `NeRF-US w/o prior`，正文却介绍其 diffusion prior；项目 protocol 要求使用完整 variant label | 表行改为 `NeRF-US w/o prior`；footnote 说明 prior 因无可用 ultrasound checkpoint 而关闭，并采用 UltraNeRF-matched protocol |
| “adaptation heatmaps” | 当前 cited combined figure只有一个 | 用单数，或引用独立 T15 multi-panel figure |
| PP abbreviation | 同一段 PP 同时可能被理解为 proposed pseudo-pose 和 `PP-S/PP-A/PP-U` prefix | 建议模块用 EP/PseudoPose，pose variants 用明确全称 |

---

## 10. 可核查文件索引

### 论文

- `docs/production/experiment_results_io/03_method.tex`
- `docs/production/experiment_results_io/04_experiments_and_results.tex`
- `docs/production/experiment_results_io/tab_comparison_four_datasets.tex`
- `docs/production/experiment_results_io/tab_ablation_joint_components.tex`
- `docs/production/experiment_results_io/tab_ablation_local_global_renderer.tex`
- `docs/production/experiment_results_io/tab_ablation_gaussian_adaptation.tex`
- `docs/production/experiment_results_io/tab_ablation_pseudo_pose.tex`

### 最终生产配置

- `configs_hydra/model/ours_us_focus.yaml`
- `configs_hydra/experiment/production/comparisons/us_recon_four_dataset_p100k_v1/gs_strict/ours.yaml`
- `configs_hydra/experiment/production/ablations/ours/base_1m30k.yaml`
- `configs_hydra/experiment/production/sensorless/ours_canonical.yaml`
- `configs_hydra/experiment/production/comparisons/us_recon_four_dataset_p100k_v1/nerfus_no_prior.yaml`
- `scripts/production/ablations/ours/submit_expanded_paper_plan_multiworker_1m30k.sh`

### 核心代码

- pseudo-pose：`scene/container/single_sweep_container.py`
- dataset calibration：`scene/meta/dataset_meta/*.py`
- local CUDA/reference：`ops/ours_renderer/include/shader/us_focus_device.cuh`、`ops/ours_renderer/ours_renderer/us_focus_reference.py`
- global integral：`ops/ours_renderer/src/shader/ray_attenuation.cu`
- renderer dispatch：`ops/ours_renderer/src/shader/wrapper_shader.cu`、`ops/ours_renderer/ours_renderer/autograd.py`
- adaptation：`methods/ours/density_control.py`、`methods/ours/dc_diagnostics.py`
- Gaussian parameterization/LR：`methods/ours/gaussian_container.py`、`methods/ours/gaussian_solver.py`
- training sampling：`methods/ours/gs_only_callbacks.py`、`scene/scene.py`、`scene/scheduler/offline_scheduler.py`
- initialization：`scene/sampler/unified_sampler.py`

### 结果

- main comparison：`docs/production/comparison_final_outputs/quantitative_tabs.md`
- comparison protocol/variant policy：`docs/production/multi_comparisons.md`
- NeRF-US no-prior protocol：`docs/methods/baselines/nerfus/no_prior_synthetic_liver_report.md`
- raw comparison：`outputs/production/comparisons/us_recon_four_dataset_p100k_v1/tables/`
- ablation lifecycle：`docs/production/ours_ablations.md`
- T01–T04：`outputs/production/ablations/ours/expanded_paper_plan_20260702/tables/`
- T14/T15 raw summary：`outputs/production/ablations/ours/expanded_paper_plan_20260702/summary.tsv`
- T07 rho：`outputs/production/ablations/ours/summary.tsv`、`outputs/production/ablations/ours/tables/t07_spacing_tolerance.md`
- sensitivity manifests：`docs/production/ablation_final_outputs/manifests/`
- multi-sweep validation：`outputs/production/ablations/ours/multisweep_mean_validation_20260702/tables/`
