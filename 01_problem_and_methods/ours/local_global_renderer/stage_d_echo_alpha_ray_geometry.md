# Ours Echo-Alpha Ray Geometry Stage D

Date: 2026-06-16
Branch: `dev/ours-echo-alpha-ray-geometry-stage-d-20260616`

## Motivation

Stage C 的 ray-CSR attenuation 解决了旧 tile-prefix 路线慢和候选不合理的问题，但实现仍有两个结构性偏差：

1. Local branch 同时使用 `opacities * intensities`，global branch 使用 `attenuations`，导致暗区可以被多种属性解释。
2. Ray attenuation backward 只回传 `d_attenuation`，没有让传播分支影响 Gaussian geometry。

UltraGray 的属性安排更干净：一个 attenuation 属性负责传播，一个 echo/SH 属性负责回波强度，geometry 同时受 transmission 和 echo 的梯度约束。

Stage D 先做最小可验证修正：

- Local intensity 从 `opacity * intensity` 改为单一 `echo_alpha`。
- Renderer 兼容旧 CUDA 签名，实际传入 `opacities=echo_alpha`、`intensities=1`。
- Ray-CSR attenuation backward 新增对 `means/scales/quats` 的几何梯度。

## Implementation

新增/改动属性：

```text
means, scales, quats, echo_alphas, attenuations
```

`opacities` 和 `intensities` 不再是训练参数，只作为兼容旧 renderer/density/loss 接口的只读 alias：

```text
opacities   -> echo_alphas
intensities -> ones_like(echo_alphas)
```

新增 alias：

```text
configs_hydra/experiment/preproduction/ideas/ours/local_global_renderer/tus_rec_2024_qc_echo_alpha_ray_csr_tail_2k.yaml
```

CUDA backward:

- `RayCsrAttenuationBackwardCUDA` 返回 `(grad_attenuations, grad_means, grad_scales, grad_quats)`。
- `grad_means` 来自 ray weight 对 Gaussian center 的 lateral/elevational 梯度。
- `grad_scales` 来自 ray weight 对 `sigma_lat/sigma_ele` 的梯度。
- `grad_quats` 先累计 `sigma_lat/sigma_ele -> R` 的梯度，再沿用项目现有 CUDA backward 的 quaternion finite-difference VJP 风格。
- Axial ordering/cutoff 的不可导候选边界仍不回传梯度。

## Verification

Build:

```text
bash -lc 'source ~/.bashrc && conda activate gaussian_splatting && module load cuda/11.8.0 && bash ops/ours_renderer/rebuild.sh'
```

Result:

```text
ours_renderer smoke check passed.
```

All jobs ran on Slurm node `gpu3` with `NVIDIA GeForce RTX 4090`.

Output root:

```text
outputs/preproduction/ideas/ours/local_global_renderer/ours_stage_d_echo_alpha_20260616
```

## Implementation Audit

User questioned whether the weak gain came from implementation errors rather than the idea itself. The audit found one real implementation gap:

- Stage D initial commit only passed global attenuation gradients to `attenuations` and `means`.
- `scales/quats` still only received local-response gradients, so the global propagation line could not adjust Gaussian footprint/orientation.

Diagnostics added:

```text
scripts/methods/ours/local_global_renderer/diagnose_ray_csr_gradients.py
scripts/methods/ours/local_global_renderer/diagnose_echo_alpha_ray_autograd.py
scripts/methods/ours/local_global_renderer/diagnose_stage_d_modulation.py
```

Ray sub-operator finite-difference check after the fix:

```text
modulation stats: min=0.898632 max=1.000000 mean=0.966103 std=0.039444
attenuation worst_rel_err <= 0.000020
means worst_rel_err      <= 0.015099
gradient check passed
```

Full renderer autograd finite-difference check after the fix:

```text
ray-enabled worst_main_rel_err       = 0.000396
ray-enabled worst_scale_quat_rel_err = 0.000522
local-only  worst_main_rel_err       = 0.005772
local-only  worst_scale_quat_rel_err = 0.001065
```

Before adding scale/quaternion VJP, the full renderer had a clear ray-enabled geometry mismatch:

```text
scales(1,0) rel_err: 0.116389 -> 0.000159
quats(1,2)  rel_err: 0.188060 -> 0.000522
```

Real checkpoint modulation diagnostic for the 2k ray run:

```text
echo_alpha:  min=0.001643 max=0.199999 mean=0.086251 std=0.063285
attenuation: min=0.001882 max=0.051398 mean=0.039153 std=0.013988
all_modulation: min=0.960000 max=1.000000 mean=0.977458 std=0.016206
```

This confirms the branch is active but weakly modulating the local image rather than dominating it.

## 200-iter Smoke

| Run | Branch | Test PSNR | Test L1 | Test SSIM | Test LPIPS | Speed |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| `stage_d_echo_local_200` | echo-alpha local only | 20.9973 | 0.0551 | 0.4687 | 0.4652 | 33.52 it/s |
| `stage_d_echo_ray_csr_200` | echo-alpha + ray-CSR attenuation | 21.0050 | 0.0550 | 0.4693 | 0.4645 | 34.35 it/s |

Smoke conclusion:

- New 5-property container trains and evaluates correctly.
- Ray branch with attenuation-to-mean gradient is stable.
- 200 iter shows a tiny positive signal, but only serves as smoke.

## 2k Check

| Run | Branch | Test PSNR | Test L1 | Test SSIM | Test LPIPS | Speed |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| `stage_d_echo_local_2k` | echo-alpha local only | 29.2022 | 0.0200 | 0.7998 | 0.2700 | 101.35 it/s |
| `stage_d_echo_ray_csr_2k` | echo-alpha + ray-CSR attenuation | 29.2501 | 0.0199 | 0.8005 | 0.2676 | 94.83 it/s |
| `stage_d_echo_ray_csr_vjp_2k` | echo-alpha + ray-CSR attenuation + full geometry VJP | 29.2572 | 0.0199 | 0.8008 | 0.2671 | 87.96 it/s |

Delta at 2k:

```text
PSNR  +0.0479
L1    -0.0001
SSIM  +0.0007
LPIPS -0.0024
Speed -6.4%
```

Delta of full geometry VJP vs initial Stage D ray run:

```text
PSNR  +0.0071
L1     0.0000
SSIM  +0.0003
LPIPS -0.0005
Speed -7.2%
```

## Interpretation

This is the first version where the local/global physical split is structurally plausible:

- Local response is no longer split into separate opacity and intensity factors.
- Global attenuation now affects geometry optimization through `means/scales/quats`.
- Ray-CSR speed remains usable, but full scale/quaternion VJP makes it slower than the initial mean-only version.

The improvement is still small, but it is directionally consistent across PSNR/L1/SSIM/LPIPS on the 2k TUS-REC QC run. The implementation audit supports that the current weak gain is no longer explained by obvious attribute wiring or missing first-order geometry gradients.

## Limitations

1. The local CUDA shader still uses legacy argument names `gauss_opacities` and `gauss_intensities`; Stage D maps them to `echo_alpha` and `1`.
2. Quaternion VJP follows the repo's existing finite-difference VJP style, so it is correct enough for diagnostics but not as efficient as a hand-derived analytic expression.
3. The attenuation model still uses a simplified lateral/elevational Gaussian weight, not UltraGray's full ray integral.
4. Candidate set changes are hard/discrete; large finite-difference eps such as `1e-2` can show false mismatches at tile/ray boundaries.

## Next Steps

1. Run 10k on the same sweep for `stage_d_echo_local` vs `stage_d_echo_ray_csr_vjp`.
2. Tune modulation strength after the VJP fix; current real-run modulation is only `0.96-1.00`.
3. Consider replacing the simplified ray weight with an UltraGray-style line integral while preserving Ours probe-to-slice coordinate transform.
4. If 10k remains weak, treat attenuation as a mild auxiliary physical prior rather than the core paper contribution.
