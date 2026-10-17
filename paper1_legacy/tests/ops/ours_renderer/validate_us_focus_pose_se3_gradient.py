"""
Phase 19: validate image-loss gradients for a 6DoF probe pose parameterization.

The older finite-difference test checks raw 4x4 pose matrix entries. This test
checks the path needed by pose optimization: image loss -> CUDA d_pose -> a
differentiable 6DoF pose parameter vector.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F

from pose_experiment_common import (
    assert_cuda,
    format_params,
    make_asymmetric_scene,
    print_pose_metrics,
    render_scene,
)


def _objective(cfg, grid, means, scales, quats, opacities, intensities, target, params):
    pred = render_scene(cfg, grid, means, scales, quats, opacities, intensities, params)
    if not torch.isfinite(pred).all():
        raise SystemExit("non-finite render during pose finite-difference check")
    return F.mse_loss(pred, target)


def main() -> None:
    device = assert_cuda()
    dtype = torch.float32
    torch.manual_seed(19)

    cfg, grid, means, scales, quats, opacities, intensities = make_asymmetric_scene(device, dtype)
    target_params = torch.zeros(6, device=device, dtype=dtype)
    with torch.no_grad():
        target = render_scene(cfg, grid, means, scales, quats, opacities, intensities, target_params).detach()

    init = torch.tensor([0.020, -0.017, 0.026, 0.070, -0.055, 0.045], device=device, dtype=dtype)
    params = init.detach().clone().requires_grad_(True)
    loss = _objective(cfg, grid, means, scales, quats, opacities, intensities, target, params)
    loss.backward()

    if params.grad is None or not torch.isfinite(params.grad).all():
        raise SystemExit("missing or non-finite gradient for 6DoF pose params")
    if float(params.grad.abs().max()) <= 1e-10:
        raise SystemExit("pose gradient is numerically zero")

    print(f"phase19 init_params={format_params(init)}")
    print_pose_metrics("phase19 init_pose", init, target_params)
    print(f"phase19 loss={float(loss.detach()):.8e} grad={format_params(params.grad)}")

    max_rel = 0.0
    max_abs = 0.0
    for i, name in enumerate(("rx", "ry", "rz", "tx", "ty", "tz")):
        eps = 5e-4 if i < 3 else 1e-3
        plus = init.detach().clone()
        minus = init.detach().clone()
        plus[i] += eps
        minus[i] -= eps
        lp = _objective(cfg, grid, means, scales, quats, opacities, intensities, target, plus)
        lm = _objective(cfg, grid, means, scales, quats, opacities, intensities, target, minus)
        numeric = float(((lp - lm) / (2.0 * eps)).detach())
        analytic = float(params.grad[i].detach())
        abs_err = abs(analytic - numeric)
        rel_err = abs_err / (abs(numeric) + 1e-8)
        max_rel = max(max_rel, rel_err)
        max_abs = max(max_abs, abs_err)
        print(
            f"phase19 {name}: analytic={analytic:.8e} numeric={numeric:.8e} "
            f"abs_err={abs_err:.8e} rel_err={rel_err:.6f}"
        )
        if abs_err > 2e-4 and rel_err > 0.35:
            raise SystemExit(f"phase19 {name} finite-difference mismatch")

    base_loss = float(loss.detach())
    grad = params.grad.detach()
    denom = grad.abs().max().clamp_min(1e-12)
    improved = False
    for step in (0.10, 0.05, 0.02, 0.01):
        trial = init - step * grad / denom
        trial_loss = float(_objective(cfg, grid, means, scales, quats, opacities, intensities, target, trial).detach())
        print(f"phase19 descent step={step:.3f} trial_loss={trial_loss:.8e}")
        if trial_loss < base_loss:
            improved = True
            break
    if not improved:
        raise SystemExit("phase19 pose gradient did not provide a descent direction")

    print(f"phase19 max_abs_err={max_abs:.8e} max_rel_err={max_rel:.6f}")
    print("validate_us_focus_pose_se3_gradient: OK")


if __name__ == "__main__":
    main()
