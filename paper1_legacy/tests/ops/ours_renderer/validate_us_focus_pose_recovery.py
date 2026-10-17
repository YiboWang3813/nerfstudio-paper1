"""
Phase 20: frozen-Gaussian pose-only recovery through the CUDA us_focus path.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F

from pose_experiment_common import (
    assert_cuda,
    format_params,
    make_asymmetric_scene,
    print_pose_metrics,
    psnr,
    render_scene,
    rotation_error_deg,
    translation_error_mm,
)


def _run_case(label: str, init_values: list[float]) -> None:
    device = assert_cuda()
    dtype = torch.float32
    cfg, grid, means, scales, quats, opacities, intensities = make_asymmetric_scene(device, dtype)
    target_params = torch.zeros(6, device=device, dtype=dtype)
    with torch.no_grad():
        target = render_scene(cfg, grid, means, scales, quats, opacities, intensities, target_params).detach()

    init = torch.tensor(init_values, device=device, dtype=dtype)
    params = init.detach().clone().requires_grad_(True)
    opt = torch.optim.Adam([params], lr=2.5e-2)

    with torch.no_grad():
        init_pred = render_scene(cfg, grid, means, scales, quats, opacities, intensities, params)
        init_loss = float(F.mse_loss(init_pred, target))
        init_psnr = psnr(init_pred, target)
        init_rot = rotation_error_deg(init, target_params)
        init_trans = translation_error_mm(init, target_params)

    for step in range(260):
        opt.zero_grad(set_to_none=True)
        pred = render_scene(cfg, grid, means, scales, quats, opacities, intensities, params)
        loss = F.mse_loss(pred, target)
        loss.backward()
        if params.grad is None or not torch.isfinite(params.grad).all():
            raise SystemExit(f"{label}: non-finite pose gradient at step {step}")
        opt.step()
        if step in (0, 25, 50, 100, 180, 259):
            print(f"phase20 {label} step={step:03d} loss={float(loss.detach()):.8e} params={format_params(params)}")

    with torch.no_grad():
        final_pred = render_scene(cfg, grid, means, scales, quats, opacities, intensities, params)
        final_loss = float(F.mse_loss(final_pred, target))
        final_psnr = psnr(final_pred, target)
        final_rot = rotation_error_deg(params, target_params)
        final_trans = translation_error_mm(params, target_params)

    print_pose_metrics(f"phase20 {label} final_pose", params, target_params)
    print(
        f"phase20 {label}: init_loss={init_loss:.8e} final_loss={final_loss:.8e} "
        f"ratio={final_loss / max(init_loss, 1e-12):.6f} init_psnr={init_psnr:.4f} final_psnr={final_psnr:.4f} "
        f"init_rot={init_rot:.4f} final_rot={final_rot:.4f} init_trans={init_trans:.4f} final_trans={final_trans:.4f}"
    )

    if final_loss > init_loss * 0.15:
        raise SystemExit(f"{label}: pose-only recovery did not reduce loss enough")
    if final_psnr < init_psnr + 8.0:
        raise SystemExit(f"{label}: pose-only recovery did not improve PSNR enough")
    if final_rot >= init_rot or final_trans >= init_trans:
        raise SystemExit(f"{label}: pose-only recovery did not reduce pose error")


def main() -> None:
    torch.manual_seed(20)
    cases = {
        "small": [0.035, -0.025, 0.030, 0.180, -0.140, 0.100],
        "medium": [0.065, -0.050, 0.055, 0.360, -0.300, 0.220],
    }
    for label, init_values in cases.items():
        _run_case(label, init_values)
    print("validate_us_focus_pose_recovery: OK")


if __name__ == "__main__":
    main()
