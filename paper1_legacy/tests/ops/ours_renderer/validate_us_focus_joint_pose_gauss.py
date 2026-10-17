"""
Phase 21: controlled joint Gaussian + pose recovery through CUDA us_focus.

This is a small synthetic optimization test. It does not replace full liver
training; it checks that pose gradients remain useful when Gaussian attributes
are trainable in the same graph.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F

from pose_experiment_common import (
    assert_cuda,
    format_params,
    make_asymmetric_scene,
    psnr,
    render_scene,
    rotation_error_deg,
    translation_error_mm,
)


def _optimize(label, optimize_pose: bool, optimize_gauss: bool):
    device = assert_cuda()
    dtype = torch.float32
    cfg, grid, target_means, scales, quats, target_opacities, target_intensities = make_asymmetric_scene(device, dtype)
    target_params = torch.zeros(6, device=device, dtype=dtype)
    with torch.no_grad():
        target = render_scene(cfg, grid, target_means, scales, quats, target_opacities, target_intensities, target_params)

    pose_init = torch.tensor([0.050, -0.040, 0.045, 0.300, -0.240, 0.160], device=device, dtype=dtype)
    mean_offsets = torch.tensor(
        [
            [0.12, -0.08, 0.05],
            [-0.09, 0.10, -0.04],
            [0.08, -0.06, 0.07],
            [-0.10, 0.09, -0.06],
            [0.06, 0.08, 0.05],
            [-0.07, -0.09, 0.04],
            [0.09, 0.05, -0.05],
        ],
        device=device,
        dtype=dtype,
    )

    params = pose_init.detach().clone().requires_grad_(optimize_pose)
    means = (target_means + mean_offsets).detach().clone().requires_grad_(optimize_gauss)
    opacity_logits = torch.logit((target_opacities * 0.90).clamp(1e-4, 1.0 - 1e-4)).detach().clone()
    intensity_logits = torch.logit((target_intensities * 0.90).clamp(1e-4, 1.0 - 1e-4)).detach().clone()
    opacity_logits.requires_grad_(optimize_gauss)
    intensity_logits.requires_grad_(optimize_gauss)

    groups = []
    if optimize_pose:
        groups.append({"params": [params], "lr": 2.0e-2})
    if optimize_gauss:
        groups.append({"params": [means], "lr": 4.0e-3})
        groups.append({"params": [opacity_logits, intensity_logits], "lr": 1.5e-2})
    opt = torch.optim.Adam(groups)

    with torch.no_grad():
        init_pred = render_scene(
            cfg,
            grid,
            means,
            scales,
            quats,
            torch.sigmoid(opacity_logits),
            torch.sigmoid(intensity_logits),
            params,
        )
        init_loss = float(F.mse_loss(init_pred, target))
        init_psnr = psnr(init_pred, target)
        init_rot = rotation_error_deg(params, target_params)
        init_trans = translation_error_mm(params, target_params)

    first_pose_grad = None
    for step in range(320):
        opt.zero_grad(set_to_none=True)
        pred = render_scene(
            cfg,
            grid,
            means,
            scales,
            quats,
            torch.sigmoid(opacity_logits),
            torch.sigmoid(intensity_logits),
            params,
        )
        image_loss = F.mse_loss(pred, target)
        mean_prior = F.mse_loss(means, target_means) if optimize_gauss else image_loss.new_zeros(())
        loss = image_loss + 5e-4 * mean_prior
        loss.backward()
        if optimize_pose:
            if params.grad is None or not torch.isfinite(params.grad).all():
                raise SystemExit(f"{label}: non-finite pose gradient")
            if first_pose_grad is None:
                first_pose_grad = float(params.grad.abs().max().detach())
        opt.step()
        if step in (0, 50, 120, 220, 319):
            print(
                f"phase21 {label} step={step:03d} image_loss={float(image_loss.detach()):.8e} "
                f"pose={format_params(params)}"
            )

    with torch.no_grad():
        final_pred = render_scene(
            cfg,
            grid,
            means,
            scales,
            quats,
            torch.sigmoid(opacity_logits),
            torch.sigmoid(intensity_logits),
            params,
        )
        final_loss = float(F.mse_loss(final_pred, target))
        final_psnr = psnr(final_pred, target)
        final_rot = rotation_error_deg(params, target_params)
        final_trans = translation_error_mm(params, target_params)
        mean_rmse = float(torch.sqrt(F.mse_loss(means, target_means)).detach())

    print(
        f"phase21 {label}: init_loss={init_loss:.8e} final_loss={final_loss:.8e} "
        f"ratio={final_loss / max(init_loss, 1e-12):.6f} init_psnr={init_psnr:.4f} final_psnr={final_psnr:.4f} "
        f"init_rot={init_rot:.4f} final_rot={final_rot:.4f} init_trans={init_trans:.4f} final_trans={final_trans:.4f} "
        f"mean_rmse={mean_rmse:.6f} first_pose_grad={first_pose_grad}"
    )
    return {
        "label": label,
        "init_loss": init_loss,
        "final_loss": final_loss,
        "init_psnr": init_psnr,
        "final_psnr": final_psnr,
        "init_rot": init_rot,
        "final_rot": final_rot,
        "init_trans": init_trans,
        "final_trans": final_trans,
        "first_pose_grad": first_pose_grad,
    }


def main() -> None:
    torch.manual_seed(21)
    gauss_only = _optimize("gauss_only_fixed_wrong_pose", optimize_pose=False, optimize_gauss=True)
    pose_only = _optimize("pose_only_fixed_gauss", optimize_pose=True, optimize_gauss=False)
    joint = _optimize("joint_pose_gauss", optimize_pose=True, optimize_gauss=True)

    if pose_only["first_pose_grad"] is None or pose_only["first_pose_grad"] <= 1e-10:
        raise SystemExit("phase21 pose-only branch has no useful pose gradient")
    if joint["first_pose_grad"] is None or joint["first_pose_grad"] <= 1e-10:
        raise SystemExit("phase21 joint branch has no useful pose gradient")
    if joint["final_loss"] > joint["init_loss"] * 0.12:
        raise SystemExit("phase21 joint optimization did not reduce image loss enough")
    if joint["final_rot"] >= joint["init_rot"] or joint["final_trans"] >= joint["init_trans"]:
        raise SystemExit("phase21 joint optimization did not reduce pose error")
    if joint["final_psnr"] < pose_only["final_psnr"] - 1.0:
        raise SystemExit("phase21 joint optimization is unexpectedly worse than pose-only")
    if gauss_only["final_trans"] != gauss_only["init_trans"]:
        raise SystemExit("phase21 gauss-only control unexpectedly changed pose")

    print("validate_us_focus_joint_pose_gauss: OK")


if __name__ == "__main__":
    main()
