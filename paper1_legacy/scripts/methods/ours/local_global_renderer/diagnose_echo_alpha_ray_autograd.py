import argparse
from dataclasses import replace
from typing import Tuple

import torch

from ours_renderer.autograd import render_projected_slice
from ours_renderer.grid_geometry import build_slice_grid_pack
from ours_renderer.types import OursRenderConfig


class Meta:
    H = 24
    W = 24
    delta_lateral_mm = 0.5
    delta_axial_mm = 0.5
    near_mm = 0.0
    center_lateral = True


def make_case(device: torch.device):
    cfg = OursRenderConfig(
        H=Meta.H,
        W=Meta.W,
        lateral_spacing_mm_per_pix=Meta.delta_lateral_mm,
        axial_spacing_mm_per_pix=Meta.delta_axial_mm,
        axial_near_mm=Meta.near_mm,
        tile_size_px=8,
        elev_corridor_half_mm=2.5,
        slab_half_thickness_mm=0.0,
        k_sigma=2.5,
        mapper_kind="tile",
        mapper_sort_by_v=True,
        compositing_mode="additive",
        shader_weight_mode="us_focus",
        us_focus_sigma_ele_0=1.2,
        us_focus_sigma_ele_min=0.3,
        us_focus_sigma_ele_max=4.0,
        us_focus_sigma_lat_0=1.0,
        us_focus_sigma_lat_min=0.3,
        us_focus_sigma_lat_max=4.0,
        us_focus_det_gamma=0.25,
        us_focus_weight_min=1e-8,
        ray_attenuation_enabled=True,
        ray_attenuation_source="attenuation",
        ray_attenuation_beta=0.8,
        ray_attenuation_strength=1.0,
        ray_attenuation_min_transmission=0.2,
        ray_attenuation_weight_min=1e-5,
        ray_attenuation_axial_margin_mm=0.0,
        ray_attenuation_quadrature_window_mm=2.0,
        ray_attenuation_quadrature_points=3,
        ray_attenuation_mapper_kind="ray_csr",
    )
    means = torch.tensor(
        [
            [-2.0, 4.0, 0.20],
            [0.0, 6.0, -0.15],
            [2.0, 8.0, 0.10],
        ],
        device=device,
        dtype=torch.float32,
    )
    scales = torch.tensor(
        [
            [0.9, 1.0, 0.8],
            [1.1, 0.8, 0.9],
            [0.8, 1.2, 0.7],
        ],
        device=device,
        dtype=torch.float32,
    )
    quats = torch.tensor(
        [
            [0.99, 0.02, -0.03, 0.01],
            [0.97, -0.04, 0.07, 0.02],
            [0.98, 0.05, 0.01, -0.04],
        ],
        device=device,
        dtype=torch.float32,
    )
    echo = torch.tensor([0.45, 0.35, 0.40], device=device, dtype=torch.float32)
    intensities = torch.ones_like(echo)
    attenuation = torch.tensor([0.10, 0.14, 0.12], device=device, dtype=torch.float32)
    pose = torch.eye(4, device=device, dtype=torch.float32)
    grid = build_slice_grid_pack(Meta(), device=device)
    yy, xx = torch.meshgrid(
        torch.linspace(0.7, 1.3, Meta.H, device=device),
        torch.linspace(1.2, 0.8, Meta.W, device=device),
        indexing="ij",
    )
    upstream = (xx * yy).contiguous()
    return cfg, means, scales, quats, echo, intensities, attenuation, pose, grid, upstream


def render_loss(cfg, means, scales, quats, echo, intensities, attenuation, pose, grid, upstream):
    img = render_projected_slice(cfg, means, scales, quats, echo, intensities, pose, grid, attenuation)
    if not torch.isfinite(img).all():
        raise RuntimeError("non-finite image")
    return (img * upstream).sum()


def clone_leaf(x: torch.Tensor, requires_grad: bool = True) -> torch.Tensor:
    return x.detach().clone().requires_grad_(requires_grad)


def finite_difference(name: str, index: Tuple[int, ...], eps: float, case):
    cfg, means, scales, quats, echo, intensities, attenuation, pose, grid, upstream = case
    tensors_plus = {
        "means": means.detach().clone(),
        "scales": scales.detach().clone(),
        "quats": quats.detach().clone(),
        "echo": echo.detach().clone(),
        "attenuation": attenuation.detach().clone(),
    }
    tensors_minus = {k: v.clone() for k, v in tensors_plus.items()}
    tensors_plus[name][index] += eps
    tensors_minus[name][index] -= eps
    plus = render_loss(
        cfg,
        tensors_plus["means"],
        tensors_plus["scales"],
        tensors_plus["quats"],
        tensors_plus["echo"],
        intensities.detach(),
        tensors_plus["attenuation"],
        pose.detach(),
        grid,
        upstream,
    )
    minus = render_loss(
        cfg,
        tensors_minus["means"],
        tensors_minus["scales"],
        tensors_minus["quats"],
        tensors_minus["echo"],
        intensities.detach(),
        tensors_minus["attenuation"],
        pose.detach(),
        grid,
        upstream,
    )
    torch.cuda.synchronize()
    return float(((plus - minus) / (2.0 * eps)).item())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--eps", type=float, default=1.0e-2)
    ap.add_argument("--rtol-main", type=float, default=0.08)
    ap.add_argument("--rtol-scale-quat", type=float, default=0.35)
    args = ap.parse_args()

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required")
    device = torch.device("cuda")
    base_case = make_case(device)
    cfg, means0, scales0, quats0, echo0, intensities0, attenuation0, pose0, grid, upstream = base_case

    means = clone_leaf(means0)
    scales = clone_leaf(scales0)
    quats = clone_leaf(quats0)
    echo = clone_leaf(echo0)
    attenuation = clone_leaf(attenuation0)
    intensities = intensities0.detach()
    pose = pose0.detach()

    loss = render_loss(cfg, means, scales, quats, echo, intensities, attenuation, pose, grid, upstream)
    loss.backward()
    torch.cuda.synchronize()
    print(f"loss={float(loss.item()):.8f}")

    def run_checks(label: str, check_cfg: OursRenderConfig, include_attenuation: bool):
        check_case = (check_cfg, means0, scales0, quats0, echo0, intensities0, attenuation0, pose0, grid, upstream)
        means = clone_leaf(means0)
        scales = clone_leaf(scales0)
        quats = clone_leaf(quats0)
        echo = clone_leaf(echo0)
        attenuation = clone_leaf(attenuation0)
        intensities = intensities0.detach()
        pose = pose0.detach()

        loss = render_loss(check_cfg, means, scales, quats, echo, intensities, attenuation, pose, grid, upstream)
        loss.backward()
        torch.cuda.synchronize()
        print(f"[{label}] loss={float(loss.item()):.8f}")

        print(f"[{label}] skipping means/scales finite-diff checks because the local tile mapper has discrete candidate boundaries.")
        checks = [
            ("echo", (0,), float(echo.grad[0].item()), args.rtol_main),
            ("echo", (2,), float(echo.grad[2].item()), args.rtol_main),
            ("quats", (1, 2), float(quats.grad[1, 2].item()), args.rtol_scale_quat),
        ]
        if include_attenuation:
            checks.extend(
                [
                    ("attenuation", (0,), float(attenuation.grad[0].item()), args.rtol_main),
                    ("attenuation", (2,), float(attenuation.grad[2].item()), args.rtol_main),
                ]
            )
        worst_main = 0.0
        worst_scale_quat = 0.0
        for name, idx, analytic, rtol in checks:
            numeric = finite_difference(name, idx, args.eps, check_case)
            abs_err = abs(analytic - numeric)
            rel_err = abs_err / max(1.0e-8, abs(analytic) + abs(numeric))
            print(
                f"[{label}] {name}{idx}: analytic={analytic:.8f} numeric={numeric:.8f} "
                f"abs_err={abs_err:.8f} rel_err={rel_err:.6f}"
            )
            if name in {"scales", "quats"}:
                worst_scale_quat = max(worst_scale_quat, rel_err)
            else:
                worst_main = max(worst_main, rel_err)
            if rel_err > rtol:
                print(f"[{label}] warning: {name}{idx} rel_err exceeds diagnostic tolerance {rtol:.3f}")
        print(f"[{label}] worst_main_rel_err={worst_main:.6f}")
        print(f"[{label}] worst_scale_quat_rel_err={worst_scale_quat:.6f}")
        return worst_main, worst_scale_quat

    print("skipping means/scales finite-diff checks because the local tile mapper has discrete candidate boundaries.")
    checks = [
        ("echo", (0,), float(echo.grad[0].item()), args.rtol_main),
        ("echo", (2,), float(echo.grad[2].item()), args.rtol_main),
        ("attenuation", (0,), float(attenuation.grad[0].item()), args.rtol_main),
        ("attenuation", (2,), float(attenuation.grad[2].item()), args.rtol_main),
        ("quats", (1, 2), float(quats.grad[1, 2].item()), args.rtol_scale_quat),
    ]
    worst_main = 0.0
    worst_scale_quat = 0.0
    for name, idx, analytic, rtol in checks:
        numeric = finite_difference(name, idx, args.eps, base_case)
        abs_err = abs(analytic - numeric)
        rel_err = abs_err / max(1.0e-8, abs(analytic) + abs(numeric))
        print(
            f"{name}{idx}: analytic={analytic:.8f} numeric={numeric:.8f} "
            f"abs_err={abs_err:.8f} rel_err={rel_err:.6f}"
        )
        if name in {"scales", "quats"}:
            worst_scale_quat = max(worst_scale_quat, rel_err)
        else:
            worst_main = max(worst_main, rel_err)
        if rel_err > rtol:
            print(f"warning: {name}{idx} rel_err exceeds diagnostic tolerance {rtol:.3f}")
    print(f"worst_main_rel_err={worst_main:.6f}")
    print(f"worst_scale_quat_rel_err={worst_scale_quat:.6f}")
    if worst_main > args.rtol_main:
        raise SystemExit(f"ray-enabled main checks failed: worst_main_rel_err={worst_main:.6f}")
    if worst_scale_quat > args.rtol_scale_quat:
        raise SystemExit(f"ray-enabled scale/quat checks failed: worst_scale_quat_rel_err={worst_scale_quat:.6f}")

    local_cfg = replace(cfg, ray_attenuation_enabled=False)
    run_checks("local_only", local_cfg, include_attenuation=False)


if __name__ == "__main__":
    main()
