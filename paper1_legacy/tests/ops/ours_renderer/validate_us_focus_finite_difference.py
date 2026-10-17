"""
Validate staged CUDA us_focus gradients with central finite differences.

This intentionally checks the CUDA forward/backward pair directly, including
the us_focus covariance and probe-pose gradient path.
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Tuple

import torch

_PKG_ROOT = Path(__file__).resolve().parents[3] / "ops" / "ours_renderer"
if str(_PKG_ROOT) not in sys.path:
    sys.path.insert(0, str(_PKG_ROOT))

from ours_renderer.autograd import render_projected_slice
from ours_renderer.grid_geometry import build_slice_grid_pack
from ours_renderer.types import OursRenderConfig


def _dataset_meta(config: OursRenderConfig):
    return type(
        "DM",
        (),
        dict(
            H=config.H,
            W=config.W,
            delta_lateral_mm=config.lateral_spacing_mm_per_pix,
            delta_axial_mm=config.axial_spacing_mm_per_pix,
            near_mm=config.axial_near_mm,
            center_lateral=True,
        ),
    )()


def _clone_leaf(x: torch.Tensor) -> torch.Tensor:
    y = x.detach().clone()
    y.requires_grad_(True)
    return y


def _make_case(
    device: torch.device,
    use_lateral_focus: bool = False,
    sigma_ele_1: float = 0.0,
    sigma_lat_1: float = 0.0,
):
    dtype = torch.float32
    cfg = OursRenderConfig(
        H=12,
        W=12,
        lateral_spacing_mm_per_pix=0.5,
        axial_spacing_mm_per_pix=0.5,
        axial_near_mm=0.0,
        tile_size_px=6,
        elev_corridor_half_mm=4.0,
        k_sigma=30.0,
        mapper_kind="tile",
        compositing_mode="additive",
        mapper_sort_by_v=False,
        shader_weight_mode="us_focus",
        us_focus_sigma_ele_0=1.2,
        us_focus_sigma_ele_1=sigma_ele_1,
        us_focus_sigma_ele_min=0.5,
        us_focus_sigma_ele_max=8.0,
        us_focus_sigma_lat_0=0.8 if use_lateral_focus else 1.0,
        us_focus_sigma_lat_1=sigma_lat_1,
        us_focus_sigma_lat_min=0.5,
        us_focus_sigma_lat_max=8.0,
        us_focus_det_gamma=0.25,
        us_focus_det_eps=1e-6,
        us_focus_weight_min=1e-12,
        us_focus_use_lateral_focus=use_lateral_focus,
    )
    means = torch.tensor(
        [
            [0.0, 2.4, 0.2],
            [0.8, 3.0, 0.6],
            [-0.7, 2.0, -0.5],
        ],
        device=device,
        dtype=dtype,
    )
    scales = torch.tensor(
        [
            [0.7, 0.8, 0.9],
            [0.9, 0.7, 0.8],
            [0.8, 1.0, 0.6],
        ],
        device=device,
        dtype=dtype,
    )
    quats = torch.tensor(
        [
            [0.98, 0.08, 0.02, -0.03],
            [0.96, -0.04, 0.12, 0.05],
            [1.0, 0.03, -0.05, 0.08],
        ],
        device=device,
        dtype=dtype,
    )
    opacities = torch.tensor([0.55, 0.35, 0.45], device=device, dtype=dtype)
    intensities = torch.tensor([0.65, 0.8, 0.5], device=device, dtype=dtype)
    pose = torch.eye(4, device=device, dtype=dtype)
    pose[:3, :3] = torch.tensor(
        [
            [1.0, 0.04, -0.02],
            [-0.03, 0.98, 0.05],
            [0.02, -0.04, 1.01],
        ],
        device=device,
        dtype=dtype,
    )
    pose[:3, 3] = torch.tensor([0.1, -0.15, 0.05], device=device, dtype=dtype)
    pack = build_slice_grid_pack(_dataset_meta(cfg), device=device)
    torch.manual_seed(17)
    upstream = torch.randn((cfg.H, cfg.W), device=device, dtype=dtype) * 0.05
    return cfg, means, scales, quats, opacities, intensities, pose, pack, upstream


def _loss(
    cfg: OursRenderConfig,
    means: torch.Tensor,
    scales: torch.Tensor,
    quats: torch.Tensor,
    opacities: torch.Tensor,
    intensities: torch.Tensor,
    pose: torch.Tensor,
    pack: torch.Tensor,
    upstream: torch.Tensor,
) -> torch.Tensor:
    img = render_projected_slice(cfg, means, scales, quats, opacities, intensities, pose, pack)
    if not torch.isfinite(img).all():
        raise SystemExit("non-finite CUDA image during finite-difference validation")
    return (img * upstream).sum()


def _finite_diff(
    name: str,
    index: Tuple[int, ...],
    eps: float,
    cfg: OursRenderConfig,
    means: torch.Tensor,
    scales: torch.Tensor,
    quats: torch.Tensor,
    opacities: torch.Tensor,
    intensities: torch.Tensor,
    pose: torch.Tensor,
    pack: torch.Tensor,
    upstream: torch.Tensor,
) -> float:
    args_plus = [
        means.detach().clone(),
        scales.detach().clone(),
        quats.detach().clone(),
        opacities.detach().clone(),
        intensities.detach().clone(),
        pose.detach().clone(),
    ]
    args_minus = [
        means.detach().clone(),
        scales.detach().clone(),
        quats.detach().clone(),
        opacities.detach().clone(),
        intensities.detach().clone(),
        pose.detach().clone(),
    ]
    tensor_slot = {"means": 0, "scales": 1, "quats": 2, "opacities": 3, "intensities": 4, "pose": 5}[name]
    args_plus[tensor_slot][index] += eps
    args_minus[tensor_slot][index] -= eps

    loss_plus = _loss(cfg, args_plus[0], args_plus[1], args_plus[2], args_plus[3], args_plus[4], args_plus[5], pack, upstream)
    loss_minus = _loss(
        cfg,
        args_minus[0],
        args_minus[1],
        args_minus[2],
        args_minus[3],
        args_minus[4],
        args_minus[5],
        pack,
        upstream,
    )
    return float(((loss_plus - loss_minus) / (2.0 * eps)).item())


def _check(
    label: str,
    name: str,
    index: Tuple[int, ...],
    analytic: float,
    numeric: float,
    atol: float,
    rtol: float,
) -> None:
    abs_err = abs(analytic - numeric)
    rel_err = abs_err / (abs(numeric) + 1e-8)
    print(
        f"{label}:{name}{index}: analytic={analytic:.6e} numeric={numeric:.6e} "
        f"abs_err={abs_err:.6e} rel_err={rel_err:.6e}"
    )
    if abs_err > atol and rel_err > rtol:
        raise SystemExit(f"{label}:{name}{index} finite-difference gradient mismatch")


def _run_case(label: str, cfg, means0, scales0, quats0, opacities0, intensities0, pose0, pack, upstream) -> None:
    means = _clone_leaf(means0)
    scales = _clone_leaf(scales0)
    quats = _clone_leaf(quats0)
    opacities = _clone_leaf(opacities0)
    intensities = _clone_leaf(intensities0)
    pose = _clone_leaf(pose0)
    loss = _loss(cfg, means, scales, quats, opacities, intensities, pose, pack, upstream)
    loss.backward()

    checks = [
        ("means", (0, 0), 1e-2, float(means.grad[0, 0].item()), 7e-3, 7e-2),
        ("means", (1, 1), 1e-2, float(means.grad[1, 1].item()), 7e-3, 7e-2),
        ("means", (2, 2), 1e-2, float(means.grad[2, 2].item()), 7e-3, 7e-2),
        ("scales", (0, 0), 1e-3, float(scales.grad[0, 0].item()), 1.5e-2, 1.5e-1),
        ("scales", (1, 2), 1e-3, float(scales.grad[1, 2].item()), 1.5e-2, 1.5e-1),
        ("quats", (0, 1), 1e-3, float(quats.grad[0, 1].item()), 2.5e-2, 2.5e-1),
        ("quats", (1, 2), 1e-3, float(quats.grad[1, 2].item()), 2.5e-2, 2.5e-1),
        ("opacities", (0,), 1e-3, float(opacities.grad[0].item()), 7e-3, 7e-2),
        ("intensities", (1,), 1e-3, float(intensities.grad[1].item()), 7e-3, 7e-2),
        ("pose", (0, 3), 1e-3, float(pose.grad[0, 3].item()), 2e-2, 2e-1),
        ("pose", (0, 1), 1e-3, float(pose.grad[0, 1].item()), 2e-2, 2e-1),
        ("pose", (2, 2), 1e-3, float(pose.grad[2, 2].item()), 2e-2, 2e-1),
    ]
    for name, index, eps, analytic, atol, rtol in checks:
        numeric = _finite_diff(
            name,
            index,
            eps,
            cfg,
            means0,
            scales0,
            quats0,
            opacities0,
            intensities0,
            pose0,
            pack,
            upstream,
        )
        _check(label, name, index, analytic, numeric, atol, rtol)

    print(f"validate_us_focus_finite_difference[{label}]: OK")


def main() -> None:
    if not torch.cuda.is_available():
        print("SKIP: need CUDA")
        return

    device = torch.device("cuda:0")
    _run_case("constant_ele", *_make_case(device, use_lateral_focus=False))
    _run_case("constant_ele_lateral", *_make_case(device, use_lateral_focus=True))
    _run_case(
        "depth_adaptive_ele_lateral",
        *_make_case(device, use_lateral_focus=True, sigma_ele_1=0.03, sigma_lat_1=0.02),
    )

    print("validate_us_focus_finite_difference: OK")


if __name__ == "__main__":
    main()
