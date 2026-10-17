from __future__ import annotations

import math
import sys
from pathlib import Path

import torch
import torch.nn.functional as F

_PKG_ROOT = Path(__file__).resolve().parents[3] / "ops" / "ours_renderer"
if str(_PKG_ROOT) not in sys.path:
    sys.path.insert(0, str(_PKG_ROOT))

from ours_renderer.autograd import render_projected_slice
from ours_renderer.grid_geometry import build_slice_grid_pack
from ours_renderer.types import OursRenderConfig


def dataset_meta(config: OursRenderConfig):
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


def us_focus_pose_test_config(H: int = 40, W: int = 40) -> OursRenderConfig:
    return OursRenderConfig(
        H=H,
        W=W,
        lateral_spacing_mm_per_pix=0.28,
        axial_spacing_mm_per_pix=0.28,
        axial_near_mm=0.7,
        tile_size_px=8,
        elev_corridor_half_mm=3.0,
        k_sigma=6.0,
        mapper_kind="tile",
        compositing_mode="additive",
        mapper_sort_by_v=False,
        shader_weight_mode="us_focus",
        us_focus_sigma_ele_0=1.0,
        us_focus_sigma_ele_1=0.02,
        us_focus_sigma_ele_min=0.8,
        us_focus_sigma_ele_max=5.0,
        us_focus_sigma_lat_0=16.0,
        us_focus_sigma_lat_1=0.0,
        us_focus_sigma_lat_min=16.0,
        us_focus_sigma_lat_max=16.0,
        us_focus_det_gamma=0.25,
        us_focus_det_eps=1e-6,
        us_focus_weight_min=1e-8,
        us_focus_use_lateral_focus=False,
    )


def params_to_pose(params: torch.Tensor) -> torch.Tensor:
    """Map rx, ry, rz, tx, ty, tz to a differentiable probe-to-world pose."""
    rx, ry, rz, tx, ty, tz = params.unbind()
    z0 = torch.zeros((), dtype=params.dtype, device=params.device)
    o1 = torch.ones((), dtype=params.dtype, device=params.device)

    cx, sx = torch.cos(rx), torch.sin(rx)
    cy, sy = torch.cos(ry), torch.sin(ry)
    cz, sz = torch.cos(rz), torch.sin(rz)

    Rx = torch.stack(
        [
            torch.stack([o1, z0, z0]),
            torch.stack([z0, cx, -sx]),
            torch.stack([z0, sx, cx]),
        ]
    )
    Ry = torch.stack(
        [
            torch.stack([cy, z0, sy]),
            torch.stack([z0, o1, z0]),
            torch.stack([-sy, z0, cy]),
        ]
    )
    Rz = torch.stack(
        [
            torch.stack([cz, -sz, z0]),
            torch.stack([sz, cz, z0]),
            torch.stack([z0, z0, o1]),
        ]
    )
    R = Rz @ Ry @ Rx

    top = torch.cat([R, torch.stack([tx, ty, tz]).reshape(3, 1)], dim=1)
    bottom = params.new_tensor([[0.0, 0.0, 0.0, 1.0]])
    return torch.cat([top, bottom], dim=0)


def rotation_error_deg(params: torch.Tensor, target_params: torch.Tensor | None = None) -> float:
    if target_params is None:
        target_params = torch.zeros_like(params)
    R = params_to_pose(params)[:3, :3]
    R_t = params_to_pose(target_params.to(device=params.device, dtype=params.dtype))[:3, :3]
    cos_theta = ((R_t.transpose(0, 1) @ R).trace() - 1.0) * 0.5
    cos_theta = torch.clamp(cos_theta, -1.0, 1.0)
    return float(torch.rad2deg(torch.acos(cos_theta)).detach().cpu())


def translation_error_mm(params: torch.Tensor, target_params: torch.Tensor | None = None) -> float:
    if target_params is None:
        target_params = torch.zeros_like(params)
    delta = params[3:6] - target_params.to(device=params.device, dtype=params.dtype)[3:6]
    return float(torch.linalg.norm(delta).detach().cpu())


def psnr(pred: torch.Tensor, target: torch.Tensor) -> float:
    mse = F.mse_loss(pred, target).detach().clamp_min(1e-12)
    return float((-10.0 * torch.log10(mse)).cpu())


def make_asymmetric_scene(device: torch.device, dtype: torch.dtype = torch.float32):
    cfg = us_focus_pose_test_config()
    grid = build_slice_grid_pack(dataset_meta(cfg), device=device)
    means = torch.tensor(
        [
            [-2.8, 2.0, -0.40],
            [-1.3, 3.7, 0.35],
            [0.2, 2.8, 0.70],
            [1.7, 5.4, -0.55],
            [2.6, 4.2, 0.18],
            [-0.5, 7.2, -0.85],
            [1.1, 8.3, 0.60],
        ],
        device=device,
        dtype=dtype,
    )
    scales = torch.tensor(
        [
            [0.45, 0.75, 0.55],
            [0.70, 0.50, 0.80],
            [0.55, 0.90, 0.65],
            [0.80, 0.55, 0.50],
            [0.60, 0.65, 0.95],
            [0.90, 0.70, 0.55],
            [0.50, 0.80, 0.70],
        ],
        device=device,
        dtype=dtype,
    )
    quats = F.normalize(
        torch.tensor(
            [
                [0.99, 0.05, 0.08, -0.02],
                [0.96, -0.10, 0.03, 0.12],
                [0.98, 0.02, -0.14, 0.06],
                [0.97, 0.12, 0.04, -0.08],
                [0.95, -0.05, 0.16, 0.03],
                [0.99, 0.07, -0.03, 0.09],
                [0.96, -0.11, -0.06, 0.07],
            ],
            device=device,
            dtype=dtype,
        ),
        dim=-1,
    )
    opacities = torch.tensor([0.45, 0.62, 0.38, 0.56, 0.48, 0.35, 0.52], device=device, dtype=dtype)
    intensities = torch.tensor([0.70, 0.40, 0.92, 0.58, 0.76, 0.50, 0.86], device=device, dtype=dtype)
    return cfg, grid, means, scales, quats, opacities, intensities


def render_scene(
    cfg: OursRenderConfig,
    grid: torch.Tensor,
    means: torch.Tensor,
    scales: torch.Tensor,
    quats: torch.Tensor,
    opacities: torch.Tensor,
    intensities: torch.Tensor,
    params: torch.Tensor,
) -> torch.Tensor:
    return render_projected_slice(
        cfg,
        means,
        scales,
        quats,
        opacities,
        intensities,
        params_to_pose(params),
        grid,
    )


def print_pose_metrics(prefix: str, params: torch.Tensor, target_params: torch.Tensor | None = None) -> None:
    print(
        f"{prefix}: rot_err_deg={rotation_error_deg(params, target_params):.6f} "
        f"trans_err_mm={translation_error_mm(params, target_params):.6f}"
    )


def assert_cuda() -> torch.device:
    if not torch.cuda.is_available():
        raise SystemExit("SKIP: need CUDA for ours_renderer pose validation")
    return torch.device("cuda:0")


def format_params(params: torch.Tensor) -> str:
    values = [float(x) for x in params.detach().cpu()]
    return "[" + ", ".join(f"{v:.6f}" for v in values) + "]"


def radians_to_degrees(value: float) -> float:
    return value * 180.0 / math.pi
