"""
Validate CUDA us_focus additive forward against the PyTorch reference.

Run after rebuilding on a GPU node:
  PYTHONPATH=ops/ours_renderer python tests/ops/ours_renderer/validate_us_focus_forward_cuda.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import torch

_PKG_ROOT = Path(__file__).resolve().parents[3] / "ops" / "ours_renderer"
if str(_PKG_ROOT) not in sys.path:
    sys.path.insert(0, str(_PKG_ROOT))

from ours_renderer.autograd import render_projected_slice
from ours_renderer.grid_geometry import build_slice_grid_pack
from ours_renderer.types import OursRenderConfig
from ours_renderer.us_focus_reference import render_us_focus_reference


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


def _cfg(**kwargs) -> OursRenderConfig:
    values = dict(
        H=16,
        W=16,
        lateral_spacing_mm_per_pix=0.5,
        axial_spacing_mm_per_pix=0.5,
        axial_near_mm=0.0,
        tile_size_px=8,
        elev_corridor_half_mm=2.0,
        k_sigma=8.0,
        mapper_kind="tile",
        compositing_mode="additive",
        mapper_sort_by_v=False,
        shader_weight_mode="us_focus",
        us_focus_sigma_ele_0=1.2,
        us_focus_sigma_ele_1=0.02,
        us_focus_sigma_ele_min=0.5,
        us_focus_sigma_ele_max=8.0,
        us_focus_det_gamma=0.25,
        us_focus_det_eps=1e-6,
        us_focus_weight_min=1e-8,
    )
    values.update(kwargs)
    return OursRenderConfig(**values)


def _run_case(label: str, cfg: OursRenderConfig, device: torch.device) -> None:
    dtype = torch.float32
    means = torch.tensor(
        [
            [0.0, 4.0, 0.0],
            [1.0, 5.0, 0.8],
            [-1.2, 3.5, 2.5],
        ],
        device=device,
        dtype=dtype,
    )
    scales = torch.tensor(
        [
            [0.8, 0.9, 0.7],
            [0.7, 0.8, 0.9],
            [1.1, 0.6, 0.8],
        ],
        device=device,
        dtype=dtype,
    )
    quats = torch.tensor(
        [
            [1.0, 0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0, 0.0],
        ],
        device=device,
        dtype=dtype,
    )
    opacities = torch.tensor([0.6, 0.4, 0.2], device=device, dtype=dtype)
    intensities = torch.tensor([0.75, 0.5, 0.9], device=device, dtype=dtype)
    pose = torch.eye(4, device=device, dtype=dtype)
    pack = build_slice_grid_pack(_dataset_meta(cfg), device=device)

    cuda_img = render_projected_slice(cfg, means, scales, quats, opacities, intensities, pose, pack)
    ref_img = render_us_focus_reference(cfg, means, scales, quats, opacities, intensities, pose)

    max_abs = (cuda_img - ref_img).abs().max().item()
    max_ref = ref_img.abs().max().item()
    rel = max_abs / (max_ref + 1e-8)
    print(f"us_focus forward max_abs={max_abs:.6e} rel={rel:.6e} max_ref={max_ref:.6e}")
    if not torch.allclose(cuda_img, ref_img, atol=2e-4, rtol=2e-3):
        raise SystemExit(f"CUDA us_focus forward does not match PyTorch reference for {label}")
    print(f"validate_us_focus_forward_cuda[{label}]: OK")


def main() -> None:
    if not torch.cuda.is_available():
        print("SKIP: need CUDA")
        return

    device = torch.device("cuda:0")
    _run_case("depth_adaptive_ele", _cfg(), device)
    _run_case(
        "depth_adaptive_ele_lateral",
        _cfg(
            us_focus_use_lateral_focus=True,
            us_focus_sigma_lat_0=0.8,
            us_focus_sigma_lat_1=0.01,
            us_focus_sigma_lat_min=0.4,
        ),
        device,
    )
    print("validate_us_focus_forward_cuda: OK")


if __name__ == "__main__":
    main()
