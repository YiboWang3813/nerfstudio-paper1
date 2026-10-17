"""
Validate CUDA us_focus additive backward against the PyTorch reference.

This keeps a stable reference sanity check for means/opacities/intensities.
Full covariance and probe-pose gradients are gated by the direct CUDA finite
difference test, which compares the CUDA backward to the CUDA forward it will
actually train against.
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


def _clone_leaf(x: torch.Tensor) -> torch.Tensor:
    y = x.detach().clone()
    y.requires_grad_(True)
    return y


def _check_close(name: str, got: torch.Tensor, expected: torch.Tensor, atol: float, rtol: float) -> None:
    max_abs = (got - expected).abs().max().item()
    rel = max_abs / (expected.abs().max().item() + 1e-8)
    print(f"{name}: max_abs={max_abs:.6e} rel={rel:.6e}")
    if not torch.allclose(got, expected, atol=atol, rtol=rtol):
        raise SystemExit(f"{name} gradient mismatch")


def main() -> None:
    if not torch.cuda.is_available():
        print("SKIP: need CUDA")
        return

    device = torch.device("cuda:0")
    dtype = torch.float32
    cfg = OursRenderConfig(
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
        us_focus_sigma_ele_1=0.0,
        us_focus_sigma_ele_min=0.5,
        us_focus_sigma_ele_max=8.0,
        us_focus_det_gamma=0.25,
        us_focus_det_eps=1e-6,
        us_focus_weight_min=1e-8,
    )
    base_means = torch.tensor(
        [[0.0, 4.0, 0.0], [1.0, 5.0, 0.8], [-1.2, 3.5, 2.5]],
        device=device,
        dtype=dtype,
    )
    scales = torch.tensor(
        [[0.8, 0.9, 0.7], [0.7, 0.8, 0.9], [1.1, 0.6, 0.8]],
        device=device,
        dtype=dtype,
    )
    quats = torch.tensor(
        [[1.0, 0.0, 0.0, 0.0], [1.0, 0.0, 0.0, 0.0], [1.0, 0.0, 0.0, 0.0]],
        device=device,
        dtype=dtype,
    )
    base_opacities = torch.tensor([0.6, 0.4, 0.2], device=device, dtype=dtype)
    base_intensities = torch.tensor([0.75, 0.5, 0.9], device=device, dtype=dtype)
    pose = torch.eye(4, device=device, dtype=dtype)
    pack = build_slice_grid_pack(_dataset_meta(cfg), device=device)
    torch.manual_seed(3)
    upstream = torch.randn((cfg.H, cfg.W), device=device, dtype=dtype) * 0.1

    means = _clone_leaf(base_means)
    opacities = _clone_leaf(base_opacities)
    intensities = _clone_leaf(base_intensities)
    cuda_img = render_projected_slice(cfg, means, scales, quats, opacities, intensities, pose, pack)
    (cuda_img * upstream).sum().backward()

    ref_means = _clone_leaf(base_means)
    ref_opacities = _clone_leaf(base_opacities)
    ref_intensities = _clone_leaf(base_intensities)
    ref_img = render_us_focus_reference(cfg, ref_means, scales, quats, ref_opacities, ref_intensities, pose)
    (ref_img * upstream).sum().backward()

    _check_close("means", means.grad, ref_means.grad, atol=3e-4, rtol=4e-3)
    _check_close("opacities", opacities.grad, ref_opacities.grad, atol=3e-4, rtol=3e-3)
    _check_close("intensities", intensities.grad, ref_intensities.grad, atol=3e-4, rtol=3e-3)
    print("validate_us_focus_backward_cuda: OK")


if __name__ == "__main__":
    main()
