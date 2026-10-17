"""
Validate the PyTorch reference for the staged us_focus renderer path.

Run:
  PYTHONPATH=ops/ours_renderer python tests/ops/ours_renderer/validate_us_focus_reference.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import torch

_PKG_ROOT = Path(__file__).resolve().parents[3] / "ops" / "ours_renderer"
if str(_PKG_ROOT) not in sys.path:
    sys.path.insert(0, str(_PKG_ROOT))

from ours_renderer import OursRenderConfig
from ours_renderer.us_focus_reference import (
    render_us_focus_reference,
    us_focus_gaussian_footprint_reference,
)


def _cfg(**kwargs) -> OursRenderConfig:
    values = dict(
        H=16,
        W=16,
        lateral_spacing_mm_per_pix=0.5,
        axial_spacing_mm_per_pix=0.5,
        axial_near_mm=0.0,
        mapper_kind="tile",
        compositing_mode="additive",
        mapper_sort_by_v=False,
        shader_weight_mode="us_focus",
        us_focus_sigma_ele_0=1.0,
        us_focus_sigma_ele_1=0.0,
        us_focus_sigma_ele_min=0.5,
        us_focus_sigma_ele_max=8.0,
        us_focus_det_gamma=0.25,
        us_focus_det_eps=1e-6,
        us_focus_weight_min=1e-9,
    )
    values.update(kwargs)
    return OursRenderConfig(**values)


def _single_gaussian(mu_ele: float = 0.0):
    dtype = torch.float64
    means = torch.tensor([[0.0, 4.0, mu_ele]], dtype=dtype)
    scales = torch.tensor([[0.8, 0.9, 0.7]], dtype=dtype)
    quats = torch.tensor([[1.0, 0.0, 0.0, 0.0]], dtype=dtype)
    opacities = torch.tensor([0.6], dtype=dtype)
    intensities = torch.tensor([0.75], dtype=dtype)
    pose = torch.eye(4, dtype=dtype)
    return means, scales, quats, opacities, intensities, pose


def test_slice_response_at_slice_is_one() -> None:
    means, scales, quats, _, _, pose = _single_gaussian(mu_ele=0.0)
    terms = us_focus_gaussian_footprint_reference(_cfg(), means, scales, quats, pose)
    assert torch.allclose(terms["slice_response"], torch.ones_like(terms["slice_response"]), atol=1e-10)


def test_far_slice_response_decays_and_sigma_ele_relaxes_it() -> None:
    means, scales, quats, _, _, pose = _single_gaussian(mu_ele=6.0)
    small = us_focus_gaussian_footprint_reference(_cfg(us_focus_sigma_ele_0=1.0), means, scales, quats, pose)
    large = us_focus_gaussian_footprint_reference(_cfg(us_focus_sigma_ele_0=4.0), means, scales, quats, pose)
    assert small["slice_response"].item() < 1e-3
    assert large["slice_response"].item() > small["slice_response"].item()


def test_det_gamma_zero_disables_norm() -> None:
    means, scales, quats, _, _, pose = _single_gaussian(mu_ele=0.0)
    terms = us_focus_gaussian_footprint_reference(_cfg(us_focus_det_gamma=0.0), means, scales, quats, pose)
    assert torch.allclose(terms["norm_factor"], torch.ones_like(terms["norm_factor"]), atol=1e-10)


def test_lateral_focus_reduces_lateral_variance() -> None:
    means, scales, quats, _, _, pose = _single_gaussian(mu_ele=0.0)
    no_lat = us_focus_gaussian_footprint_reference(
        _cfg(us_focus_use_lateral_focus=False),
        means,
        scales,
        quats,
        pose,
    )
    with_lat = us_focus_gaussian_footprint_reference(
        _cfg(us_focus_use_lateral_focus=True, us_focus_sigma_lat_0=0.8),
        means,
        scales,
        quats,
        pose,
    )
    assert with_lat["sigma2"][0, 0, 0].item() < no_lat["sigma2"][0, 0, 0].item()


def test_reference_render_is_finite_and_differentiable() -> None:
    means, scales, quats, opacities, intensities, pose = _single_gaussian(mu_ele=0.2)
    means.requires_grad_(True)
    img = render_us_focus_reference(_cfg(), means, scales, quats, opacities, intensities, pose)
    assert img.shape == (16, 16)
    assert torch.isfinite(img).all()
    assert img.max().item() > 0.0
    img.sum().backward()
    assert means.grad is not None
    assert torch.isfinite(means.grad).all()


def main() -> None:
    test_slice_response_at_slice_is_one()
    test_far_slice_response_decays_and_sigma_ele_relaxes_it()
    test_det_gamma_zero_disables_norm()
    test_lateral_focus_reduces_lateral_variance()
    test_reference_render_is_finite_and_differentiable()
    print("validate_us_focus_reference: OK")


if __name__ == "__main__":
    main()
