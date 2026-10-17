import sys
from pathlib import Path

import torch

OPS_PATH = Path(__file__).resolve().parents[3] / "ops" / "gaussianplie_renderer"
if str(OPS_PATH) not in sys.path:
    sys.path.insert(0, str(OPS_PATH))

from gaussianplie_renderer.reference import build_covariance, focus_project, render_reference, world_to_camera


def _case(dtype=torch.float64):
    means = torch.tensor([[0.0, 0.8, 0.0], [0.25, 0.8, 0.1]], dtype=dtype)
    scales = torch.full((2, 3), 0.3, dtype=dtype)
    quats = torch.tensor([[1.0, 0.0, 0.0, 0.0], [1.0, 0.0, 0.0, 0.0]], dtype=dtype)
    alphas = torch.tensor([0.2, 0.15], dtype=dtype)
    pose = torch.eye(4, dtype=dtype)
    xs = torch.linspace(-0.4, 0.4, 5, dtype=dtype)
    ys = torch.linspace(0.4, 1.2, 5, dtype=dtype)
    yy, xx = torch.meshgrid(ys, xs, indexing="ij")
    xy_grid = torch.stack([xx, yy], dim=-1)
    return means, scales, quats, alphas, pose, xy_grid


def test_reference_render_is_additive_in_single_alpha():
    means, scales, quats, alphas, pose, xy_grid = _case()
    image = render_reference(means, scales, quats, alphas, pose, xy_grid, sigma_z=0.2)
    doubled = render_reference(means, scales, quats, alphas * 2.0, pose, xy_grid, sigma_z=0.2)

    torch.testing.assert_close(doubled, image * 2.0, atol=1.0e-10, rtol=1.0e-10)


def test_reference_has_single_alpha_gradient():
    means, scales, quats, alphas, pose, xy_grid = _case()
    alphas = alphas.clone().requires_grad_(True)
    image = render_reference(means, scales, quats, alphas, pose, xy_grid, sigma_z=0.2)
    image.sum().backward()

    assert alphas.grad is not None
    assert alphas.grad.shape == alphas.shape
    assert torch.all(torch.isfinite(alphas.grad))


def test_focus_response_suppresses_off_plane_gaussian():
    means, scales, quats, alphas, pose, _xy_grid = _case()
    covs = build_covariance(scales, quats)
    means_c, covs_c = world_to_camera(means, covs, pose)

    _m2d, _c2d, alpha_tilde = focus_project(means_c, covs_c, alphas, sigma_z=0.05)

    assert alpha_tilde[1] < alpha_tilde[0]
