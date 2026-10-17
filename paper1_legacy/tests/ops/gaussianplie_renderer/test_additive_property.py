import sys
from pathlib import Path

import torch

OPS_PATH = Path(__file__).resolve().parents[3] / "ops" / "gaussianplie_renderer"
if str(OPS_PATH) not in sys.path:
    sys.path.insert(0, str(OPS_PATH))

from gaussianplie_renderer.reference import render_reference


def test_reference_render_is_additive_for_identical_gaussians():
    dtype = torch.float64
    means = torch.tensor([[0.0, 0.0, 0.0]], dtype=dtype)
    scales = torch.tensor([[0.5, 0.5, 0.5]], dtype=dtype)
    quats = torch.tensor([[1.0, 0.0, 0.0, 0.0]], dtype=dtype)
    opacities = torch.tensor([0.7], dtype=dtype)
    pose = torch.eye(4, dtype=dtype)
    ys, xs = torch.meshgrid(
        torch.linspace(-1.0, 1.0, 5, dtype=dtype),
        torch.linspace(-1.0, 1.0, 5, dtype=dtype),
        indexing="ij",
    )
    grid = torch.stack([xs, ys], dim=-1)

    one = render_reference(means, scales, quats, opacities, pose, grid, sigma_z=0.5)
    two = render_reference(
        means.repeat(2, 1),
        scales.repeat(2, 1),
        quats.repeat(2, 1),
        opacities.repeat(2),
        pose,
        grid,
        sigma_z=0.5,
    )

    assert torch.allclose(two, one * 2.0, rtol=1.0e-6, atol=1.0e-6)
