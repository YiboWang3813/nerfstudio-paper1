import sys
from pathlib import Path

import pytest
import torch

OPS_PATH = Path(__file__).resolve().parents[3] / "ops" / "gaussianplie_renderer"
if str(OPS_PATH) not in sys.path:
    sys.path.insert(0, str(OPS_PATH))

from gaussianplie_renderer import GaussianPlieRenderConfig, GaussianPlieRenderer
from gaussianplie_renderer.grid_geometry import build_slice_grid_pack


class _Meta:
    H = 8
    W = 8
    delta_lateral_mm = 0.2
    delta_axial_mm = 0.2
    delta_elevational_mm = 0.2
    near_mm = 0.0
    center_lateral = True
    probe_type = "linear"


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is required for wrapper smoke.")
def test_cuda_wrapper_forward_backward_smoke():
    device = torch.device("cuda")
    cfg = GaussianPlieRenderConfig(
        H=8,
        W=8,
        lateral_spacing_mm_per_pix=0.2,
        axial_spacing_mm_per_pix=0.2,
        axial_near_mm=0.0,
        sigma_z_mm=0.2,
    )
    renderer = GaussianPlieRenderer(cfg).to(device)
    renderer.set_slice_grid_geometry(build_slice_grid_pack(_Meta(), device=device))

    means = torch.tensor([[0.0, 0.8, 0.0], [0.2, 0.8, 0.1]], device=device, requires_grad=True)
    scales = torch.full((2, 3), 0.3, device=device, requires_grad=True)
    quats = torch.tensor([[1.0, 0.0, 0.0, 0.0], [1.0, 0.0, 0.0, 0.0]], device=device, requires_grad=True)
    alphas = torch.full((2,), 0.2, device=device, requires_grad=True)
    pose = torch.eye(4, device=device)

    image = renderer(means, scales, quats, alphas, pose)
    assert image.shape == (8, 8)
    assert torch.isfinite(image).all()
    loss = image.sum()
    loss.backward()

    assert means.grad is not None
    assert scales.grad is not None
    assert quats.grad is not None
    assert alphas.grad is not None
