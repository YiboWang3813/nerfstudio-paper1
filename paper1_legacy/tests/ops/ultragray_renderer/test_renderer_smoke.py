import pytest
import torch


ultragray_renderer = pytest.importorskip("ultragray_renderer")


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is required for ultragray renderer smoke test")
def test_ultragray_renderer_forward_backward_smoke():
    renderer = ultragray_renderer.UltragrayRenderer(
        ultragray_renderer.UltragrayRenderConfig(
            image_width=4,
            image_height=3,
            width_mm=4.0,
            near_mm=0.0,
            ops_max_offset_mm=0.0,
            quadrature_window_radius_mm=2.0,
        )
    )

    device = torch.device("cuda")
    means = torch.tensor([[0.0, 0.0, 1.0], [0.5, 0.0, 1.5]], device=device, requires_grad=True)
    scales = torch.full((2, 3), 0.5, device=device, requires_grad=True)
    quats = torch.tensor([[1.0, 0.0, 0.0, 0.0], [1.0, 0.0, 0.0, 0.0]], device=device, requires_grad=True)
    attenuations = torch.full((2,), 4.59512, device=device, requires_grad=True)
    sh_coeffs = torch.tensor([[0.2, 0.0, 0.0, 0.0], [0.1, 0.0, 0.0, 0.0]], device=device, requires_grad=True)

    xs = torch.linspace(-1.5, 1.5, 4, device=device)
    zs = torch.linspace(0.5, 2.0, 3, device=device)
    grid_x, grid_z = torch.meshgrid(xs, zs, indexing="ij")
    pixel_positions = torch.stack([grid_x, torch.zeros_like(grid_x), grid_z], dim=-1).contiguous()
    ray_origins = torch.stack([xs, torch.zeros_like(xs), torch.zeros_like(xs)], dim=-1).contiguous()
    ray_dirs = torch.tensor([[0.0, 0.0, 1.0]], device=device).expand(4, 3).contiguous()

    geometry = ultragray_renderer.ProbeGeometry(
        origin_world=torch.zeros(3, device=device),
        axis_lateral_world=torch.tensor([1.0, 0.0, 0.0], device=device),
        axis_elevational_world=torch.tensor([0.0, 1.0, 0.0], device=device),
        axis_axial_world=torch.tensor([0.0, 0.0, 1.0], device=device),
    )

    image = renderer(means, scales, quats, attenuations, sh_coeffs, pixel_positions, ray_origins, ray_dirs, geometry)
    assert image.shape == (3, 4)
    assert torch.isfinite(image).all()

    image.sum().backward()
    for tensor in (means, scales, quats, attenuations, sh_coeffs):
        assert tensor.grad is not None
        assert torch.isfinite(tensor.grad).all()
