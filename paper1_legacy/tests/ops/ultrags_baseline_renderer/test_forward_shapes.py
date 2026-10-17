import pytest
import torch

from methods.ultrags_baseline.camera_adapter import _projection_matrix


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA renderer test requires GPU")
def test_renderer_forward_shapes():
    renderer = pytest.importorskip("ultrags_baseline_renderer")
    pytest.importorskip("ultrags_baseline_renderer._C")

    means = torch.tensor([[0.0, 0.0, 2.0], [0.1, 0.1, 2.5]], device="cuda", requires_grad=True)
    means2d = torch.zeros_like(means, requires_grad=True)
    shs = torch.zeros((2, 9, 3), device="cuda", requires_grad=True)
    opacities = torch.full((2, 1), 0.5, device="cuda", requires_grad=True)
    scales = torch.full((2, 2), 0.05, device="cuda", requires_grad=True)
    rotations = torch.zeros((2, 4), device="cuda", requires_grad=True)
    rotations.data[:, 0] = 1.0

    view = torch.eye(4, device="cuda")
    proj = _projection_matrix(0.01, 100.0, 1.0, 1.0, torch.device("cuda"))
    settings = renderer.GaussianRasterizationSettings(
        image_height=16,
        image_width=16,
        tanfovx=0.5,
        tanfovy=0.5,
        bg=torch.zeros(3, device="cuda"),
        scale_modifier=1.0,
        viewmatrix=view,
        projmatrix=proj,
        sh_degree=2,
        campos=torch.zeros(3, device="cuda"),
        prefiltered=False,
        debug=False,
    )
    rasterizer = renderer.GaussianRasterizer(settings)
    image, radii, allmap = rasterizer(
        means3D=means,
        means2D=means2d,
        shs=shs,
        opacities=opacities,
        scales=scales,
        rotations=rotations,
    )
    assert image.shape == (3, 16, 16)
    assert radii.shape == (2,)
    assert allmap.shape[-2:] == (16, 16)
    loss = image.mean() + allmap.mean()
    loss.backward()
    assert means.grad is not None
