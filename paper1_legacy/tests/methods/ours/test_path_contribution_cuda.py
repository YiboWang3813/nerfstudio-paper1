from types import SimpleNamespace

import pytest
import torch

from ours_renderer import OursRenderConfig, OursRenderer, build_slice_grid_pack, path_contribution_stats


pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is required")


def _case(attenuations: torch.Tensor):
    device = torch.device("cuda")
    meta = SimpleNamespace(
        H=8,
        W=3,
        delta_lateral_mm=0.5,
        delta_axial_mm=1.0,
        near_mm=0.0,
        center_lateral=True,
        probe_type="linear",
    )
    cfg = OursRenderConfig(
        H=meta.H,
        W=meta.W,
        lateral_spacing_mm_per_pix=meta.delta_lateral_mm,
        axial_spacing_mm_per_pix=meta.delta_axial_mm,
        axial_near_mm=meta.near_mm,
        tile_size_px=2,
        elev_corridor_half_mm=1.0,
        k_sigma=2.0,
        mapper_kind="tile",
        compositing_mode="additive",
        shader_weight_mode="us_focus",
        ray_attenuation_enabled=True,
        ray_attenuation_source="attenuation",
        ray_attenuation_beta=0.8,
        ray_attenuation_strength=0.55,
        ray_attenuation_min_transmission=0.35,
        ray_attenuation_weight_min=1e-4,
        ray_attenuation_quadrature_window_mm=2.0,
        ray_attenuation_quadrature_points=3,
        ray_attenuation_mapper_kind="ray_csr",
        ray_attenuation_formula="integral",
        ray_attenuation_candidate_k_sigma=2.0,
    )
    # Two identical pre-pixel Gaussians plus one behind every pixel in the image.
    means = torch.tensor(
        [[0.0, 2.0, 0.0], [0.0, 2.0, 0.0], [0.0, 20.0, 0.0]],
        device=device,
    )
    scales = torch.full((3, 3), 0.5, device=device)
    quats = torch.tensor([[1.0, 0.0, 0.0, 0.0]] * 3, device=device)
    pose = torch.eye(4, device=device)
    grid = build_slice_grid_pack(meta, device=device)
    return path_contribution_stats(
        cfg,
        means,
        scales,
        quats,
        attenuations.to(device=device),
        pose,
        grid,
    )


def test_no_prepixel_effect_has_zero_ag():
    path, support = _case(torch.tensor([0.1, 0.9, 0.1]))
    assert torch.isfinite(path).all()
    assert torch.isfinite(support).all()
    assert path[0] > 0
    assert support[0] > 0
    assert path[2].item() == pytest.approx(0.0, abs=1e-7)
    assert support[2].item() == pytest.approx(0.0, abs=1e-7)


def test_more_transmission_effect_yields_larger_ag():
    path, _ = _case(torch.tensor([0.1, 0.9, 0.1]))
    assert path[0] > path[1] > 0


def test_collection_flag_reuses_forward_csr_without_changing_render_or_gradients():
    device = torch.device("cuda")
    meta = SimpleNamespace(
        H=8,
        W=3,
        delta_lateral_mm=0.5,
        delta_axial_mm=1.0,
        near_mm=0.0,
        center_lateral=True,
        probe_type="linear",
    )
    cfg = OursRenderConfig(
        H=meta.H,
        W=meta.W,
        lateral_spacing_mm_per_pix=meta.delta_lateral_mm,
        axial_spacing_mm_per_pix=meta.delta_axial_mm,
        axial_near_mm=meta.near_mm,
        tile_size_px=2,
        elev_corridor_half_mm=1.0,
        k_sigma=2.0,
        mapper_kind="tile",
        compositing_mode="additive",
        shader_weight_mode="us_focus",
        ray_attenuation_enabled=True,
        ray_attenuation_source="attenuation",
        ray_attenuation_beta=0.8,
        ray_attenuation_strength=0.55,
        ray_attenuation_min_transmission=0.35,
        ray_attenuation_weight_min=1e-4,
        ray_attenuation_quadrature_window_mm=2.0,
        ray_attenuation_quadrature_points=3,
        ray_attenuation_mapper_kind="ray_csr",
        ray_attenuation_formula="integral",
        ray_attenuation_candidate_k_sigma=2.0,
    )
    means = torch.tensor([[0.0, 2.0, 0.0], [0.0, 2.0, 0.0], [0.0, 20.0, 0.0]], device=device)
    scales = torch.full((3, 3), 0.5, device=device)
    quats = torch.tensor([[1.0, 0.0, 0.0, 0.0]] * 3, device=device)
    attenuations = torch.tensor([[0.1], [0.9], [0.1]], device=device, requires_grad=True)
    opacities = torch.full((3, 1), 0.1, device=device, requires_grad=True)
    intensities = torch.ones((3, 1), device=device, requires_grad=True)
    pose = torch.eye(4, device=device)
    grid = build_slice_grid_pack(meta, device=device)
    renderer = OursRenderer(cfg).to(device)
    renderer.set_slice_grid_geometry(grid)

    image_off = renderer(means, scales, quats, opacities, intensities, pose, attenuations)
    image_on, path, support = renderer(
        means,
        scales,
        quats,
        opacities,
        intensities,
        pose,
        attenuations,
        collect_path_stats=True,
    )
    expected_path, expected_support = path_contribution_stats(
        cfg, means, scales, quats, attenuations.detach(), pose, grid
    )

    assert torch.equal(image_on, image_off)
    assert not path.requires_grad and not support.requires_grad
    assert torch.allclose(path, expected_path, rtol=1e-5, atol=1e-5)
    assert torch.equal(support, expected_support)
    grads_off = torch.autograd.grad(image_off.sum(), (opacities, intensities, attenuations), retain_graph=True)
    grads_on = torch.autograd.grad(image_on.sum(), (opacities, intensities, attenuations))
    for grad_off, grad_on in zip(grads_off, grads_on):
        assert torch.equal(grad_off, grad_on)
