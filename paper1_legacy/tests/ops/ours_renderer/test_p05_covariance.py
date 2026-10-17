from dataclasses import replace
from types import SimpleNamespace

import pytest
import torch
from ours_renderer import _C
from ours_renderer.autograd import render_projected_slice
from ours_renderer.grid_geometry import build_slice_grid_pack
from ours_renderer.us_focus_reference import render_us_focus_reference
from ours_renderer.us_focus_reference import _quat_to_rotmat_wxyz as _quat_to_rot
from scripts.production.ablations.ours.p0_renderer_probe import sample


@pytest.mark.parametrize("identity", [False, True])
@pytest.mark.parametrize("zero_scale", [False, True])
def test_covariance_symmetry_spectrum(identity, zero_scale):
    torch.manual_seed(42)
    q = torch.nn.functional.normalize(torch.randn(128, 4), dim=-1)
    if identity:
        q[:] = torch.tensor([1., 0., 0., 0.])
    s = .1 + 3 * torch.rand(128, 3)
    if zero_scale:
        s[:, 0] = 0
    R = _quat_to_rot(q)
    expected = R @ torch.diag_embed(s.square()) @ R.transpose(-1, -2)
    assert torch.allclose(expected, expected.transpose(-1, -2), atol=2e-6)
    if not torch.cuda.is_available():
        actual = expected
    else:
        actual = _C.CovarianceDiagnosticCUDA(s.cuda(), q.cuda()).cpu()
        print("COVARIANCE_MAX_ABS", (actual - expected).abs().max().item())
        assert torch.allclose(actual, expected, atol=5e-6, rtol=2e-5)
    assert torch.allclose(actual, actual.transpose(-1, -2), atol=2e-6)
    eig = torch.linalg.eigvalsh(actual)
    assert eig.min() >= (-3e-6 if zero_scale else .009)
    assert torch.allclose(eig, s.square().sort(dim=-1).values, atol=8e-6, rtol=2e-5)
    if identity:
        assert torch.allclose(actual, torch.diag_embed(s.square()), atol=1e-6)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="compute node required")
@pytest.mark.parametrize("mode", ["none", "fixed", "adaptive"])
@pytest.mark.parametrize("seed", [7, 42, 173])
def test_random_rotation_forward_backward_reference(mode, seed):
    torch.manual_seed(seed)
    cfg, m, s, q, a, intensity, _, pose = sample("cuda", mode)
    with torch.no_grad():
        q.copy_(torch.nn.functional.normalize(torch.randn_like(q), dim=-1))
        s.copy_(.4 + torch.rand_like(s))
    cfg = replace(cfg, k_sigma=12., ray_attenuation_enabled=False)
    grid = build_slice_grid_pack(SimpleNamespace(H=16, W=16, delta_lateral_mm=.5,
        delta_axial_mm=.5, near_mm=0., center_lateral=True), device=m.device)
    pred = render_projected_slice(cfg, m, s, q, a, intensity, pose, grid)
    ref = render_us_focus_reference(cfg, m, s, q, a, intensity, pose)
    print("RANDOM_FORWARD_MAX_ABS", mode, seed, (pred-ref).abs().max().item())
    assert torch.allclose(pred, ref, atol=2e-5, rtol=2e-4)
    variables = (m, s, q, a)
    actual = torch.autograd.grad(pred.square().sum(), variables)
    expected = torch.autograd.grad(ref.square().sum(), variables)
    for x, y in zip(actual, expected):
        assert torch.isfinite(x).all()
        print("RANDOM_GRAD_MAX_ABS", mode, seed, (x-y).abs().max().item())
        assert torch.allclose(x, y, atol=2e-4, rtol=.02)
