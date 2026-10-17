"""Same renderer family, three local conditions, independent reference/gradient checks."""
import importlib.util
from dataclasses import replace
from pathlib import Path

import pytest
import torch
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[3] / "ops/ours_renderer"))

from ours_renderer.autograd import _us_focus_params_vector, render_projected_slice
from ours_renderer.grid_geometry import build_slice_grid_pack
from ours_renderer.us_focus_reference import us_focus_gaussian_footprint_reference, render_us_focus_reference
from scripts.production.ablations.ours.p0_renderer_probe import sample
from types import SimpleNamespace

ROOT = Path(__file__).resolve().parents[3]
PRE = ROOT / "outputs/preproduction/regression/p0_semantic_freeze_20260908/pre/tree"


def test_default_reference_identical_to_pre_p0():
    if not PRE.exists():
        pytest.skip("pre-P0 snapshot unavailable")
    spec = importlib.util.spec_from_file_location("ours_renderer._p0_reference", PRE / "ops/ours_renderer/ours_renderer/us_focus_reference.py")
    old = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(old)
    cfg, m, s, q, a, i, _, pose = sample("cpu")
    before = old.render_us_focus_reference(cfg, m, s, q, a, i, pose)
    after = render_us_focus_reference(cfg, m, s, q, a, i, pose)
    assert torch.equal(before, after)
    assert torch.equal(old.us_focus_params_from_config(cfg, device=m.device), _us_focus_params_vector(cfg, m.device)[:14])


def test_fixed_is_exactly_zero_depth_slope():
    cfg, m, s, q, a, i, _, pose = sample("cpu")
    fixed = replace(cfg, local_conditioning_mode="fixed")
    zero_slope = replace(cfg, us_focus_sigma_ele_1=0.)
    assert torch.equal(render_us_focus_reference(fixed, m, s, q, a, i, pose),
                       render_us_focus_reference(zero_slope, m, s, q, a, i, pose))
    terms = us_focus_gaussian_footprint_reference(fixed, m, s, q, pose)
    assert torch.equal(terms["sigma_ele"], torch.ones_like(terms["sigma_ele"]))


def test_none_is_unconditioned_marginal_not_just_eta_one():
    cfg, m, s, q, _, _, _, pose = sample("cpu")
    none = us_focus_gaussian_footprint_reference(replace(cfg, local_conditioning_mode="none"), m, s, q, pose)
    adaptive = us_focus_gaussian_footprint_reference(cfg, m, s, q, pose)
    assert torch.equal(none["sigma_e"], none["sigma_c"])
    assert torch.equal(none["mu2"], none["mu_c"][:, :2])
    assert torch.equal(none["slice_response"], torch.ones_like(none["slice_response"]))
    assert not torch.allclose(none["sigma2"], adaptive["sigma2"])
    assert not torch.allclose(none["mu2"], adaptive["mu2"])


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires GPU compute node")
@pytest.mark.parametrize("mode", ["none", "fixed", "adaptive"])
@pytest.mark.parametrize("rotated", [False, True])
def test_cuda_modes_reference_and_gradients(mode, rotated):
    cfg, m, s, q, a, i, _, pose = sample("cuda", mode)
    if not rotated:
        with torch.no_grad():
            q[:] = torch.tensor([1., 0., 0., 0.], device=q.device)
    cfg = replace(cfg, k_sigma=8., ray_attenuation_enabled=False)
    grid = build_slice_grid_pack(SimpleNamespace(H=16, W=16, delta_lateral_mm=.5,
        delta_axial_mm=.5, near_mm=0., center_lateral=True), device=m.device)
    def render():
        return render_projected_slice(cfg, m, s, q, a, i, pose, grid)
    pred = render()
    reference = render_us_focus_reference(cfg, m, s, q, a, i, pose)
    print("CUDA_REFERENCE_MAX_ABS", mode, rotated, (pred-reference).abs().max().item())
    assert torch.allclose(pred, reference, atol=2e-5, rtol=2e-4)
    pred.square().sum().backward()
    for x in (m, s, q, a, pose):
        assert x.grad is not None and torch.isfinite(x.grad).all()
    # Direct finite differences exercise shape and position gradients, not just alpha.
    for x, index in ((m, (0, 0)), (m, (0, 2)), (s, (0, 0)), (q, (0, 1)), (a, (0,))):
        expected = x.grad[index].item()
        eps = 1e-3
        with torch.no_grad():
            original = x[index].item()
            x[index] = original + eps
            plus = render().square().sum().item()
            x[index] = original - eps
            minus = render().square().sum().item()
            x[index] = original
        fd = (plus - minus) / (2 * eps)
        assert abs(expected - fd) <= 2e-3 + .08 * abs(fd), (mode, index, expected, fd)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires GPU compute node")
@pytest.mark.parametrize("zero_parameter", ["ray_attenuation_beta", "ray_attenuation_strength"])
def test_zero_global_parameter_matches_local_only(zero_parameter):
    results = []
    for enabled in (False, True):
        cfg, m, s, q, a, i, tau, pose = sample("cuda")
        cfg = replace(cfg, ray_attenuation_enabled=enabled, **{zero_parameter: 0.})
        grid = build_slice_grid_pack(SimpleNamespace(H=16, W=16, delta_lateral_mm=.5,
            delta_axial_mm=.5, near_mm=0., center_lateral=True), device=m.device)
        pred = render_projected_slice(cfg, m, s, q, a, i, pose, grid, tau)
        pred.square().sum().backward()
        results.append([pred.detach(), m.grad, s.grad, q.grad, a.grad, tau.grad, pose.grad])
    assert torch.equal(results[0][0], results[1][0])
    for local, zero in zip(*results):
        # CUDA backward reduces with atomics, so operation order need not be bitwise stable.
        print("zero-global equivalence", zero_parameter, (local - zero).abs().max().item())
        # Legacy shape gradients use finite differences; reduction roundoff is
        # amplified by their 1e-3 step. Observed pre/pre spread is ~1.5e-5.
        assert torch.allclose(local, zero, atol=3e-5, rtol=1e-5)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires GPU compute node")
def test_rotated_gaussian_mathematical_reference_corrected():
    cfg, m, s, q, a, i, _, pose = sample("cuda")
    cfg = replace(cfg, ray_attenuation_enabled=False, k_sigma=8.)
    grid = build_slice_grid_pack(SimpleNamespace(H=16, W=16, delta_lateral_mm=.5,
        delta_axial_mm=.5, near_mm=0., center_lateral=True), device=m.device)
    actual = render_projected_slice(cfg, m, s, q, a, i, pose, grid)
    expected = render_us_focus_reference(cfg, m, s, q, a, i, pose)
    assert torch.allclose(actual, expected, atol=2e-5, rtol=2e-4)
