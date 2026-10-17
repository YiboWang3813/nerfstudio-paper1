"""P0 policy isolation and actual topology regression against the pre-P0 snapshot."""
import importlib.util
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

from methods.ours.configs import ModelConfig
from methods.ours.density_control import OursOfflineDensityController
from methods.ours.gaussian_container import OursGaussContainer
from methods.ours.gaussian_solver import OursGaussSolver

ROOT = Path(__file__).resolve().parents[3]
PRE = ROOT / "outputs/preproduction/regression/p0_semantic_freeze_20260908/pre/tree"


def config(densify="residual_guided", prune="path_aware", global_on=True):
    cfg = SimpleNamespace(model=ModelConfig(), engine=SimpleNamespace(stream="offline", iterations=30000))
    sr = cfg.model.slice_render
    sr.ray_attenuation_enabled = global_on
    sr.ray_attenuation_beta, sr.ray_attenuation_strength = .8, .55
    dc = cfg.model.offline.density_control
    dc.densification_policy, dc.pruning_policy = densify, prune
    dc.strategy = "gradient" if densify == "gradient" else "contribution_residual"
    dc.responsibility.enabled = densify != "gradient"
    dc.responsibility.protect_attenuation_mode = "path_contribution"
    dc.responsibility.protect_attenuation_enabled = prune == "path_aware"
    dc.enable.prune = prune != "none"
    dc.thresholds.prune_opacity = 1e-4
    dc.responsibility.low_contribution_quantile = .1
    dc.timing.start_iter, dc.timing.end_iter, dc.timing.freq = 1000, 25000, 2000
    cfg.model.gauss_container.constraints.scale_activation = "bounded_sigmoid"
    return cfg


def event(cls, cfg):
    torch.manual_seed(42)
    c = OursGaussContainer(cfg, torch.device("cpu"))
    n = 20
    c.init_from_scene(dict(means=torch.arange(n * 3).reshape(n, 3).float(),
        scales=torch.linspace(.3, 1.2, n)[:, None].repeat(1, 3),
        quats=torch.tensor([[1., 0., 0., 0.]]).repeat(n, 1)))
    with torch.no_grad():
        c._param_dict["echo_alphas"].tensor[:6] = -15.
    solver = OursGaussSolver(cfg, c)
    solver.setup()
    dc = cls(cfg)
    dc.xyz_grad_accum = torch.ones(n, 1) * 1e-4
    dc.denom = torch.ones(n, 1)
    contribution = torch.cat([torch.zeros(6), torch.arange(1, 15).float()])
    dc.update_responsibility_stats(c, contribution, contribution * .1,
                                   torch.ones(n), torch.ones(n))
    path = torch.arange(n).float()
    path[0] = 100.
    dc.update_path_stats(c, path, torch.ones(n))
    dc.prune_and_densify(2000, solver, c)
    return {k: p.tensor.detach().clone() for k, p in c._param_dict.items()}, dc.last_event_stats


@pytest.mark.parametrize("densify,prune", [("gradient", "opacity"),
    ("residual_guided", "none"), ("residual_guided", "local_aware"),
    ("residual_guided", "path_aware")])
@pytest.mark.parametrize("global_on", [False, True])
def test_pre_post_topology_identical(densify, prune, global_on):
    if not PRE.exists():
        pytest.skip("Historical snapshot required for pre/post regression")
    spec = importlib.util.spec_from_file_location("methods.ours._p0_legacy_dc", PRE / "methods/ours/density_control.py")
    old = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(old)
    cfg = config(densify, prune, global_on)
    legacy = config(densify, prune, global_on)
    # Stage-2 explicitly disabled global protection in no-LGR+CGA rows.
    if not global_on:
        legacy.model.offline.density_control.responsibility.protect_attenuation_enabled = False
    a, sa = event(old.OursOfflineDensityController, legacy)
    b, sb = event(OursOfflineDensityController, cfg)
    assert a.keys() == b.keys()
    for key in a:
        assert torch.equal(a[key], b[key]), key
    for key in ("clone", "split", "prune_weak", "num_gaussians_after", "path_protected_candidate_count"):
        assert sa[key] == sb[key], key


def test_opacity_is_independent_of_residual_densification():
    _, op = event(OursOfflineDensityController, config(prune="opacity"))
    _, lp = event(OursOfflineDensityController, config(prune="local_aware"))
    _, pa = event(OursOfflineDensityController, config(prune="path_aware"))
    assert op["prune_weak"] >= lp["prune_weak"] > pa["prune_weak"]
    assert op["responsibility_selected"] == lp["responsibility_selected"] == pa["responsibility_selected"]


def test_global_off_falls_back_without_all_protected_artifact():
    cfg = config(global_on=False)
    dc = OursOfflineDensityController(cfg)
    assert dc.pruning_policy == "local_aware"
    assert not dc.should_collect_path_stats(2000)
    _, stats = event(OursOfflineDensityController, cfg)
    assert stats["prune_weak"] > 0
    assert stats["path_protected_candidate_count"] == 0
    assert OursOfflineDensityController(config()).should_collect_path_stats(2000)


def test_ppp_geometry_independent_and_partial_oracles():
    from scene.container.single_sweep_container import SingleSweepContainer
    obj = SingleSweepContainer.__new__(SingleSweepContainer)
    loading = SimpleNamespace(canonical_sweep_spacing_mode="uniform", canonical_sweep_axis="z",
        canonical_sweep_delta_mm=-1., canonical_sweep_delta_scale=1.1, canonical_sweep_ref_frame=1)
    obj.cfg = SimpleNamespace(loading=loading)
    obj.meta = SimpleNamespace(delta_elevational_mm=.404)
    poses = torch.eye(4).repeat(3, 1, 1)
    poses[:, :3, 3] = torch.tensor([[0., 0., 0.], [3., 0., 4.], [3., 0., 2.]])
    arbitrary = torch.randn_like(poses)
    a = obj._make_canonical_sweep_poses(poses)
    b = obj._make_canonical_sweep_poses(arbitrary)
    assert torch.equal(a, b)
    assert torch.allclose(a[:, 2, 3], torch.tensor([-1., 0., 1.]) * .404 * 1.1)
    loading.canonical_sweep_spacing_mode = "file_step_norm"
    assert torch.allclose(obj._make_canonical_sweep_offsets(poses), torch.tensor([-5., 0., 2.]) * 1.1)
    loading.canonical_sweep_spacing_mode = "file_axis_projection"
    assert torch.allclose(obj._make_canonical_sweep_offsets(poses), torch.tensor([-4., 0., -2.]) * 1.1)
