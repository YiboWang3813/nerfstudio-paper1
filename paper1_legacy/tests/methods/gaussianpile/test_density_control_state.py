from types import SimpleNamespace

import torch

from methods.gaussianpile.configs import ModelConfig
from methods.gaussianpile.density_control import GaussianPileOfflineDensityController
from methods.gaussianpile.gaussian_container import GaussianPileGaussContainer
from methods.gaussianpile.gaussian_solver import GaussianPileGaussSolver


def _cfg():
    return SimpleNamespace(model=ModelConfig(), engine=SimpleNamespace(iterations=1000))


def _container_and_solver(n=4):
    cfg = _cfg()
    device = torch.device("cpu")
    container = GaussianPileGaussContainer(cfg, device=device)
    container.init_from_scene(
        {
            "means": torch.zeros((n, 3), device=device),
            "scales": torch.full((n, 3), 0.2, device=device),
            "quats": torch.tensor([[1.0, 0.0, 0.0, 0.0]] * n, device=device),
        }
    )
    solver = GaussianPileGaussSolver(cfg, container)
    solver.setup()
    for wrapper in container._param_dict.values():
        wrapper.tensor.grad = torch.ones_like(wrapper.tensor)
    solver.step()
    solver.zero_grad()
    return cfg, container, solver


def _assert_optimizer_state_matches_container(container, solver):
    assert container.num_points > 0
    for group in solver.optimizer.param_groups:
        name = group["name"]
        param = group["params"][0]
        assert container._param_dict[name].tensor is param
        assert param.shape[0] == container.num_points
        state = solver.optimizer.state.get(param, {})
        for key in ("exp_avg", "exp_avg_sq"):
            if key in state:
                assert state[key].shape == param.shape


def test_density_control_clone_keeps_optimizer_state_lengths_in_sync():
    cfg, container, solver = _container_and_solver(n=3)
    cfg.model.offline.density_control.enable.prune = False
    cfg.model.offline.density_control.thresholds.densify_grad = 1.0e-5
    cfg.model.offline.density_control.n_max = 16
    dc = GaussianPileOfflineDensityController(cfg)
    n = container.num_points
    dc.xyz_grad_accum = torch.full((n, 1), 1.0e-3)
    dc.focus_response_accum = torch.ones((n, 1))
    dc.tile_radius_accum = torch.zeros((n, 1))
    dc.denom = torch.ones((n, 1))

    dc.prune_and_densify(500, solver, container)

    assert container.num_points == n * 2
    _assert_optimizer_state_matches_container(container, solver)


def test_density_control_refreshes_point_count_when_pruning_after_densify():
    cfg, container, solver = _container_and_solver(n=3)
    cfg.model.offline.density_control.thresholds.densify_grad = 1.0e-5
    cfg.model.offline.density_control.n_max = 16
    dc = GaussianPileOfflineDensityController(cfg)
    n = container.num_points
    dc.xyz_grad_accum = torch.full((n, 1), 1.0e-3)
    dc.focus_response_accum = torch.ones((n, 1))
    dc.tile_radius_accum = torch.zeros((n, 1))
    dc.prune_observation_accum = torch.full((n, 1), 2.0)
    dc.denom = torch.ones((n, 1))

    dc.prune_and_densify(500, solver, container)

    assert container.num_points == n * 2
    _assert_optimizer_state_matches_container(container, solver)


def test_density_control_prune_keeps_optimizer_state_lengths_in_sync():
    cfg, container, solver = _container_and_solver(n=4)
    cfg.model.offline.density_control.enable.densify = False
    cfg.model.offline.density_control.thresholds.min_keep_ratio = 0.0
    dc = GaussianPileOfflineDensityController(cfg)
    alpha_values = torch.tensor([0.005, 0.10, 0.20, 0.30])
    container._param_dict["opacities"].tensor.data.copy_(
        container._specs["opacities"].inverse_fn(alpha_values)
    )

    dc.prune_and_densify(500, solver, container)

    assert container.num_points == 3
    assert torch.all(container.alphas > cfg.model.offline.density_control.thresholds.prune_opacity)
    _assert_optimizer_state_matches_container(container, solver)


def test_density_control_prune_uses_focus_footprint_opacity_score():
    cfg, container, solver = _container_and_solver(n=4)
    cfg.model.offline.density_control.enable.densify = False
    cfg.model.offline.density_control.thresholds.prune_opacity = 0.02
    cfg.model.offline.density_control.thresholds.max_prune_fraction_per_step = 0.0
    cfg.model.offline.density_control.thresholds.min_keep_ratio = 0.0
    dc = GaussianPileOfflineDensityController(cfg)
    alpha_values = torch.tensor([0.015, 0.015, 0.030, 0.001])
    container._param_dict["opacities"].tensor.data.copy_(
        container._specs["opacities"].inverse_fn(alpha_values)
    )
    dc.focus_response_accum = torch.tensor([[1.0], [1.0], [1.0], [0.1]])
    dc.tile_radius_accum = torch.tensor([[0.0], [0.25], [0.0], [0.25]])
    dc.prune_observation_accum = torch.full((4, 1), 2.0)
    dc.denom = torch.ones((4, 1))

    dc.prune_and_densify(500, solver, container)

    assert container.num_points == 2
    assert torch.allclose(container.alphas, torch.tensor([0.015, 0.030]))
    _assert_optimizer_state_matches_container(container, solver)


def test_density_control_prune_keeps_weak_points_without_enough_observations():
    cfg, container, solver = _container_and_solver(n=4)
    cfg.model.offline.density_control.enable.densify = False
    cfg.model.offline.density_control.thresholds.prune_opacity = 0.02
    cfg.model.offline.density_control.thresholds.min_prune_observations = 2
    cfg.model.offline.density_control.thresholds.max_prune_fraction_per_step = 0.0
    cfg.model.offline.density_control.thresholds.min_keep_ratio = 0.0
    dc = GaussianPileOfflineDensityController(cfg)
    alpha_values = torch.tensor([0.005, 0.006, 0.20, 0.30])
    container._param_dict["opacities"].tensor.data.copy_(
        container._specs["opacities"].inverse_fn(alpha_values)
    )
    dc.focus_response_accum = torch.ones((4, 1))
    dc.tile_radius_accum = torch.ones((4, 1))
    dc.prune_observation_accum = torch.tensor([[0.0], [1.0], [2.0], [2.0]])
    dc.denom = torch.ones((4, 1))

    dc.prune_and_densify(500, solver, container)

    assert container.num_points == 4
    assert torch.allclose(container.alphas, alpha_values)
    _assert_optimizer_state_matches_container(container, solver)


def test_density_control_min_keep_guard_limits_aggressive_pruning():
    cfg, container, solver = _container_and_solver(n=6)
    cfg.model.gauss_init.num_points = 6
    cfg.model.offline.density_control.enable.densify = False
    cfg.model.offline.density_control.thresholds.prune_opacity = 0.1
    cfg.model.offline.density_control.thresholds.max_prune_fraction_per_step = 0.0
    cfg.model.offline.density_control.thresholds.min_keep_ratio = 0.5
    dc = GaussianPileOfflineDensityController(cfg)
    alpha_values = torch.tensor([0.001, 0.002, 0.003, 0.004, 0.50, 0.60])
    container._param_dict["opacities"].tensor.data.copy_(
        container._specs["opacities"].inverse_fn(alpha_values)
    )

    dc.prune_and_densify(500, solver, container)

    assert container.num_points == 3
    assert int((container.alphas < cfg.model.offline.density_control.thresholds.prune_opacity).sum()) == 1
    _assert_optimizer_state_matches_container(container, solver)


def test_density_control_prune_fraction_limit_removes_lowest_alpha_candidates_first():
    cfg, container, solver = _container_and_solver(n=10)
    cfg.model.offline.density_control.enable.densify = False
    cfg.model.offline.density_control.thresholds.prune_opacity = 0.1
    cfg.model.offline.density_control.thresholds.max_prune_fraction_per_step = 0.3
    cfg.model.offline.density_control.thresholds.min_keep_ratio = 0.0
    dc = GaussianPileOfflineDensityController(cfg)
    alpha_values = torch.tensor([0.001, 0.002, 0.003, 0.004, 0.005, 0.006, 0.50, 0.60, 0.70, 0.80])
    container._param_dict["opacities"].tensor.data.copy_(
        container._specs["opacities"].inverse_fn(alpha_values)
    )

    dc.prune_and_densify(500, solver, container)

    assert container.num_points == 7
    assert torch.allclose(container.alphas[:3], torch.tensor([0.004, 0.005, 0.006]))
    _assert_optimizer_state_matches_container(container, solver)
