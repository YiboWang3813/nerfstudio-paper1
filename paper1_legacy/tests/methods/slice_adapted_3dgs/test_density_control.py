import torch

from configs.setup import load_hydra_config
from methods.slice_adapted_3dgs.density_control import SliceAdapted3DGSDensityController
from methods.slice_adapted_3dgs.gaussian_container import SliceAdapted3DGSContainer
from methods.slice_adapted_3dgs.gaussian_solver import SliceAdapted3DGSSolver


def _build_container_solver(n: int = 3):
    cfg = load_hydra_config(
        "preproduction/baseline_reproduction/slice_adapted_3dgs/smoke",
        overrides=[
            "task=dry_run",
            "system.device=cpu",
            "engine.iterations=10",
            "model.render.backend=torch_reference",
            "model.offline.density_control.timing.start_iter=1",
            "model.offline.density_control.timing.end_iter=10",
            "model.offline.density_control.timing.freq=1",
            "model.offline.density_control.n_max=0",
        ],
    )
    device = torch.device("cpu")
    container = SliceAdapted3DGSContainer(cfg, device)
    container.init_from_scene(
        {
            "means": torch.arange(n, dtype=torch.float32).view(n, 1).expand(n, 3).clone(),
            "intensities": torch.full((n, 1), 0.5),
            "scales": torch.full((n, 3), 0.8),
            "quats": torch.tensor([[1.0, 0.0, 0.0, 0.0]]).expand(n, 4).clone(),
        }
    )
    solver = SliceAdapted3DGSSolver(cfg, container)
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


def test_density_control_split_prunes_original_and_syncs_optimizer_state_cpu():
    cfg, container, solver = _build_container_solver(n=3)
    cfg.model.offline.density_control.enable.prune = False
    cfg.model.offline.density_control.thresholds.densify_grad = 1.0e-5
    cfg.model.offline.density_control.thresholds.split_scale_mm = 0.2
    dc = SliceAdapted3DGSDensityController(cfg)
    dc.xyz_grad_accum = torch.tensor([[1.0], [0.0], [0.0]])
    dc.denom = torch.ones((3, 1))

    dc.prune_and_densify(1, solver, container)

    assert container.num_points == 4
    assert torch.allclose(container.means[:1], torch.tensor([[1.0, 1.0, 1.0]]), atol=1.0e-3)
    _assert_optimizer_state_matches_container(container, solver)


def test_density_control_prune_by_opacity_keeps_at_least_one_point_cpu():
    cfg, container, solver = _build_container_solver(n=4)
    cfg.model.offline.density_control.enable.densify = False
    cfg.model.offline.density_control.thresholds.prune_opacity = 0.2
    dc = SliceAdapted3DGSDensityController(cfg)
    container._param_dict["opacities"].tensor.data.copy_(
        container._specs["opacities"].inverse_fn(torch.tensor([[0.01], [0.01], [0.01], [0.03]]))
    )

    dc.prune_and_densify(1, solver, container)

    assert container.num_points == 1
    assert torch.allclose(container.opacities, torch.tensor([[0.03]]), atol=1.0e-6)
    _assert_optimizer_state_matches_container(container, solver)


def test_density_control_reset_opacity_replaces_optimizer_state_cpu():
    cfg, container, solver = _build_container_solver(n=2)
    dc = SliceAdapted3DGSDensityController(cfg)
    container._param_dict["opacities"].tensor.data.copy_(
        container._specs["opacities"].inverse_fn(torch.tensor([[0.5], [0.005]]))
    )

    dc.reset_opacity(solver, container)

    assert torch.allclose(container.opacities, torch.tensor([[0.01], [0.005]]), atol=1.0e-6)
    _assert_optimizer_state_matches_container(container, solver)
