import torch

from configs.setup import load_hydra_config
from methods.ultrags_baseline.density_control import UltraGSBaselineDensityController
from methods.ultrags_baseline.gaussian_container import UltraGSBaselineContainer
from methods.ultrags_baseline.gaussian_solver import UltraGSBaselineSolver


def test_density_control_uses_freq_and_inclusive_end_cpu():
    cfg = load_hydra_config(
        "preproduction/baseline_reproduction/ultrags_baseline/smoke",
        overrides=[
            "task=dry_run",
            "system.device=cpu",
            "model.offline.density_control.enable.densify=true",
            "model.offline.density_control.enable.prune=false",
            "model.offline.density_control.timing.start_iter=1000",
            "model.offline.density_control.timing.end_iter=25000",
            "model.offline.density_control.timing.freq=2000",
            "model.offline.density_control.timing.densification_interval=1500",
        ],
    )

    controller = UltraGSBaselineDensityController(cfg)

    assert not controller.should_densify_prune(1500)
    assert controller.should_densify_prune(2000)
    assert controller.should_accumulate_grad_stats(25000)
    assert not controller.should_accumulate_grad_stats(25001)


def test_split_densification_prunes_original_point_cpu():
    cfg = load_hydra_config(
        "preproduction/baseline_reproduction/ultrags_baseline/smoke",
        overrides=[
            "task=dry_run",
            "system.device=cpu",
            "engine.iterations=10",
            "model.offline.density_control.enable.densify=true",
            "model.offline.density_control.enable.prune=false",
            "model.offline.density_control.thresholds.n_max=0",
            "model.offline.density_control.thresholds.percent_dense=0.01",
        ],
    )
    device = torch.device("cpu")
    container = UltraGSBaselineContainer(cfg, device)
    container.init_from_scene(
        {
            "means": torch.tensor(
                [
                    [0.0, 0.0, 0.0],
                    [1.0, 0.0, 0.0],
                    [2.0, 0.0, 0.0],
                ],
                dtype=torch.float32,
            ),
            "intensities": torch.full((3, 1), 0.5),
            "scales": torch.full((3, 3), 0.3),
            "quats": torch.tensor([[1.0, 0.0, 0.0, 0.0]]).expand(3, 4).clone(),
        }
    )
    solver = UltraGSBaselineSolver(cfg, container)
    solver.setup()

    controller = UltraGSBaselineDensityController(cfg)
    controller.set_scene_radius(1.0)
    controller.xyz_grad_accum = torch.tensor([[1.0], [0.0], [0.0]], device=device)
    controller.denom = torch.ones((3, 1), device=device)

    controller.prune_and_densify(1, solver, container)

    assert container.num_points == 4
    assert torch.allclose(container.means[:1], torch.tensor([[1.0, 0.0, 0.0]]))
