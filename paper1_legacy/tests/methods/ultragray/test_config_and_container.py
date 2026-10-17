import torch

from configs.setup import load_hydra_config
from methods.ultragray.density_control import UltraGrayOfflineDensityController
from methods.ultragray.gaussian_container import UltraGrayGaussContainer
from methods.ultragray.gaussian_solver import UltraGrayGaussSolver
from methods.ultragray.render_geometry import pixel_positions_for_renderer
from scene.meta.dataset_meta import DatasetMeta


def test_ultragray_smoke_config_loads_expected_repro_settings():
    cfg = load_hydra_config("preproduction/baseline_reproduction/ultragray/smoke")

    assert cfg.model.name == "ultragray"
    assert cfg.engine.state == "UltraGrayOfflineState"
    assert cfg.scene.sampler.num_init == 10000
    assert cfg.scene.sampler.scales_init == [0.5, 0.5, 0.5]
    assert cfg.model.render.ops_max_offset_mm == 2.0
    assert cfg.model.render.ops_train_only is True
    assert cfg.model.render.quadrature_window_radius_mm == 2.0
    assert cfg.model.offline.sh_degree1_start_iter == 1000


def test_ultragray_repro_configs_load():
    for experiment in (
        "preproduction/baseline_reproduction/ultragray/smoke",
        "preproduction/baseline_reproduction/ultragray/main_10k_dc",
        "preproduction/baseline_reproduction/ultragray/main_10k_no_dc",
        "preproduction/baseline_reproduction/ultragray/main_10k_no_dc_fast",
        "preproduction/baseline_reproduction/ultragray/main_10k_fast",
        "preproduction/baseline_reproduction/ultragray/paper_30k",
    ):
        cfg = load_hydra_config(experiment)
        assert cfg.model.name == "ultragray"


def test_ultragray_pixel_positions_are_w_h_order_for_renderer():
    dataset_meta = DatasetMeta(
        name="toy",
        probe_type="linear",
        H=2,
        W=3,
        near_mm=0.0,
        far_mm=2.0,
        width_mm=3.0,
        radius_mm=0.0,
        fov_deg=0.0,
        delta_lateral_mm=1.0,
        delta_axial_mm=1.0,
        delta_elevational_mm=1.0,
        center_lateral=True,
        pose_init_type="probe->world",
    )
    grid_hw = torch.tensor(
        [
            [[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [2.0, 0.0, 0.0]],
            [[0.0, 1.0, 0.0], [1.0, 1.0, 0.0], [2.0, 1.0, 0.0]],
        ],
        dtype=torch.float32,
    )

    renderer_grid = pixel_positions_for_renderer(grid_hw.view(-1, 3), dataset_meta)

    assert renderer_grid.shape == (3, 2, 3)
    assert torch.allclose(renderer_grid[0, 0], torch.tensor([0.0, 0.0, 0.0]))
    assert torch.allclose(renderer_grid[1, 0], torch.tensor([1.0, 0.0, 0.0]))
    assert torch.allclose(renderer_grid[0, 1], torch.tensor([0.0, 1.0, 0.0]))


def test_ultragray_container_initializes_paper_parameters_on_cpu():
    cfg = load_hydra_config("preproduction/baseline_reproduction/ultragray/smoke")
    container = UltraGrayGaussContainer(cfg, torch.device("cpu"))
    means = torch.zeros((3, 3), dtype=torch.float32)

    container.init_from_scene({"means": means})

    assert container.num_points == 3
    assert container.means.shape == (3, 3)
    assert torch.allclose(container.scales, torch.full((3, 3), 0.5))
    assert torch.allclose(container.quats[:, 0], torch.ones(3))
    assert torch.allclose(container.attenuations, torch.full((3,), 0.99), atol=1e-6)
    assert container.sh_coeffs.shape == (3, 4)


def test_ultragray_container_reinjects_parameters_on_cpu():
    cfg = load_hydra_config("preproduction/baseline_reproduction/ultragray/smoke")
    container = UltraGrayGaussContainer(cfg, torch.device("cpu"))
    container.init_from_scene({"means": torch.zeros((3, 3), dtype=torch.float32)})

    container.init_from_scene({"means": torch.ones((2, 3), dtype=torch.float32)})

    assert container.num_points == 2
    assert set(container._parameters.keys()) == {"_means", "_scales", "_quats", "_attenuations", "_sh_dc", "_sh_rest"}
    assert torch.allclose(container.means, torch.ones((2, 3)))


def test_ultragray_density_control_prunes_degenerate_scale():
    cfg = load_hydra_config("preproduction/baseline_reproduction/ultragray/smoke")
    cfg.model.offline.density_control.enable.densify = False
    cfg.model.offline.density_control.enable.prune = True
    cfg.model.offline.density_control.timing.start_iter = 1
    cfg.model.offline.density_control.timing.end_iter = 10
    cfg.model.offline.density_control.timing.freq = 1

    container = UltraGrayGaussContainer(cfg, torch.device("cpu"))
    container.init_from_scene(
        {
            "means": torch.zeros((2, 3), dtype=torch.float32),
            "scales": torch.tensor([[1.0e-6, 0.5, 0.5], [0.5, 0.5, 0.5]], dtype=torch.float32),
        }
    )
    solver = UltraGrayGaussSolver(cfg, container)
    solver.setup()

    dc = UltraGrayOfflineDensityController(cfg)
    dc.prune_and_densify(1, solver, container)

    assert container.num_points == 1
    assert torch.allclose(container.scales, torch.full((1, 3), 0.5))
