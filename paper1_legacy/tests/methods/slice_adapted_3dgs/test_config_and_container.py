import torch

from configs.setup import load_hydra_config
from methods.slice_adapted_3dgs.gaussian_container import SliceAdapted3DGSContainer


def test_smoke_config_builds_slice_adapted_3dgs():
    cfg = load_hydra_config(
        "preproduction/baseline_reproduction/slice_adapted_3dgs/smoke",
        overrides=["task=dry_run", "system.device=cpu", "engine.iterations=1"],
    )
    assert cfg.model.name == "slice_adapted_3dgs"
    assert cfg.engine.state == "SliceAdapted3DGSOfflineState"
    assert "SliceAdapted3DGSOfflineTrainCallback" in cfg.engine.needed_callbacks
    assert cfg.model.render.backend == "cuda"


def test_container_init_from_project_scene_samples_cpu():
    cfg = load_hydra_config(
        "preproduction/baseline_reproduction/slice_adapted_3dgs/smoke",
        overrides=["task=dry_run", "system.device=cpu", "engine.iterations=1", "model.render.backend=torch_reference"],
    )
    container = SliceAdapted3DGSContainer(cfg, torch.device("cpu"))
    scene_info = {
        "means": torch.randn(7, 3),
        "intensities": torch.linspace(0.1, 0.7, 7).unsqueeze(-1),
        "scales": torch.full((7, 3), 0.3),
        "quats": torch.tensor([[1.0, 0.0, 0.0, 0.0]]).expand(7, 4).clone(),
    }
    container.init_from_scene(scene_info)

    assert container.num_points == 7
    assert container.means.shape == (7, 3)
    assert container.scales_3d.shape == (7, 3)
    assert container.quats.shape == (7, 4)
    assert container.opacities.shape == (7, 1)
    assert container.sh_coeffs.shape == (7, 1, 3)
    assert torch.isfinite(container.means).all()
    assert torch.isfinite(container.scales_3d).all()
    assert torch.isfinite(container.opacities).all()
    assert not torch.allclose(container.dc_colors, torch.zeros_like(container.dc_colors))


def test_container_opacity_cap_bounds_values_cpu():
    cfg = load_hydra_config(
        "preproduction/baseline_reproduction/slice_adapted_3dgs/smoke",
        overrides=[
            "task=dry_run",
            "system.device=cpu",
            "engine.iterations=1",
            "model.render.backend=torch_reference",
            "model.gauss_container.constraints.opacity_max=0.2",
            "model.gauss_container.init_val.opacity=0.1",
        ],
    )
    container = SliceAdapted3DGSContainer(cfg, torch.device("cpu"))
    container.init_from_scene({"means": torch.randn(5, 3)})

    assert torch.all(container.opacities <= 0.2 + 1.0e-6)
    assert torch.allclose(container.opacities, torch.full((5, 1), 0.1), atol=1.0e-6)
