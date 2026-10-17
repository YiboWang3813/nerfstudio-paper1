import torch

from methods.gaussianpile.configs import ModelConfig
from methods.gaussianpile.gaussian_container import GaussianPileGaussContainer


class _Cfg:
    model = ModelConfig()


def test_container_init_from_scene_shapes():
    device = torch.device("cpu")
    container = GaussianPileGaussContainer(_Cfg(), device=device)
    scene_info = {
        "means": torch.zeros((4, 3), device=device),
        "scales": torch.full((4, 3), 0.2, device=device),
        "quats": torch.tensor([[1.0, 0.0, 0.0, 0.0]] * 4, device=device),
    }

    container.init_from_scene(scene_info)

    assert container.num_points == 4
    assert container.means.shape == (4, 3)
    assert container.scales.shape == (4, 3)
    assert container.quats.shape == (4, 4)
    assert container.opacities.shape == (4,)
    assert container.alphas.shape == (4,)
    assert "intensities" not in container._param_dict
    assert torch.all(container.alphas > 0.0)
    assert torch.all(container.alphas < _Cfg.model.gauss_container.constraints.opacity_max)


def test_container_load_reinjects_registered_parameters(tmp_path):
    device = torch.device("cpu")
    scene_info = {
        "means": torch.zeros((4, 3), device=device),
        "scales": torch.full((4, 3), 0.2, device=device),
        "quats": torch.tensor([[1.0, 0.0, 0.0, 0.0]] * 4, device=device),
    }
    src = GaussianPileGaussContainer(_Cfg(), device=device)
    src.init_from_scene(scene_info)
    path = tmp_path / "container.pth"
    src.save(str(path))

    dst = GaussianPileGaussContainer(_Cfg(), device=device)
    dst.init_from_scene(
        {
            "means": torch.ones((2, 3), device=device),
            "scales": torch.full((2, 3), 0.5, device=device),
            "quats": torch.tensor([[1.0, 0.0, 0.0, 0.0]] * 2, device=device),
        }
    )
    dst.load(str(path))

    assert dst.num_points == 4
    assert set(dst._parameters.keys()) == {"_means", "_scales", "_quats", "_opacities"}
    assert torch.allclose(dst.means, src.means)
