import sys
from pathlib import Path

import torch

OPS_PATH = Path(__file__).resolve().parents[3] / "ops" / "gaussianplie_renderer"
if str(OPS_PATH) not in sys.path:
    sys.path.insert(0, str(OPS_PATH))

from gaussianplie_renderer.reference import build_covariance, focus_project, world_to_camera


def test_identity_world_to_camera_keeps_mean_and_covariance():
    means = torch.tensor([[1.0, 2.0, 3.0]], dtype=torch.float64)
    scales = torch.tensor([[0.5, 0.6, 0.7]], dtype=torch.float64)
    quats = torch.tensor([[1.0, 0.0, 0.0, 0.0]], dtype=torch.float64)
    covs = build_covariance(scales, quats, cov_eps=1.0e-9)
    pose = torch.eye(4, dtype=torch.float64)

    means_c, covs_c = world_to_camera(means, covs, pose)

    assert torch.allclose(means_c, means)
    assert torch.allclose(covs_c, covs)


def test_focus_large_sigma_approaches_original_gaussian():
    means_c = torch.tensor([[0.1, -0.2, 0.3]], dtype=torch.float64)
    covs_c = torch.diag_embed(torch.tensor([[0.4, 0.5, 0.6]], dtype=torch.float64))
    opacities = torch.tensor([0.7], dtype=torch.float64)

    means_2d, covs_2d, alpha = focus_project(
        means_c,
        covs_c,
        opacities,
        sigma_z=1.0e6,
        det_eps=1.0e-12,
        cov_eps=1.0e-12,
    )

    assert torch.allclose(means_2d, means_c[:, :2], atol=1.0e-6)
    assert torch.allclose(covs_2d, covs_c[:, :2, :2], atol=1.0e-6)
    expected_alpha = opacities / torch.sqrt(torch.det(covs_c[:, :2, :2]))
    assert torch.allclose(alpha, expected_alpha, rtol=1.0e-6, atol=1.0e-6)


def test_thin_slice_suppresses_far_gaussian_more():
    covs_c = torch.eye(3, dtype=torch.float64).reshape(1, 3, 3).repeat(2, 1, 1)
    means_c = torch.tensor([[0.0, 0.0, 0.1], [0.0, 0.0, 2.0]], dtype=torch.float64)
    opacities = torch.ones(2, dtype=torch.float64)

    _, _, alpha = focus_project(means_c, covs_c, opacities, sigma_z=0.2)

    assert alpha[0] > alpha[1]
