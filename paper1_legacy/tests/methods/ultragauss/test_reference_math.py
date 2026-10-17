import math
import sys
from pathlib import Path

import torch

sys.path.append(str(Path(__file__).resolve().parents[3]))

from methods.ultragauss.density_control import _approx_scale_from_precision_tril
from methods.ultragauss.gaussian_container import _precision_tril_raw_from_scales_quats


def _build_l(tri: torch.Tensor, beta: float) -> torch.Tensor:
    l = torch.zeros((*tri.shape[:-1], 3, 3), dtype=tri.dtype, device=tri.device)
    l[..., 0, 0] = tri[..., 0] * tri[..., 0] + beta
    l[..., 1, 0] = tri[..., 1]
    l[..., 2, 0] = tri[..., 2]
    l[..., 1, 1] = tri[..., 3] * tri[..., 3] + beta
    l[..., 2, 1] = tri[..., 4]
    l[..., 2, 2] = tri[..., 5] * tri[..., 5] + beta
    return l


def _precision_from_tril(tri: torch.Tensor, beta: float, eps: float = 0.0) -> torch.Tensor:
    l = _build_l(tri, beta)
    eye = torch.eye(3, dtype=tri.dtype, device=tri.device)
    return l @ l.transpose(-1, -2) + eps * eye


def _render_dense(
    means: torch.Tensor,
    precision: torch.Tensor,
    opacity: torch.Tensor,
    intensity: torch.Tensor,
    pixels: torch.Tensor,
    bg_alpha: float,
    bg_intensity: float,
) -> torch.Tensor:
    d = pixels[:, None, :] - means[None, :, :]
    q = torch.einsum("png,ngh,pnh->pn", d, precision, d)
    alpha_hat = opacity[None, :] * torch.exp(-0.5 * q)
    alpha = alpha_hat.sum(dim=1) + bg_alpha
    value = (alpha_hat * intensity[None, :]).sum(dim=1) + bg_alpha * bg_intensity
    return value / alpha.clamp_min(1e-12)


def test_triangular_precision_is_positive_definite_and_matches_formula():
    tri = torch.tensor(
        [
            [4.0, 0.2, -0.1, 4.5, 0.3, 5.0],
            [4.8, -0.4, 0.5, 4.2, -0.2, 4.6],
        ],
        dtype=torch.float64,
    )
    precision = _precision_from_tril(tri, beta=1e-4, eps=1e-6)

    eigvals = torch.linalg.eigvalsh(precision)
    assert torch.all(eigvals > 0.0)

    l = _build_l(tri, beta=1e-4)
    expected_00 = l[:, 0, 0] * l[:, 0, 0]
    expected_11 = l[:, 1, 0] * l[:, 1, 0] + l[:, 1, 1] * l[:, 1, 1]
    expected_22 = (
        l[:, 2, 0] * l[:, 2, 0]
        + l[:, 2, 1] * l[:, 2, 1]
        + l[:, 2, 2] * l[:, 2, 2]
    )
    assert torch.allclose(precision[:, 0, 0] - 1e-6, expected_00)
    assert torch.allclose(precision[:, 1, 1] - 1e-6, expected_11)
    assert torch.allclose(precision[:, 2, 2] - 1e-6, expected_22)


def test_bbox_half_width_uses_covariance_diagonal():
    tri = torch.tensor([[4.0, 1.0, -0.7, 4.5, 0.8, 5.0]], dtype=torch.float64)
    q_chi2 = 7.815
    precision = _precision_from_tril(tri, beta=1e-4)[0]
    covariance = torch.linalg.inv(precision)
    radius = torch.sqrt(q_chi2 * torch.diagonal(covariance))

    for axis in range(3):
        direction = covariance[:, axis] / torch.sqrt(covariance[axis, axis])
        delta = math.sqrt(q_chi2) * direction
        q = delta @ precision @ delta
        assert torch.allclose(q, torch.tensor(q_chi2, dtype=torch.float64), atol=1e-10)
        assert torch.allclose(delta[axis].abs(), radius[axis], atol=1e-10)


def test_ultragauss_scale_proxy_matches_covariance_diagonal():
    beta = 1e-4
    tri = torch.tensor(
        [
            [4.0, 1.0, -0.7, 4.5, 0.8, 5.0],
            [3.5, -0.4, 0.6, 4.2, -0.3, 4.8],
        ],
        dtype=torch.float64,
    )
    precision = _precision_from_tril(tri, beta=beta)
    expected = torch.sqrt(torch.diagonal(torch.linalg.inv(precision), dim1=1, dim2=2))

    actual = _approx_scale_from_precision_tril(tri, beta=beta)

    torch.testing.assert_close(actual, expected, rtol=1e-10, atol=1e-10)


def test_scene_scales_initialize_ultragauss_precision_scale():
    beta = 1e-4
    scales = torch.tensor([[0.3, 0.4, 0.5], [0.2, 0.25, 0.35]], dtype=torch.float32)
    quats = torch.tensor([[1.0, 0.0, 0.0, 0.0], [0.9238795, 0.0, 0.0, 0.3826834]], dtype=torch.float32)

    tri = _precision_tril_raw_from_scales_quats(scales, quats, beta)
    precision = _precision_from_tril(tri.double(), beta=beta)
    covariance = torch.linalg.inv(precision)
    eig_scales = torch.sqrt(torch.linalg.eigvalsh(covariance)).float()

    torch.testing.assert_close(eig_scales, torch.sort(scales, dim=1).values, rtol=1e-4, atol=1e-5)


def test_probe_space_and_world_space_dense_render_are_equivalent():
    means_world = torch.tensor(
        [[1.0, -0.5, 0.25], [-0.75, 0.6, 0.8]],
        dtype=torch.float64,
    )
    tri = torch.tensor(
        [[4.0, 0.15, -0.1, 4.3, 0.05, 4.7], [4.8, -0.2, 0.25, 4.1, -0.1, 4.4]],
        dtype=torch.float64,
    )
    precision_world = _precision_from_tril(tri, beta=1e-4)
    opacity = torch.tensor([0.7, 0.4], dtype=torch.float64)
    intensity = torch.tensor([0.2, 0.9], dtype=torch.float64)

    theta = torch.tensor(0.37, dtype=torch.float64)
    c, s = torch.cos(theta), torch.sin(theta)
    axis_lat = torch.stack([c, s, torch.tensor(0.0, dtype=torch.float64)])
    axis_ax = torch.stack([-s, c, torch.tensor(0.0, dtype=torch.float64)])
    axis_ele = torch.tensor([0.0, 0.0, 1.0], dtype=torch.float64)
    axes = torch.stack([axis_lat, axis_ax, axis_ele], dim=1)
    origin = torch.tensor([0.2, -0.3, 0.4], dtype=torch.float64)

    pixels_probe = torch.tensor(
        [[-0.5, 0.1, 0.0], [0.0, 0.0, 0.0], [0.4, 0.6, 0.0], [0.8, -0.2, 0.0]],
        dtype=torch.float64,
    )
    pixels_world = origin + pixels_probe @ axes.T

    world_render = _render_dense(
        means_world,
        precision_world,
        opacity,
        intensity,
        pixels_world,
        bg_alpha=1e-6,
        bg_intensity=0.0,
    )

    means_probe = (means_world - origin) @ axes
    precision_probe = axes.T @ precision_world @ axes
    probe_render = _render_dense(
        means_probe,
        precision_probe,
        opacity,
        intensity,
        pixels_probe,
        bg_alpha=1e-6,
        bg_intensity=0.0,
    )

    assert torch.allclose(world_render, probe_render, atol=1e-12)


def test_ultragauss_weighted_average_matches_hand_computation():
    means = torch.tensor([[0.0, 0.0, 0.0], [0.2, 0.0, 0.0]], dtype=torch.float64)
    precision = torch.eye(3, dtype=torch.float64).expand(2, 3, 3).clone()
    opacity = torch.tensor([0.5, 0.25], dtype=torch.float64)
    intensity = torch.tensor([0.2, 0.8], dtype=torch.float64)
    pixel = torch.tensor([[0.0, 0.0, 0.0]], dtype=torch.float64)

    rendered = _render_dense(
        means,
        precision,
        opacity,
        intensity,
        pixel,
        bg_alpha=1e-3,
        bg_intensity=0.1,
    )

    w0 = torch.tensor(1.0, dtype=torch.float64)
    w1 = torch.exp(torch.tensor(-0.5 * 0.2 * 0.2, dtype=torch.float64))
    a0 = opacity[0] * w0
    a1 = opacity[1] * w1
    expected = (a0 * intensity[0] + a1 * intensity[1] + 1e-3 * 0.1) / (a0 + a1 + 1e-3)
    assert torch.allclose(rendered[0], expected)
