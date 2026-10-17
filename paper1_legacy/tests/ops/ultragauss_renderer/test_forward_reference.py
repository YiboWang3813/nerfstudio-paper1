import pytest
import sys
from pathlib import Path

torch = pytest.importorskip("torch")

_PKG_ROOT = Path(__file__).resolve().parents[3] / "ops" / "ultragauss_renderer"
if str(_PKG_ROOT) not in sys.path:
    sys.path.insert(0, str(_PKG_ROOT))

from ultragauss_renderer import SliceGeometry, UltraGaussRenderConfig, UltraGaussRenderer


def _build_precision(tri: torch.Tensor, beta: float) -> torch.Tensor:
    l = torch.zeros((tri.shape[0], 3, 3), dtype=tri.dtype, device=tri.device)
    l[:, 0, 0] = tri[:, 0] * tri[:, 0] + beta
    l[:, 1, 0] = tri[:, 1]
    l[:, 2, 0] = tri[:, 2]
    l[:, 1, 1] = tri[:, 3] * tri[:, 3] + beta
    l[:, 2, 1] = tri[:, 4]
    l[:, 2, 2] = tri[:, 5] * tri[:, 5] + beta
    return l @ l.transpose(1, 2)


def _dense_reference(
    means: torch.Tensor,
    precision_tril: torch.Tensor,
    opacities: torch.Tensor,
    intensities: torch.Tensor,
    pixels: torch.Tensor,
    chi2_threshold: float,
    bg_alpha: float,
    bg_intensity: float,
    beta: float,
) -> torch.Tensor:
    precision = _build_precision(precision_tril, beta)
    d = pixels[:, None, :] - means[None, :, :]
    q = torch.einsum("png,ngh,pnh->pn", d, precision, d)
    weight = torch.exp(-0.5 * q) * (q <= chi2_threshold)
    alpha_hat = opacities[None, :] * weight
    alpha = alpha_hat.sum(dim=1) + bg_alpha
    value = (alpha_hat * intensities[None, :]).sum(dim=1) + bg_alpha * bg_intensity
    return value / alpha.clamp_min(1e-8)


def _make_identity_geometry(device: torch.device) -> SliceGeometry:
    return SliceGeometry(
        origin_world=torch.zeros(3, device=device, dtype=torch.float32),
        axis_lateral_world=torch.tensor([1.0, 0.0, 0.0], device=device),
        axis_axial_world=torch.tensor([0.0, 1.0, 0.0], device=device),
        axis_elevational_world=torch.tensor([0.0, 0.0, 1.0], device=device),
    )


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
def test_ultragauss_forward_matches_dense_reference_identity_probe():
    device = torch.device("cuda")
    h, w = 4, 5
    delta_lat = 0.5
    delta_ax = 0.6
    near = 0.2
    beta = 1e-4
    chi2 = 7.815
    bg_alpha = 1e-6

    xs = (torch.arange(w, device=device, dtype=torch.float32) - (w - 1) * 0.5) * delta_lat
    ys = torch.arange(h, device=device, dtype=torch.float32) * delta_ax + near
    yy, xx = torch.meshgrid(ys, xs, indexing="ij")
    pixels = torch.stack([xx.reshape(-1), yy.reshape(-1), torch.zeros(h * w, device=device)], dim=1)

    means = torch.tensor(
        [[0.0, 0.8, 0.0], [0.4, 1.1, 0.15], [-0.5, 1.4, -0.1]],
        device=device,
        dtype=torch.float32,
    )
    precision_tril = torch.tensor(
        [[1.1, 0.2, -0.1, 1.0, 0.15, 1.2], [1.3, -0.25, 0.1, 1.1, -0.2, 1.0], [1.0, 0.1, 0.2, 1.4, 0.05, 1.2]],
        device=device,
        dtype=torch.float32,
    )
    opacities = torch.tensor([0.7, 0.4, 0.55], device=device, dtype=torch.float32)
    intensities = torch.tensor([0.2, 0.9, 0.5], device=device, dtype=torch.float32)

    renderer = UltraGaussRenderer(
        UltraGaussRenderConfig(
            H=h,
            W=w,
            lateral_spacing_mm_per_pix=delta_lat,
            axial_spacing_mm_per_pix=delta_ax,
            axial_near_mm=near,
            chi2_threshold=chi2,
            mapper_sort_by_depth=False,
            fixed_bg_alpha=bg_alpha,
            fixed_bg_intensity=0.0,
            diag_beta=beta,
            enable_shadow=False,
        )
    )
    image, alpha, depth, intermediates = renderer(
        gauss_means_world=means,
        gauss_precision_tril=precision_tril,
        gauss_opacities=opacities,
        gauss_intensities=intensities,
        pixel_positions_world=pixels,
        geometry=_make_identity_geometry(device),
        return_intermediates=True,
    )

    expected = _dense_reference(
        means,
        precision_tril,
        opacities,
        intensities,
        pixels,
        chi2_threshold=chi2,
        bg_alpha=bg_alpha,
        bg_intensity=0.0,
        beta=beta,
    ).view(h, w)

    assert torch.all(alpha >= bg_alpha)
    assert int(intermediates["pixel_gauss_indices"].numel()) > 0
    torch.testing.assert_close(image, expected, rtol=1e-5, atol=1e-5)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
def test_ultragauss_backward_matches_dense_reference_identity_probe():
    device = torch.device("cuda")
    h, w = 3, 4
    delta_lat = 0.4
    delta_ax = 0.5
    near = 0.1
    beta = 1e-4
    chi2 = 7.815
    bg_alpha = 1e-6

    xs = (torch.arange(w, device=device, dtype=torch.float32) - (w - 1) * 0.5) * delta_lat
    ys = torch.arange(h, device=device, dtype=torch.float32) * delta_ax + near
    yy, xx = torch.meshgrid(ys, xs, indexing="ij")
    pixels = torch.stack([xx.reshape(-1), yy.reshape(-1), torch.zeros(h * w, device=device)], dim=1)

    base_means = torch.tensor(
        [[-0.15, 0.45, 0.05], [0.35, 0.85, -0.08]],
        device=device,
        dtype=torch.float32,
    )
    base_precision_tril = torch.tensor(
        [[1.4, 0.35, -0.18, 1.1, 0.22, 1.25], [1.2, -0.28, 0.24, 1.35, -0.31, 1.05]],
        device=device,
        dtype=torch.float32,
    )
    base_opacities = torch.tensor([0.62, 0.48], device=device, dtype=torch.float32)
    base_intensities = torch.tensor([0.25, 0.82], device=device, dtype=torch.float32)
    loss_weights = torch.linspace(0.7, 1.4, h * w, device=device, dtype=torch.float32).view(h, w)

    cuda_inputs = [x.detach().clone().requires_grad_(True) for x in (base_means, base_precision_tril, base_opacities, base_intensities)]
    ref_inputs = [x.detach().clone().requires_grad_(True) for x in (base_means, base_precision_tril, base_opacities, base_intensities)]

    renderer = UltraGaussRenderer(
        UltraGaussRenderConfig(
            H=h,
            W=w,
            lateral_spacing_mm_per_pix=delta_lat,
            axial_spacing_mm_per_pix=delta_ax,
            axial_near_mm=near,
            chi2_threshold=chi2,
            mapper_sort_by_depth=False,
            fixed_bg_alpha=bg_alpha,
            fixed_bg_intensity=0.0,
            diag_beta=beta,
            enable_shadow=False,
        )
    )
    image, _, _, _ = renderer(
        gauss_means_world=cuda_inputs[0],
        gauss_precision_tril=cuda_inputs[1],
        gauss_opacities=cuda_inputs[2],
        gauss_intensities=cuda_inputs[3],
        pixel_positions_world=pixels,
        geometry=_make_identity_geometry(device),
        return_intermediates=False,
    )
    (image * loss_weights).sum().backward()

    expected = _dense_reference(
        ref_inputs[0],
        ref_inputs[1],
        ref_inputs[2],
        ref_inputs[3],
        pixels,
        chi2_threshold=chi2,
        bg_alpha=bg_alpha,
        bg_intensity=0.0,
        beta=beta,
    ).view(h, w)
    (expected * loss_weights).sum().backward()

    for actual, reference in zip(cuda_inputs, ref_inputs):
        torch.testing.assert_close(actual.grad, reference.grad, rtol=2e-4, atol=2e-4)
