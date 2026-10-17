import importlib

import pytest
import torch


def _reference(pixels, means, invs, weights, colors):
    d = pixels[:, None, :] - means[None, :, :]
    power = -0.5 * torch.einsum("pgi,gij,pgj->pg", d, invs, d)
    gaussian = torch.exp(torch.clamp(power, min=-50.0, max=0.0))
    weighted = gaussian.unsqueeze(-1) * weights.unsqueeze(0)
    image = (weighted * colors.unsqueeze(0)).sum(dim=1)
    alpha = weighted.sum(dim=1)
    return image, alpha


def _bounded_reference(
    image_height,
    image_width,
    half_width,
    half_height,
    means,
    invs,
    weights,
    colors,
    radius_sigma,
):
    xs = torch.linspace(-half_width, half_width, int(image_width), device=means.device, dtype=means.dtype)
    ys = torch.linspace(-half_height, half_height, int(image_height), device=means.device, dtype=means.dtype)
    yy, xx = torch.meshgrid(ys, xs, indexing="ij")
    pixels = torch.stack([xx.reshape(-1), yy.reshape(-1)], dim=-1)
    d = pixels[:, None, :] - means[None, :, :]
    quad = torch.einsum("pgi,gij,pgj->pg", d, invs, d)
    gaussian = torch.exp(torch.clamp(-0.5 * quad, min=-50.0, max=0.0))
    gaussian = torch.where(quad <= float(radius_sigma) ** 2, gaussian, torch.zeros_like(gaussian))
    weighted = gaussian.unsqueeze(-1) * weights.unsqueeze(0)
    image = (weighted * colors.unsqueeze(0)).sum(dim=1)
    alpha = weighted.sum(dim=1)
    return image, alpha


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is required")
def test_additive_splat_cuda_matches_torch_forward_backward():
    renderer = pytest.importorskip("slice_adapted_3dgs_renderer")
    importlib.import_module("slice_adapted_3dgs_renderer._C")

    device = torch.device("cuda")
    pixels = torch.tensor(
        [
            [-1.0, -1.0],
            [0.0, 0.0],
            [1.0, 0.5],
            [2.0, -0.5],
        ],
        device=device,
        dtype=torch.float32,
    )
    means = torch.tensor(
        [
            [0.0, 0.0],
            [1.0, -0.25],
            [-0.5, 0.75],
        ],
        device=device,
        dtype=torch.float32,
        requires_grad=True,
    )
    invs = torch.tensor(
        [
            [[1.0, 0.1], [0.1, 1.2]],
            [[0.8, -0.05], [-0.05, 1.1]],
            [[1.3, 0.0], [0.0, 0.9]],
        ],
        device=device,
        dtype=torch.float32,
        requires_grad=True,
    )
    weights = torch.tensor([[0.7], [0.2], [0.5]], device=device, dtype=torch.float32, requires_grad=True)
    colors = torch.tensor(
        [
            [0.9, 0.1, 0.2],
            [0.1, 0.8, 0.3],
            [0.4, 0.3, 0.7],
        ],
        device=device,
        dtype=torch.float32,
        requires_grad=True,
    )

    cuda_image, cuda_alpha = renderer.additive_splat(pixels, means, invs, weights, colors)
    ref_image, ref_alpha = _reference(pixels, means, invs, weights, colors)
    torch.testing.assert_close(cuda_image, ref_image, rtol=1.0e-5, atol=1.0e-5)
    torch.testing.assert_close(cuda_alpha, ref_alpha, rtol=1.0e-5, atol=1.0e-5)

    grad_image = torch.tensor(
        [
            [0.1, -0.2, 0.3],
            [0.4, 0.5, -0.6],
            [-0.7, 0.8, 0.9],
            [1.0, -1.1, 1.2],
        ],
        device=device,
        dtype=torch.float32,
    )
    grad_alpha = torch.tensor([[0.2], [-0.3], [0.4], [-0.5]], device=device, dtype=torch.float32)

    cuda_loss = (cuda_image * grad_image).sum() + (cuda_alpha * grad_alpha).sum()
    cuda_grads = torch.autograd.grad(cuda_loss, [means, invs, weights, colors], retain_graph=True)

    ref_loss = (ref_image * grad_image).sum() + (ref_alpha * grad_alpha).sum()
    ref_grads = torch.autograd.grad(ref_loss, [means, invs, weights, colors])

    for got, expected in zip(cuda_grads, ref_grads):
        torch.testing.assert_close(got, expected, rtol=2.0e-4, atol=2.0e-4)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is required")
def test_additive_splat_bounded_cuda_matches_truncated_reference():
    renderer = pytest.importorskip("slice_adapted_3dgs_renderer")
    importlib.import_module("slice_adapted_3dgs_renderer._C")

    device = torch.device("cuda")
    image_height = 7
    image_width = 9
    half_width = 2.0
    half_height = 1.5
    radius_sigma = 2.25
    means = torch.tensor(
        [
            [-0.4, 0.1],
            [0.9, -0.5],
            [1.6, 0.8],
        ],
        device=device,
        dtype=torch.float32,
        requires_grad=True,
    )
    invs = torch.tensor(
        [
            [[1.2, 0.08], [0.08, 1.4]],
            [[0.9, -0.03], [-0.03, 1.1]],
            [[1.6, 0.12], [0.12, 1.3]],
        ],
        device=device,
        dtype=torch.float32,
        requires_grad=True,
    )
    weights = torch.tensor(
        [[0.35], [0.6], [0.2]],
        device=device,
        dtype=torch.float32,
        requires_grad=True,
    )
    colors = torch.tensor(
        [
            [0.8, 0.2, 0.1],
            [0.2, 0.7, 0.4],
            [0.5, 0.3, 0.9],
        ],
        device=device,
        dtype=torch.float32,
        requires_grad=True,
    )

    cuda_image, cuda_alpha = renderer.additive_splat_bounded(
        image_height,
        image_width,
        half_width,
        half_height,
        means,
        invs,
        weights,
        colors,
        radius_sigma,
    )
    ref_image, ref_alpha = _bounded_reference(
        image_height,
        image_width,
        half_width,
        half_height,
        means,
        invs,
        weights,
        colors,
        radius_sigma,
    )
    torch.testing.assert_close(cuda_image, ref_image, rtol=2.0e-4, atol=2.0e-4)
    torch.testing.assert_close(cuda_alpha, ref_alpha, rtol=2.0e-4, atol=2.0e-4)

    grad_image = torch.linspace(
        -0.3,
        0.4,
        image_height * image_width * 3,
        device=device,
        dtype=torch.float32,
    ).reshape(image_height * image_width, 3)
    grad_alpha = torch.linspace(
        0.2,
        -0.1,
        image_height * image_width,
        device=device,
        dtype=torch.float32,
    ).reshape(image_height * image_width, 1)

    cuda_loss = (cuda_image * grad_image).sum() + (cuda_alpha * grad_alpha).sum()
    cuda_grads = torch.autograd.grad(cuda_loss, [means, invs, weights, colors], retain_graph=True)

    ref_loss = (ref_image * grad_image).sum() + (ref_alpha * grad_alpha).sum()
    ref_grads = torch.autograd.grad(ref_loss, [means, invs, weights, colors])

    for got, expected in zip(cuda_grads, ref_grads):
        torch.testing.assert_close(got, expected, rtol=2.0e-4, atol=2.0e-4)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is required")
def test_additive_splat_bounded_tile_binned_path_matches_truncated_reference():
    renderer = pytest.importorskip("slice_adapted_3dgs_renderer")
    importlib.import_module("slice_adapted_3dgs_renderer._C")

    device = torch.device("cuda")
    image_height = 2
    image_width = 3
    half_width = 1.0
    half_height = 0.8
    radius_sigma = 2.0
    n_gauss = 17000
    gen = torch.Generator(device=device).manual_seed(123)
    means = (torch.rand((n_gauss, 2), device=device, dtype=torch.float32, generator=gen) - 0.5).requires_grad_(True)
    inv_diag = 3.0 + torch.rand((n_gauss, 2), device=device, dtype=torch.float32, generator=gen)
    invs = torch.zeros((n_gauss, 2, 2), device=device, dtype=torch.float32)
    invs[:, 0, 0] = inv_diag[:, 0]
    invs[:, 1, 1] = inv_diag[:, 1]
    invs.requires_grad_(True)
    weights = (0.002 * torch.rand((n_gauss, 1), device=device, dtype=torch.float32, generator=gen)).requires_grad_(True)
    colors = torch.rand((n_gauss, 3), device=device, dtype=torch.float32, generator=gen).requires_grad_(True)

    cuda_image, cuda_alpha = renderer.additive_splat_bounded(
        image_height,
        image_width,
        half_width,
        half_height,
        means,
        invs,
        weights,
        colors,
        radius_sigma,
    )
    ref_image, ref_alpha = _bounded_reference(
        image_height,
        image_width,
        half_width,
        half_height,
        means,
        invs,
        weights,
        colors,
        radius_sigma,
    )
    torch.testing.assert_close(cuda_image, ref_image, rtol=2.0e-4, atol=2.0e-4)
    torch.testing.assert_close(cuda_alpha, ref_alpha, rtol=2.0e-4, atol=2.0e-4)

    grad_image = torch.linspace(
        -0.2,
        0.3,
        image_height * image_width * 3,
        device=device,
        dtype=torch.float32,
    ).reshape(image_height * image_width, 3)
    grad_alpha = torch.linspace(
        0.1,
        -0.2,
        image_height * image_width,
        device=device,
        dtype=torch.float32,
    ).reshape(image_height * image_width, 1)

    cuda_loss = (cuda_image * grad_image).sum() + (cuda_alpha * grad_alpha).sum()
    cuda_grads = torch.autograd.grad(cuda_loss, [means, invs, weights, colors], retain_graph=True)

    ref_loss = (ref_image * grad_image).sum() + (ref_alpha * grad_alpha).sum()
    ref_grads = torch.autograd.grad(ref_loss, [means, invs, weights, colors])

    for got, expected in zip(cuda_grads, ref_grads):
        torch.testing.assert_close(got, expected, rtol=4.0e-4, atol=4.0e-4)
