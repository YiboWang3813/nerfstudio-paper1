import torch

from utils.loss.image_loss import HighPassL1Loss


def test_highpass_l1_zero_for_identical_images():
    loss = HighPassL1Loss(kernel_size=5, sigma=1.0)
    img = torch.rand(16, 16)
    assert torch.isclose(loss(img, img), torch.tensor(0.0), atol=1.0e-7)


def test_highpass_l1_responds_to_local_detail():
    loss = HighPassL1Loss(kernel_size=5, sigma=1.0)
    pred = torch.zeros(16, 16)
    gt = torch.zeros(16, 16)
    gt[8, 8] = 1.0
    assert float(loss(pred, gt)) > 0.0


def test_highpass_l1_supports_batched_images():
    loss = HighPassL1Loss(kernel_size=5, sigma=1.0)
    pred = torch.zeros(2, 1, 16, 16)
    gt = torch.zeros(2, 1, 16, 16)
    gt[:, :, 8, 8] = 1.0
    assert loss(pred, gt).ndim == 0

