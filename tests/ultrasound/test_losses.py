"""Native loss and schedule contracts, independent of retired source code."""

import torch

from nerfstudio.engine.ultrasound_scheduler import UltrasoundExponentialSchedulerConfig
from nerfstudio.model_components.ultrasound_losses import UltrasoundSSIM


def test_native_schedule_preserves_one_based_update_timing():
    initial = 1.6e-4
    optimizer = torch.optim.Adam([torch.nn.Parameter(torch.zeros(1))], lr=initial)
    scheduler = UltrasoundExponentialSchedulerConfig().setup().get_scheduler(optimizer, initial)
    for step in (0, 1, 29998, 29999, 50000):
        expected = initial * 0.1 ** min((step + 1) / 30000, 1)
        actual = scheduler.lr_lambdas[0](step) * initial
        assert abs(actual - expected) < 1e-12


def test_ssim_identity_and_directional_gradient():
    torch.manual_seed(4)
    metric = UltrasoundSSIM().double()
    target = torch.rand(2, 3, 19, 23, dtype=torch.float64)
    torch.testing.assert_close(metric(target, target), torch.ones((), dtype=torch.float64))
    pred = (torch.randn_like(target) * 0.7).requires_grad_()
    loss = 1 - metric(pred, target)
    gradient = torch.autograd.grad(loss, pred)[0]
    assert torch.isfinite(loss) and torch.isfinite(gradient).all()
    direction = torch.randn_like(pred)
    direction /= direction.norm()
    epsilon = 1e-5
    with torch.no_grad():
        difference = metric(pred - epsilon * direction, target) - metric(pred + epsilon * direction, target)
        numerical = difference / (2 * epsilon)
    torch.testing.assert_close((gradient * direction).sum(), numerical, atol=1e-8, rtol=1e-5)
