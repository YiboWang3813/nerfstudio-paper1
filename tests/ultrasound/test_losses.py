"""Check the native loss against the preserved algorithm without loading its runtime."""

import ast
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn

from nerfstudio.engine.ultrasound_scheduler import UltrasoundExponentialSchedulerConfig
from nerfstudio.model_components.ultrasound_losses import UltrasoundSSIM


def test_native_schedule_preserves_one_based_update_timing():
    source = Path(__file__).resolve().parents[2] / "paper1_legacy/gaussian/lr_scheduler_manager.py"
    node = next(
        node
        for node in ast.parse(source.read_text()).body
        if isinstance(node, ast.FunctionDef) and node.name == "get_expon_lr_func"
    )
    namespace = {"np": np}
    exec(compile(ast.Module(body=[node], type_ignores=[]), str(source), "exec"), namespace)
    expected = namespace["get_expon_lr_func"](1.6e-4, 1.6e-5, max_steps=30000)
    optimizer = torch.optim.Adam([torch.nn.Parameter(torch.zeros(1))], lr=1.6e-4)
    scheduler = UltrasoundExponentialSchedulerConfig().setup().get_scheduler(optimizer, 1.6e-4)
    for step in (0, 1, 29998, 29999, 50000):
        actual = scheduler.lr_lambdas[0](step) * 1.6e-4
        assert abs(actual - expected(step + 1)) < 1e-12


def test_ssim_matches_frozen_loss_and_gradient():
    source = Path(__file__).resolve().parents[2] / "paper1_legacy/utils/loss/image_loss.py"
    node = next(
        node
        for node in ast.parse(source.read_text()).body
        if isinstance(node, ast.ClassDef) and node.name == "SSIMLoss"
    )
    node.decorator_list = []
    module = ast.Module(body=[node], type_ignores=[])
    namespace = {"torch": torch, "nn": nn, "F": F}
    exec(compile(module, str(source), "exec"), namespace)
    torch.manual_seed(4)
    pred = (torch.randn(2, 1, 19, 23) * 0.7).requires_grad_()
    target = torch.rand_like(pred)
    expected = namespace["SSIMLoss"]()(pred, target)
    actual = 1 - UltrasoundSSIM()(pred, target)
    torch.testing.assert_close(actual, expected, atol=1e-7, rtol=1e-7)
    torch.testing.assert_close(
        torch.autograd.grad(actual, pred)[0], torch.autograd.grad(expected, pred)[0], atol=1e-7, rtol=1e-6
    )
