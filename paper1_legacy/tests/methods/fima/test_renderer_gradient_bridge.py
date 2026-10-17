import sys
from pathlib import Path

import pytest
import torch
import torch.nn.functional as F

from methods.fima.configs import FiMAConfig
from methods.fima.pose_estimator_adjacent import FiMAAdjacentPairEstimator
from methods.fima.pose_geometry import se3_from_6dof


def _max_finite_grad(module: torch.nn.Module) -> float:
    vals = []
    for p in module.parameters():
        if p.grad is not None:
            assert torch.isfinite(p.grad).all()
            vals.append(float(p.grad.detach().abs().max().cpu()))
    return max(vals) if vals else 0.0


def _make_tiny_adjacent_cfg() -> FiMAConfig:
    cfg = FiMAConfig()
    cfg.pem.input_mode = "adjacent_pair"
    cfg.pem.d_model = 16
    cfg.pem.stem_channels = [4, 8]
    cfg.pem.input_downsample_stride = 1
    cfg.pem.occ_always_two_channels = False
    return cfg


def test_ours_renderer_loss_backpropagates_to_fima_adjacent_estimator():
    if not torch.cuda.is_available():
        pytest.skip("CUDA is required for ours renderer gradient bridge validation")

    repo = Path(__file__).resolve().parents[3]
    ops_root = repo / "ops" / "ours_renderer"
    common_root = repo / "tests" / "ops" / "ours_renderer"
    sys.path.insert(0, str(ops_root))
    sys.path.insert(0, str(common_root))

    try:
        from ours_renderer.autograd import render_projected_slice
        from pose_experiment_common import dataset_meta, make_asymmetric_scene
        from ours_renderer.grid_geometry import build_slice_grid_pack
    except Exception as exc:
        pytest.skip(f"ours renderer extension is not available: {exc!r}")

    torch.manual_seed(20260615)
    device = torch.device("cuda:0")
    dtype = torch.float32

    render_cfg, _, means, scales, quats, opacities, intensities = make_asymmetric_scene(device, dtype)
    grid = build_slice_grid_pack(dataset_meta(render_cfg), device=device)
    target_pose6 = torch.zeros(6, device=device, dtype=dtype)
    target = render_projected_slice(
        render_cfg,
        means,
        scales,
        quats,
        opacities,
        intensities,
        se3_from_6dof(target_pose6),
        grid,
    ).detach()

    images = target.clamp(0.0, 1.0).view(1, 1, 1, render_cfg.H, render_cfg.W).repeat(1, 2, 1, 1, 1)

    model = FiMAAdjacentPairEstimator(_make_tiny_adjacent_cfg()).to(device)
    with torch.no_grad():
        last = model.head[-1]
        last.weight.normal_(mean=0.0, std=1.0e-3)
        last.bias.copy_(torch.tensor([0.7, -0.4, 0.25, 0.035, -0.030, 0.025], device=device))

    def render_loss():
        pred_pose6 = model(images)["pred_pose_6"][0]
        pred = render_projected_slice(
            render_cfg,
            means,
            scales,
            quats,
            opacities,
            intensities,
            se3_from_6dof(pred_pose6),
            grid,
        )
        return F.mse_loss(pred, target), pred_pose6

    loss, pred_pose6 = render_loss()
    assert torch.isfinite(loss)
    assert torch.isfinite(pred_pose6).all()
    loss.backward()

    assert _max_finite_grad(model.head) > 1.0e-10
    assert _max_finite_grad(model.stem) > 1.0e-12

    opt = torch.optim.Adam(model.parameters(), lr=3.0e-3)
    initial = float(loss.detach().cpu())
    best = initial
    for _ in range(12):
        opt.zero_grad(set_to_none=True)
        step_loss, _ = render_loss()
        step_loss.backward()
        opt.step()
        best = min(best, float(step_loss.detach().cpu()))

    final_loss, final_pose6 = render_loss()
    final = float(final_loss.detach().cpu())
    best = min(best, final)

    assert torch.isfinite(final_loss)
    assert torch.isfinite(final_pose6).all()
    assert best < initial, f"render loss did not decrease: initial={initial:.8e}, best={best:.8e}, final={final:.8e}"
