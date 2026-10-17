import sys
from pathlib import Path

import torch

sys.path.append(str(Path(__file__).resolve().parents[3]))

from methods.nerfus.configs import FieldConfig, ModelConfig
from methods.nerfus.field import NerfusField
from methods.nerfus.ray_sampler import PatchSpec
from methods.nerfus.render import _indicator, render_slice
from scene.meta.dataset_meta.synthetic_liver import SYNTHETIC_LIVER
from scene.meta.slice_meta.linear import LinearSliceMeta


def _slice_meta():
    device = torch.device("cpu")
    image = torch.zeros(SYNTHETIC_LIVER.H, SYNTHETIC_LIVER.W)
    pose = torch.eye(4)
    return LinearSliceMeta(SYNTHETIC_LIVER, device).initialize_from_image_pose(image, pose, frame_id=0)


def test_render_patch_backward():
    cfg = ModelConfig()
    cfg.field = FieldConfig(depth=4, width=16, pe_num_freqs=4)
    cfg.ray.train_patch_height = 16
    cfg.ray.train_patch_width = 16
    cfg.ray.eval_chunk_size = 1024
    field = NerfusField(cfg.field)
    pred, depth, info = render_slice(cfg, _slice_meta(), field, patch=PatchSpec(0, 0, 16, 16))
    assert pred.shape == (16, 16)
    assert depth.shape == (16, 16)
    assert torch.isfinite(pred).all()
    assert "transmission" in info
    loss = pred.mean()
    loss.backward()
    assert any(p.grad is not None for p in field.parameters())


def test_psf_toggle_preserves_shape():
    cfg = ModelConfig()
    cfg.field = FieldConfig(depth=4, width=16, pe_num_freqs=4)
    cfg.ray.eval_chunk_size = 1024
    field = NerfusField(cfg.field)
    patch = PatchSpec(0, 0, 8, 8)
    cfg.render.enable_psf = True
    pred_psf, _, _ = render_slice(cfg, _slice_meta(), field, patch=patch)
    cfg.render.enable_psf = False
    pred_no_psf, _, _ = render_slice(cfg, _slice_meta(), field, patch=patch)
    assert pred_psf.shape == pred_no_psf.shape == (8, 8)


def test_straight_through_bernoulli_is_hard_with_probability_gradient():
    prob = torch.full((4, 4), 0.5, requires_grad=True)
    sample = _indicator(prob, "straight_through_bernoulli", temperature=0.1)
    assert torch.all((sample == 0) | (sample == 1))
    sample.mean().backward()
    assert prob.grad is not None
    assert torch.isfinite(prob.grad).all()


def test_bernoulli_mode_is_detached():
    prob = torch.full((4, 4), 0.5, requires_grad=True)
    sample = _indicator(prob, "bernoulli", temperature=0.1)
    assert torch.all((sample == 0) | (sample == 1))
    assert sample.requires_grad is False


def test_bernoulli_seed_is_repeatable():
    prob = torch.full((16, 16), 0.5)
    a = _indicator(prob, "bernoulli", temperature=0.1, sampling_seed=123)
    b = _indicator(prob, "bernoulli", temperature=0.1, sampling_seed=123)
    c = _indicator(prob, "bernoulli", temperature=0.1, sampling_seed=124)
    assert torch.equal(a, b)
    assert not torch.equal(a, c)


def test_render_seed_controls_stochastic_output():
    cfg = ModelConfig()
    cfg.field = FieldConfig(depth=4, width=16, pe_num_freqs=4)
    cfg.ray.eval_chunk_size = 1024
    cfg.render.boundary_sampling_mode = "bernoulli"
    cfg.render.scatter_sampling_mode = "bernoulli"
    field = NerfusField(cfg.field)
    patch = PatchSpec(0, 0, 16, 16)
    pred_a, _, _ = render_slice(cfg, _slice_meta(), field, patch=patch, sampling_seed=7)
    pred_b, _, _ = render_slice(cfg, _slice_meta(), field, patch=patch, sampling_seed=7)
    pred_c, _, _ = render_slice(cfg, _slice_meta(), field, patch=patch, sampling_seed=8)
    assert torch.equal(pred_a, pred_b)
    assert not torch.equal(pred_a, pred_c)
