import sys
from pathlib import Path

import pytest
import torch

sys.path.append(str(Path(__file__).resolve().parents[2]))

from methods.nerfus.configs import FieldConfig as NerfusFieldConfig
from methods.nerfus.configs import ModelConfig as NerfusModelConfig
from methods.nerfus.field import NerfusField
from methods.nerfus.ray_sampler import PatchSpec as NerfusPatchSpec
from methods.nerfus.ray_sampler import extract_slice_grid as nerfus_extract_slice_grid
from methods.nerfus.render import render_slice as nerfus_render_slice
from methods.ultranerf.configs import FieldConfig as UltraNeRFFieldConfig
from methods.ultranerf.configs import ModelConfig as UltraNeRFModelConfig
from methods.ultranerf.field import UltraNeRFField
from methods.ultranerf.ray_sampler import PatchSpec as UltraNeRFPatchSpec
from methods.ultranerf.ray_sampler import extract_slice_grid as ultranerf_extract_slice_grid
from methods.ultranerf.render import render_slice as ultranerf_render_slice
from scene.meta.dataset_meta.tus_rec_2024 import TUS_REC_2024
from scene.meta.slice_meta.linear import LinearSliceMeta


def _tus_native_slice_meta():
    device = torch.device("cpu")
    image = torch.zeros(TUS_REC_2024.H, TUS_REC_2024.W, dtype=torch.float32)
    pose = torch.eye(4, dtype=torch.float32)
    return LinearSliceMeta(TUS_REC_2024, device).initialize_from_image_pose(image, pose, frame_id=0)


@pytest.mark.parametrize(
    "extract_slice_grid",
    [ultranerf_extract_slice_grid, nerfus_extract_slice_grid],
)
def test_tus_rec_2024_native_slice_grid_uses_left_lateral_origin(extract_slice_grid):
    meta = _tus_native_slice_meta()

    points, depths, image = extract_slice_grid(meta)

    assert TUS_REC_2024.center_lateral is False
    assert points.shape == (TUS_REC_2024.H, TUS_REC_2024.W, 3)
    assert depths.shape == (TUS_REC_2024.H, TUS_REC_2024.W)
    assert image.shape == (TUS_REC_2024.H, TUS_REC_2024.W)

    expected_last_x = (TUS_REC_2024.W - 1) * TUS_REC_2024.delta_lateral_mm
    expected_last_y = (TUS_REC_2024.H - 1) * TUS_REC_2024.delta_axial_mm

    assert torch.isclose(points[0, 0, 0], torch.tensor(0.0), atol=1.0e-5)
    assert torch.isclose(points[0, -1, 0], torch.tensor(expected_last_x), atol=1.0e-5)
    assert torch.isclose(points[0, 0, 1], torch.tensor(TUS_REC_2024.near_mm), atol=1.0e-5)
    assert torch.isclose(points[-1, 0, 1], torch.tensor(expected_last_y), atol=1.0e-5)
    assert torch.isclose(depths[0, 0], torch.tensor(0.0), atol=1.0e-5)
    assert torch.isclose(depths[-1, 0], torch.tensor(expected_last_y), atol=1.0e-5)


def test_ultranerf_render_supports_native_tus_rec_2024_forward_backward():
    cfg = UltraNeRFModelConfig()
    cfg.field = UltraNeRFFieldConfig(depth=3, width=12, pe_num_freqs=3)
    cfg.ray.eval_chunk_size = 256
    cfg.render.boundary_sampling_mode = "expectation"
    cfg.render.scatter_sampling_mode = "expectation"
    cfg.render.enable_psf = False

    field = UltraNeRFField(cfg.field)
    patch = UltraNeRFPatchSpec(top=3, left=5, height=8, width=8)
    pred, depth, info = ultranerf_render_slice(cfg, _tus_native_slice_meta(), field, patch=patch)

    assert pred.shape == (8, 8)
    assert depth.shape == (8, 8)
    assert torch.isfinite(pred).all()
    assert "transmission" in info
    pred.mean().backward()
    assert any(p.grad is not None for p in field.parameters())


def test_nerfus_render_supports_native_tus_rec_2024_forward_backward():
    cfg = NerfusModelConfig()
    cfg.field = NerfusFieldConfig(depth=3, width=12, pe_num_freqs=3)
    cfg.ray.eval_chunk_size = 256
    cfg.render.boundary_sampling_mode = "expectation"
    cfg.render.scatter_sampling_mode = "expectation"
    cfg.render.enable_psf = False

    field = NerfusField(cfg.field)
    patch = NerfusPatchSpec(top=3, left=5, height=8, width=8)
    pred, depth, info = nerfus_render_slice(cfg, _tus_native_slice_meta(), field, patch=patch)

    assert pred.shape == (8, 8)
    assert depth.shape == (8, 8)
    assert torch.isfinite(pred).all()
    assert "transmission" in info
    pred.mean().backward()
    assert any(p.grad is not None for p in field.parameters())
