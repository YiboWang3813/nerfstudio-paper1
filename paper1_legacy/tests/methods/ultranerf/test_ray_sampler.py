import sys
from pathlib import Path

import torch

sys.path.append(str(Path(__file__).resolve().parents[3]))

from methods.ultranerf.ray_sampler import extract_slice_grid, random_patch_spec
from scene.meta.dataset_meta.synthetic_liver import SYNTHETIC_LIVER
from scene.meta.slice_meta.linear import LinearSliceMeta


def _slice_meta():
    device = torch.device("cpu")
    image = torch.zeros(SYNTHETIC_LIVER.H, SYNTHETIC_LIVER.W)
    pose = torch.eye(4)
    return LinearSliceMeta(SYNTHETIC_LIVER, device).initialize_from_image_pose(image, pose, frame_id=0)


def test_full_grid_shape_and_range():
    meta = _slice_meta()
    points, depths, image = extract_slice_grid(meta)
    assert points.shape == (512, 256, 3)
    assert depths.shape == (512, 256)
    assert image.shape == (512, 256)
    assert torch.isclose(points[..., 0].min(), torch.tensor(-39.84375), atol=1.0e-5)
    assert torch.isclose(points[..., 0].max(), torch.tensor(39.84375), atol=1.0e-5)
    assert torch.isclose(depths.min(), torch.tensor(0.0), atol=1.0e-5)
    assert torch.isclose(depths.max(), torch.tensor(139.7265625), atol=1.0e-5)


def test_patch_extract_shape():
    meta = _slice_meta()
    patch = random_patch_spec(512, 256, 64, 32)
    points, depths, image = extract_slice_grid(meta, patch)
    assert points.shape == (64, 32, 3)
    assert depths.shape == (64, 32)
    assert image.shape == (64, 32)
