import torch

from configs.setup import load_hydra_config
from methods.ultrags_baseline.camera_adapter import build_virtual_camera
from methods.ultrags_baseline.gaussian_container import UltraGSBaselineContainer
from methods.ultrags_baseline.render import _empty_render, _slice_candidates_and_attenuation
from scene.meta.dataset_meta.synthetic_liver import SYNTHETIC_LIVER
from scene.meta.slice_meta.linear import LinearSliceMeta


def _physical_cfg():
    return load_hydra_config(
        "preproduction/baseline_reproduction/ultrags_baseline/smoke",
        overrides=[
            "task=dry_run",
            "system.device=cpu",
            "engine.iterations=1",
            "model.render.camera_mode=slice_plane_physical",
            "model.render.enable_slice_cull=true",
            "model.render.slice_thickness_ratio=0.75",
            "model.render.slice_thickness_mm=null",
            "model.render.slice_cull_std=3.0",
            "model.render.slice_cull_scale_multiplier=3.0",
            "model.render.opacity_slice_attenuation=true",
        ],
    )


def _slice_meta(device):
    return LinearSliceMeta(SYNTHETIC_LIVER, device).initialize_from_image_pose(
        torch.zeros((SYNTHETIC_LIVER.H, SYNTHETIC_LIVER.W), dtype=torch.float32, device=device),
        torch.eye(4, dtype=torch.float32, device=device),
        frame_id=0,
    )


def test_slice_cull_keeps_near_plane_disks_and_rejects_far_disks_cpu():
    cfg = _physical_cfg()
    device = torch.device("cpu")
    slice_meta = _slice_meta(device)
    center_idx = slice_meta.pixel_pos.shape[0] // 2
    center = slice_meta.pixel_pos[center_idx]
    elev = torch.nn.functional.normalize(slice_meta.vec_elevational.float(), dim=0)
    means = torch.stack([center, center + elev * 0.5, center + elev * 10.0], dim=0)

    container = UltraGSBaselineContainer(cfg, device)
    container.init_from_scene(
        {
            "means": means,
            "scales": torch.full((3, 3), 0.3, dtype=torch.float32),
            "intensities": torch.full((3, 1), 0.5, dtype=torch.float32),
        }
    )
    camera = build_virtual_camera(cfg.model.render, SYNTHETIC_LIVER, slice_meta)

    indices, attenuation = _slice_candidates_and_attenuation(cfg.model, SYNTHETIC_LIVER, camera, container)

    assert indices.tolist() == [0, 1]
    assert attenuation.shape == (2, 1)
    assert torch.allclose(attenuation[0], torch.ones_like(attenuation[0]), atol=1.0e-6)
    assert 0.0 < float(attenuation[1].item()) < 1.0


def test_empty_slice_render_returns_full_size_density_control_tensors_cpu():
    cfg = _physical_cfg()
    device = torch.device("cpu")
    slice_meta = _slice_meta(device)
    far_means = slice_meta.pixel_pos[:4] + slice_meta.vec_elevational.float() * 100.0

    container = UltraGSBaselineContainer(cfg, device)
    container.init_from_scene({"means": far_means})
    camera = build_virtual_camera(cfg.model.render, SYNTHETIC_LIVER, slice_meta)
    screenspace = torch.zeros_like(container.means, requires_grad=True)

    pred, depth, info = _empty_render(cfg.model, camera, container, screenspace)

    assert pred.shape == (1, SYNTHETIC_LIVER.H, SYNTHETIC_LIVER.W)
    assert depth.shape == (1, SYNTHETIC_LIVER.H, SYNTHETIC_LIVER.W)
    assert info["visibility_filter"].shape == (4,)
    assert info["radii"].shape == (4,)
    assert not bool(info["visibility_filter"].any().item())
