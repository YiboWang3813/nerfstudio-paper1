"""Deterministic tiny pre/post renderer evidence. Not a paper experiment."""
import argparse
from pathlib import Path
from types import SimpleNamespace

import torch
from ours_renderer.types import OursRenderConfig
from ours_renderer.autograd import render_projected_slice
from ours_renderer.grid_geometry import build_slice_grid_pack
from ours_renderer.us_focus_reference import render_us_focus_reference


def sample(device, mode="adaptive"):
    cfg = OursRenderConfig(H=16, W=16, lateral_spacing_mm_per_pix=.5,
        axial_spacing_mm_per_pix=.5, axial_near_mm=0., tile_size_px=4, k_sigma=2.5,
        mapper_sort_by_v=False, shader_weight_mode="us_focus", compositing_mode="additive",
        us_focus_sigma_ele_0=1., us_focus_sigma_ele_1=.01,
        us_focus_sigma_ele_min=.8, us_focus_sigma_ele_max=5.,
        ray_attenuation_enabled=True, ray_attenuation_source="attenuation",
        ray_attenuation_mapper_kind="ray_csr", ray_attenuation_beta=.8,
        ray_attenuation_strength=.55, ray_attenuation_min_transmission=.35,
        ray_attenuation_candidate_k_sigma=2.)
    if hasattr(cfg, "local_conditioning_mode"):
        cfg.local_conditioning_mode = mode
    means = torch.tensor([[0., 2., .3], [1., 4., -.5], [-1., 5., 1.4]], device=device)
    scales = torch.tensor([[.8, .9, .7], [.7, .8, .9], [1.1, .6, .8]], device=device)
    quats = torch.tensor([[1., .1, .2, .0], [1., .2, -.1, .1], [1., -.1, .0, .1]], device=device)
    quats = torch.nn.functional.normalize(quats, dim=-1)
    alpha = torch.tensor([.08, .1, .12], device=device)
    tau = torch.tensor([.016, .02, .024], device=device)
    pose = torch.eye(4, device=device)
    for x in (means, scales, quats, alpha, tau, pose):
        x.requires_grad_(True)
    intensity = torch.ones_like(alpha)
    return cfg, means, scales, quats, alpha, intensity, tau, pose


def capture():
    p = argparse.ArgumentParser()
    p.add_argument("output", type=Path)
    args = p.parse_args()
    result = {}
    cfg, m, s, q, a, i, tau, pose = sample("cpu")
    result["reference"] = render_us_focus_reference(cfg, m, s, q, a, i, pose).detach()
    assert torch.cuda.is_available(), "P0 probe must execute on a GPU compute node"
    for repeat in range(2):
        cfg, m, s, q, a, i, tau, pose = sample("cuda")
        meta = SimpleNamespace(H=16, W=16, delta_lateral_mm=.5, delta_axial_mm=.5,
                               near_mm=0., center_lateral=True)
        grid = build_slice_grid_pack(meta, device=torch.device("cuda"))
        pred, path, support = render_projected_slice(cfg, m, s, q, a, i, pose, grid,
                                                    tau, collect_path_stats=True)
        pred.square().sum().backward()
        for name, value in dict(pred=pred, path=path, support=support,
                grad_means=m.grad, grad_scales=s.grad, grad_quats=q.grad,
                grad_alpha=a.grad, grad_tau=tau.grad, grad_pose=pose.grad).items():
            assert value is not None and torch.isfinite(value).all(), name
            result[f"{repeat}/{name}"] = value.detach().cpu()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    torch.save(result, args.output)
    print("probe saved", args.output, torch.cuda.get_device_name())


if __name__ == "__main__":
    capture()
