#!/usr/bin/env python
"""Analyze per-ray candidate capacity for Ours ray-CSR attenuation."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch
from omegaconf import OmegaConf

PROJECT_DIR = Path(__file__).resolve().parents[3]
if str(PROJECT_DIR) not in sys.path:
    sys.path.insert(0, str(PROJECT_DIR))
for path in [PROJECT_DIR / "ops" / "ours_renderer", PROJECT_DIR / "ops" / "gaussian_renderer"]:
    if path.is_dir() and str(path) not in sys.path:
        sys.path.insert(0, str(path))

from configs.setup import prepare_config  # noqa: E402
from methods.ours.gaussian_container import OursGaussContainer  # noqa: E402
from methods.ours.render import build_ours_render_config, build_slice_grid_geometry  # noqa: E402
from scene.factory import SceneFactory  # noqa: E402
import ours_renderer._C as _C  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", required=True, type=Path)
    parser.add_argument("--iteration", required=True, type=int)
    parser.add_argument("--ray-candidate-k-sigma", default=2.0, type=float)
    parser.add_argument("--ray-beta", default=0.60, type=float)
    parser.add_argument("--ray-strength", default=0.30, type=float)
    parser.add_argument("--ray-min-transmission", default=0.55, type=float)
    parser.add_argument("--frames", default="all", help="'all' or comma-separated train frame indices.")
    parser.add_argument("--out-json", required=True, type=Path)
    parser.add_argument("--device", default="cuda")
    return parser.parse_args()


def parse_frames(raw: str, n_frames: int) -> list[int]:
    if raw.strip().lower() == "all":
        return list(range(n_frames))
    return [int(x.strip()) for x in raw.split(",") if x.strip()]


def main() -> None:
    args = parse_args()
    device = torch.device(args.device)
    cfg = prepare_config(OmegaConf.load(args.run_dir / "resolved_config.yaml"))
    sr = cfg.model.slice_render
    sr.ray_attenuation_enabled = True
    sr.ray_attenuation_source = "attenuation"
    sr.ray_attenuation_mapper_kind = "ray_csr"
    sr.ray_attenuation_formula = "slice_depth"
    sr.ray_attenuation_candidate_k_sigma = float(args.ray_candidate_k_sigma)

    scene = SceneFactory.build_scene(cfg, device=device)
    container = OursGaussContainer(cfg, device=device)
    container.load(args.run_dir / f"checkpoints/iteration_{int(args.iteration)}_container.pth")
    r_cfg = build_ours_render_config(scene.dataset_meta, cfg.model)
    grid = build_slice_grid_geometry(scene.dataset_meta, container.means.device).float().contiguous()
    frames = parse_frames(args.frames, len(scene.train_container))

    rows = []
    max_count = 0
    max_frame = None
    for frame in frames:
        slice_meta = scene.train_container.get(int(frame))
        pose = slice_meta.pose.to(device=container.means.device, dtype=torch.float32).contiguous()
        _, ray_start, ray_end, ray_indices, _ = _C.RayCsrAttenuationForwardCUDA(
            container.means.float().contiguous(),
            container.scales.float().contiguous(),
            container.quats.float().contiguous(),
            container.attenuations.float().contiguous(),
            pose,
            grid,
            int(r_cfg.H),
            int(r_cfg.W),
            float(r_cfg.lateral_spacing_mm_per_pix),
            float(r_cfg.axial_spacing_mm_per_pix),
            float(r_cfg.axial_near_mm),
            int(r_cfg.tile_size_px),
            float(r_cfg.elev_corridor_half_mm),
            float(r_cfg.slab_half_thickness_mm),
            float(args.ray_candidate_k_sigma),
            float(args.ray_beta),
            float(args.ray_strength),
            float(args.ray_min_transmission),
            float(sr.ray_attenuation_weight_min),
            float(sr.ray_attenuation_axial_margin_mm),
            float(sr.ray_attenuation_quadrature_window_mm),
            1,
            1,
        )
        counts = (ray_end - ray_start).clamp_min(0).detach().cpu().numpy()
        total = int(ray_indices.numel())
        frame_max = int(counts.max()) if counts.size else 0
        if frame_max > max_count:
            max_count = frame_max
            max_frame = int(frame)
        rows.append(
            {
                "frame": int(frame),
                "total_candidates": total,
                "max_per_ray": frame_max,
                "p95_per_ray": float(np.percentile(counts, 95)) if counts.size else 0.0,
                "p99_per_ray": float(np.percentile(counts, 99)) if counts.size else 0.0,
                "p999_per_ray": float(np.percentile(counts, 99.9)) if counts.size else 0.0,
                "nonzero_rays": int((counts > 0).sum()),
            }
        )

    out = {
        "run_dir": str(args.run_dir),
        "iteration": int(args.iteration),
        "num_frames": len(frames),
        "num_gaussians": int(container.means.shape[0]),
        "H": int(r_cfg.H),
        "W": int(r_cfg.W),
        "ray_candidate_k_sigma": float(args.ray_candidate_k_sigma),
        "max_per_ray": int(max_count),
        "max_frame": max_frame,
        "rows": rows,
    }
    args.out_json.parent.mkdir(parents=True, exist_ok=True)
    args.out_json.write_text(json.dumps(out, indent=2), encoding="utf-8")
    print(json.dumps({k: v for k, v in out.items() if k != "rows"}, indent=2))


if __name__ == "__main__":
    main()
