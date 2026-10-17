import argparse
from pathlib import Path

import numpy as np
import torch

from ours_renderer import _C
from ours_renderer.grid_geometry import build_slice_grid_pack


class TusRec2024Meta:
    name = "tus_rec_2024"
    probe_type = "linear"
    H = 288
    W = 384
    near_mm = 0.0
    far_mm = 480 * 0.220979690551758
    width_mm = 640 * 0.229389190673828
    delta_lateral_mm = width_mm / W
    delta_axial_mm = far_mm / H
    radius_mm = 0.0
    fov_deg = 0.0
    center_lateral = False


TUS_REC_2024 = TusRec2024Meta()


def activate_checkpoint(ckpt, device):
    means = ckpt["means"].to(device=device, dtype=torch.float32).contiguous()
    scale_min = 0.05
    scale_max = 3.0
    scales = (scale_min + (scale_max - scale_min) * torch.sigmoid(ckpt["scales"].to(device))).float().contiguous()
    quats = torch.nn.functional.normalize(ckpt["quats"].to(device=device, dtype=torch.float32), dim=-1).contiguous()
    atten = torch.sigmoid(ckpt["attenuations"].to(device=device, dtype=torch.float32)).contiguous()
    echo = (0.2 * torch.sigmoid(ckpt["echo_alphas"].to(device=device, dtype=torch.float32))).contiguous()
    return means, scales, quats, atten, echo


def summarize(name, x):
    x = x.detach().float()
    print(
        f"{name}: min={float(x.min()):.6f} max={float(x.max()):.6f} "
        f"mean={float(x.mean()):.6f} std={float(x.std()):.6f}"
    )


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument(
        "--data-dir",
        default="/home/boywang25/ultrasound-scene-reconstruction/data/tus_rec_2024_qc/010_LH_Per_L_DtP_qc_0005_0370",
    )
    ap.add_argument("--num-frames", type=int, default=5)
    args = ap.parse_args()

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required")
    device = torch.device("cuda")
    ckpt = torch.load(args.checkpoint, map_location="cpu")
    means, scales, quats, atten, echo = activate_checkpoint(ckpt, device)
    grid = build_slice_grid_pack(TUS_REC_2024, device=device)
    poses_np = np.load(str(Path(args.data_dir) / "poses.npy"))

    summarize("echo_alpha", echo)
    summarize("attenuation", atten)
    print("num_gaussians:", int(means.shape[0]))

    mods = []
    for i in range(min(args.num_frames, poses_np.shape[0])):
        pose = torch.as_tensor(poses_np[i], device=device, dtype=torch.float32).contiguous()
        mod = _C.RayCsrAttenuationCUDA(
            means,
            scales,
            quats,
            atten,
            pose,
            grid,
            int(TUS_REC_2024.H),
            int(TUS_REC_2024.W),
            float(TUS_REC_2024.delta_lateral_mm),
            float(TUS_REC_2024.delta_axial_mm),
            float(TUS_REC_2024.near_mm),
            8,
            0.202,
            0.0,
            2.5,
            0.2,
            0.08,
            0.8,
            1.0e-4,
            0.0,
            2.0,
            3,
            0,
        )
        torch.cuda.synchronize()
        mods.append(mod.detach())
        near_one = (mod > 0.999).float().mean()
        near_floor = (mod <= 0.840001).float().mean()
        print(
            f"frame {i}: min={float(mod.min()):.6f} max={float(mod.max()):.6f} "
            f"mean={float(mod.mean()):.6f} std={float(mod.std()):.6f} "
            f"near_one={float(near_one):.4f} near_floor={float(near_floor):.4f}"
        )
    all_mod = torch.stack(mods, dim=0)
    summarize("all_modulation", all_mod)


if __name__ == "__main__":
    main()
