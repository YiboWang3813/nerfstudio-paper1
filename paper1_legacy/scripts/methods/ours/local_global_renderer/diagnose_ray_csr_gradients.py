import argparse
import math

import torch

from ours_renderer import _C
from ours_renderer.grid_geometry import build_slice_grid_pack


class Meta:
    H = 32
    W = 32
    delta_lateral_mm = 0.5
    delta_axial_mm = 0.5
    near_mm = 0.0
    center_lateral = True


def make_case(device: torch.device):
    means = torch.tensor(
        [
            [-2.0, 5.13, 0.15],
            [0.0, 7.23, -0.10],
            [2.0, 9.17, 0.05],
        ],
        device=device,
        dtype=torch.float32,
    )
    scales = torch.tensor(
        [
            [1.10, 1.25, 1.15],
            [1.20, 1.05, 1.30],
            [1.15, 1.35, 1.10],
        ],
        device=device,
        dtype=torch.float32,
    )
    quats = torch.tensor(
        [
            [0.99, 0.03, -0.04, 0.02],
            [0.98, -0.05, 0.06, 0.03],
            [0.97, 0.04, 0.02, -0.06],
        ],
        device=device,
        dtype=torch.float32,
    )
    atten = torch.tensor([0.08, 0.12, 0.10], device=device, dtype=torch.float32)
    pose = torch.eye(4, device=device, dtype=torch.float32)
    grid = build_slice_grid_pack(Meta(), device=device)
    return means, scales, quats, atten, pose, grid


def modulation(means, scales, quats, atten, pose, grid):
    return _C.RayCsrAttenuationCUDA(
        means,
        scales,
        quats,
        atten,
        pose,
        grid,
        32,
        32,
        0.5,
        0.5,
        0.0,
        8,
        1.0,
        0.0,
        2.5,
        0.5,
        1.0,
        0.4,
        1.0e-5,
        0.0,
        2.0,
        3,
        0,
    )


def analytic_backward(means, scales, quats, atten, pose, grid):
    mod, ray_start, ray_end, ray_indices, ray_depths = _C.RayCsrAttenuationForwardCUDA(
        means,
        scales,
        quats,
        atten,
        pose,
        grid,
        32,
        32,
        0.5,
        0.5,
        0.0,
        8,
        1.0,
        0.0,
        2.5,
        0.5,
        1.0,
        0.4,
        1.0e-5,
        0.0,
        2.0,
        3,
        0,
    )
    local = torch.ones_like(mod)
    grad = torch.ones_like(mod)
    d_att, d_means, d_scales, d_quats = _C.RayCsrAttenuationBackwardCUDA(
        means,
        scales,
        quats,
        atten,
        pose,
        grid,
        ray_start,
        ray_end,
        ray_indices,
        ray_depths,
        local,
        mod,
        grad,
        32,
        32,
        0.5,
        0.5,
        0.0,
        8,
        1.0,
        0.0,
        2.5,
        0.5,
        1.0,
        0.4,
        1.0e-5,
        0.0,
        2.0,
        3,
        0,
    )
    return mod, d_att, d_means, d_scales, d_quats


def finite_difference(tensor_name: str, index, eps: float):
    device = torch.device("cuda")
    means, scales, quats, atten, pose, grid = make_case(device)
    base = {"means": means, "scales": scales, "quats": quats, "atten": atten}[tensor_name]
    with torch.no_grad():
        orig = base[index].item()
        base[index] = orig + eps
        plus = modulation(means, scales, quats, atten, pose, grid).sum().item()
        base[index] = orig - eps
        minus = modulation(means, scales, quats, atten, pose, grid).sum().item()
        base[index] = orig
    return (plus - minus) / (2.0 * eps)


def rel_err(a: float, b: float):
    return abs(a - b) / max(1.0e-8, abs(a) + abs(b))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--eps", type=float, default=1.0e-2)
    args = parser.parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required")

    device = torch.device("cuda")
    means, scales, quats, atten, pose, grid = make_case(device)
    mod, d_att, d_means, d_scales, d_quats = analytic_backward(means, scales, quats, atten, pose, grid)
    torch.cuda.synchronize()
    print("modulation stats:", float(mod.min()), float(mod.max()), float(mod.mean()), float(mod.std()))
    print("analytic d_att:", d_att.detach().cpu().tolist())
    print("analytic d_means:", d_means.detach().cpu().tolist())
    print("analytic d_scales:", d_scales.detach().cpu().tolist())
    print("analytic d_quats:", d_quats.detach().cpu().tolist())

    checks = []
    for i in range(atten.numel()):
        ana = float(d_att[i].item())
        num = finite_difference("atten", (i,), args.eps)
        checks.append(("atten", (i,), ana, num, rel_err(ana, num)))
    for idx in [(0, 1), (0, 2), (1, 0), (1, 1), (1, 2), (2, 0), (2, 1), (2, 2)]:
        ana = float(d_means[idx].item())
        num = finite_difference("means", idx, args.eps)
        checks.append(("means", idx, ana, num, rel_err(ana, num)))
    for idx in [(0, 1), (1, 1), (2, 2)]:
        ana = float(d_scales[idx].item())
        num = finite_difference("scales", idx, args.eps)
        checks.append(("scales", idx, ana, num, rel_err(ana, num)))
    for idx in [(0, 2), (1, 1), (2, 3)]:
        ana = float(d_quats[idx].item())
        num = finite_difference("quats", idx, args.eps)
        checks.append(("quats", idx, ana, num, rel_err(ana, num)))

    worst = 0.0
    for name, idx, ana, num, err in checks:
        worst = max(worst, err)
        print(f"{name}{idx}: analytic={ana:.8f} numeric={num:.8f} rel_err={err:.6f}")
    if not math.isfinite(worst) or worst > 0.08:
        raise SystemExit(f"gradient check failed: worst_rel_err={worst:.6f}")
    print(f"gradient check passed: worst_rel_err={worst:.6f}")


if __name__ == "__main__":
    main()
