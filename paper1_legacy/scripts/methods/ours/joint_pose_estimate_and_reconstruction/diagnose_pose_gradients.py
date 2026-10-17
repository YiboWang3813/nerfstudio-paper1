#!/usr/bin/env python3
"""Diagnose Ours renderer pose-gradient strength on a real sweep.

The script intentionally performs only one backward pass. It compares the raw
image-loss gradient on a small per-frame SE(3) pose residual against gradients
on the Gaussian parameters, then probes whether first-order pose / Gaussian
updates are descent directions.
"""

from __future__ import annotations

import argparse
import json
import math
import random
from pathlib import Path
from typing import Dict, Iterable, List, Tuple

import numpy as np
import torch
import torch.nn.functional as F

from configs.setup import load_hydra_config
from methods.ours.gaussian_container import OursGaussContainer
from methods.ours.pose_optimization import _euler_xyz_to_matrix
from methods.ours.render import render
from scene.factory import SceneFactory


def _parse_frame_ids(value: str) -> List[int]:
    return [int(x.strip()) for x in value.split(",") if x.strip()]


def _grad_stats(t: torch.Tensor) -> Dict[str, float]:
    v = t.detach()
    if v.numel() == 0:
        return {"rms": 0.0, "mean_abs": 0.0, "max_abs": 0.0, "l2": 0.0}
    return {
        "rms": float(torch.sqrt(torch.mean(v.float().pow(2))).detach().cpu()),
        "mean_abs": float(v.abs().mean().detach().cpu()),
        "max_abs": float(v.abs().max().detach().cpu()),
        "l2": float(torch.linalg.norm(v.float()).detach().cpu()),
    }


def _clone_raw_params(container: OursGaussContainer) -> Dict[str, torch.Tensor]:
    return {name: wrapper.tensor.detach().clone() for name, wrapper in container._param_dict.items()}


def _restore_raw_params(container: OursGaussContainer, saved: Dict[str, torch.Tensor]) -> None:
    with torch.no_grad():
        for name, value in saved.items():
            container._param_dict[name].tensor.copy_(value)


def _zero_container_grads(container: OursGaussContainer) -> None:
    for wrapper in container._param_dict.values():
        if wrapper.tensor.grad is not None:
            wrapper.tensor.grad = None


def _pose_error(params: torch.Tensor) -> Dict[str, float]:
    rot = torch.linalg.norm(params[:, :3], dim=-1)
    trans = torch.linalg.norm(params[:, 3:], dim=-1)
    return {
        "rot_deg_mean": float(torch.rad2deg(rot).mean().detach().cpu()),
        "trans_mm_mean": float(trans.mean().detach().cpu()),
    }


def _make_loss(
    cfg,
    scene,
    container: OursGaussContainer,
    frame_ids: Iterable[int],
    pose_delta: torch.Tensor,
    loss_kind: str,
) -> torch.Tensor:
    losses = []
    for row, frame_id in enumerate(frame_ids):
        item = scene.train_container.get(int(frame_id))
        delta = _euler_xyz_to_matrix(pose_delta[row])
        pose = item.pose.to(device=container.means.device, dtype=container.means.dtype) @ delta
        pred, _, _ = render(cfg.model, scene.train_container.meta, item, container, pose_override=pose)
        gt = item.image.to(device=pred.device, dtype=pred.dtype)
        if loss_kind == "mse":
            losses.append(F.mse_loss(pred, gt))
        elif loss_kind == "l1":
            losses.append(F.l1_loss(pred, gt))
        else:
            raise ValueError(f"unsupported loss_kind={loss_kind!r}")
    return torch.stack(losses).mean()


def _trial_loss(
    cfg,
    scene,
    container: OursGaussContainer,
    frame_ids: List[int],
    pose_delta: torch.Tensor,
    loss_kind: str,
    saved_raw: Dict[str, torch.Tensor],
    pose_trial: torch.Tensor = None,
    gauss_step_scale: float = 0.0,
) -> float:
    _restore_raw_params(container, saved_raw)
    if gauss_step_scale != 0.0:
        with torch.no_grad():
            for name, wrapper in container._param_dict.items():
                grad = wrapper.tensor.grad
                if grad is None:
                    continue
                lr = float(wrapper.spec.lr)
                wrapper.tensor.add_(grad, alpha=-gauss_step_scale * lr)
    trial_pose = pose_delta if pose_trial is None else pose_trial
    with torch.no_grad():
        loss = _make_loss(cfg, scene, container, frame_ids, trial_pose, loss_kind)
    _restore_raw_params(container, saved_raw)
    return float(loss.detach().cpu())


def _parse_float_list(value: str) -> List[float]:
    return [float(x.strip()) for x in value.split(",") if x.strip()]


def _build_cfg(args: argparse.Namespace):
    overrides = [
        "scene=tus_rec_2024_qc",
        "scene.dataset_name=tus_rec_2024",
        f"scene.loading.sweep_name={args.sweep}",
        f"scene.loading.holdout_num_samples={args.holdout_samples}",
        f"scene.sampler.num_init={args.num_init}",
        f"model.gauss_init.num_points={args.num_init}",
        "model.offline.density_control.timing.start_iter=250",
        "model.offline.density_control.timing.end_iter=800",
        "model.offline.density_control.timing.freq=250",
    ]
    return load_hydra_config(args.experiment, overrides)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--experiment", default="preproduction/ideas/ours/pose_uncertainty_prior/pose_prior")
    parser.add_argument("--sweep", default="010_LH_Per_L_DtP_qc_0005_0370")
    parser.add_argument("--frame-ids", default="40,120,220", help="Comma-separated train frame ids.")
    parser.add_argument("--num-init", type=int, default=100000)
    parser.add_argument("--holdout-samples", type=int, default=20)
    parser.add_argument("--loss-kind", choices=("l1", "mse"), default="l1")
    parser.add_argument(
        "--container-checkpoint",
        type=Path,
        default=None,
        help="Optional Ours container checkpoint. If omitted, use fresh scene initialization.",
    )
    parser.add_argument("--noise-trans-mm", type=float, default=0.5)
    parser.add_argument("--noise-rot-deg", type=float, default=0.0)
    parser.add_argument("--pose-lr-trans", type=float, default=3.0e-4)
    parser.add_argument("--pose-lr-rot", type=float, default=3.0e-5)
    parser.add_argument("--high-pose-lr-trans", type=float, default=1.0e-2)
    parser.add_argument("--gauss-step-scale", type=float, default=1.0)
    parser.add_argument(
        "--normalized-trans-steps-mm",
        default="0.01,0.1,0.5",
        help="Comma-separated fixed-magnitude translation steps along -grad direction.",
    )
    parser.add_argument(
        "--normalized-rot-steps-deg",
        default="0.01,0.1,1.0",
        help="Comma-separated fixed-magnitude rotation steps along -grad direction.",
    )
    parser.add_argument("--seed", type=int, default=20260615)
    parser.add_argument("--out", type=Path, default=None)
    args = parser.parse_args()

    if not torch.cuda.is_available():
        raise SystemExit("diagnose_pose_gradients requires CUDA")

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)

    device = torch.device("cuda:0")
    cfg = _build_cfg(args)
    frame_ids = _parse_frame_ids(args.frame_ids)

    scene = SceneFactory.build_scene(cfg, device)
    container = OursGaussContainer(cfg, device=device)
    if args.container_checkpoint is not None:
        container.load(str(args.container_checkpoint))
        print(f"[diagnose_pose_gradients] loaded container checkpoint: {args.container_checkpoint}")
    else:
        init_data = scene.build_init_gaussians()
        container.init_from_scene(init_data["gaussians"])

    pose_delta = torch.zeros((len(frame_ids), 6), device=device, dtype=container.means.dtype)
    if args.noise_rot_deg != 0.0:
        pose_delta[:, 2] = math.radians(float(args.noise_rot_deg))
    if args.noise_trans_mm != 0.0:
        pose_delta[:, 3] = float(args.noise_trans_mm)
    pose_delta.requires_grad_(True)

    _zero_container_grads(container)
    loss = _make_loss(cfg, scene, container, frame_ids, pose_delta, args.loss_kind)
    loss.backward()

    if pose_delta.grad is None:
        raise RuntimeError("pose_delta.grad is None; pose gradient path is broken")

    saved_raw = _clone_raw_params(container)
    base_loss = float(loss.detach().cpu())
    pose_grad = pose_delta.grad.detach().clone()

    gauss_stats = {}
    for name, wrapper in container._param_dict.items():
        grad = wrapper.tensor.grad
        if grad is None:
            gauss_stats[name] = {"lr": float(wrapper.spec.lr), "has_grad": False}
            continue
        stats = _grad_stats(grad)
        stats["lr"] = float(wrapper.spec.lr)
        stats["lr_scaled_rms"] = stats["rms"] * float(wrapper.spec.lr)
        stats["lr_scaled_max_abs"] = stats["max_abs"] * float(wrapper.spec.lr)
        gauss_stats[name] = {"has_grad": True, **stats}

    rot_grad = pose_grad[:, :3]
    trans_grad = pose_grad[:, 3:]
    pose_stats = {
        "all": _grad_stats(pose_grad),
        "rot": _grad_stats(rot_grad),
        "trans": _grad_stats(trans_grad),
        "lr_scaled_rot_rms": _grad_stats(rot_grad)["rms"] * float(args.pose_lr_rot),
        "lr_scaled_trans_rms": _grad_stats(trans_grad)["rms"] * float(args.pose_lr_trans),
        "high_lr_scaled_trans_rms": _grad_stats(trans_grad)["rms"] * float(args.high_pose_lr_trans),
    }

    correct_trans_update = -pose_delta.detach()[:, 3:].reshape(-1)
    actual_trans_update = -trans_grad.reshape(-1)
    denom = torch.linalg.norm(correct_trans_update) * torch.linalg.norm(actual_trans_update)
    if float(denom.detach().cpu()) > 0:
        trans_update_cosine_to_gt = float((correct_trans_update @ actual_trans_update / denom).detach().cpu())
    else:
        trans_update_cosine_to_gt = 0.0
    per_frame_trans_cos = []
    for row in range(pose_delta.shape[0]):
        correct = -pose_delta.detach()[row, 3:]
        actual = -trans_grad[row]
        row_denom = torch.linalg.norm(correct) * torch.linalg.norm(actual)
        if float(row_denom.detach().cpu()) > 0:
            per_frame_trans_cos.append(float((correct @ actual / row_denom).detach().cpu()))
        else:
            per_frame_trans_cos.append(0.0)

    pose_trial_std = pose_delta.detach().clone()
    pose_trial_std[:, :3] -= float(args.pose_lr_rot) * rot_grad
    pose_trial_std[:, 3:] -= float(args.pose_lr_trans) * trans_grad
    pose_trial_high = pose_delta.detach().clone()
    pose_trial_high[:, :3] -= float(args.pose_lr_rot) * rot_grad
    pose_trial_high[:, 3:] -= float(args.high_pose_lr_trans) * trans_grad

    pose_std_loss = _trial_loss(
        cfg, scene, container, frame_ids, pose_delta.detach(), args.loss_kind, saved_raw, pose_trial=pose_trial_std
    )
    pose_high_loss = _trial_loss(
        cfg, scene, container, frame_ids, pose_delta.detach(), args.loss_kind, saved_raw, pose_trial=pose_trial_high
    )
    gauss_loss = _trial_loss(
        cfg,
        scene,
        container,
        frame_ids,
        pose_delta.detach(),
        args.loss_kind,
        saved_raw,
        gauss_step_scale=float(args.gauss_step_scale),
    )
    joint_loss = _trial_loss(
        cfg,
        scene,
        container,
        frame_ids,
        pose_delta.detach(),
        args.loss_kind,
        saved_raw,
        pose_trial=pose_trial_std,
        gauss_step_scale=float(args.gauss_step_scale),
    )

    normalized_trans_losses = {}
    trans_norm = torch.linalg.norm(trans_grad, dim=-1, keepdim=True).clamp_min(1e-12)
    trans_dir = -trans_grad / trans_norm
    for step_mm in _parse_float_list(args.normalized_trans_steps_mm):
        trial = pose_delta.detach().clone()
        trial[:, 3:] = trial[:, 3:] + float(step_mm) * trans_dir
        normalized_trans_losses[f"{step_mm:g}mm"] = _trial_loss(
            cfg, scene, container, frame_ids, pose_delta.detach(), args.loss_kind, saved_raw, pose_trial=trial
        )

    normalized_rot_losses = {}
    rot_norm = torch.linalg.norm(rot_grad, dim=-1, keepdim=True).clamp_min(1e-12)
    rot_dir = -rot_grad / rot_norm
    for step_deg in _parse_float_list(args.normalized_rot_steps_deg):
        trial = pose_delta.detach().clone()
        trial[:, :3] = trial[:, :3] + math.radians(float(step_deg)) * rot_dir
        normalized_rot_losses[f"{step_deg:g}deg"] = _trial_loss(
            cfg, scene, container, frame_ids, pose_delta.detach(), args.loss_kind, saved_raw, pose_trial=trial
        )

    result = {
        "experiment": args.experiment,
        "sweep": args.sweep,
        "frame_ids": frame_ids,
        "num_init": int(args.num_init),
        "container_checkpoint": str(args.container_checkpoint) if args.container_checkpoint is not None else "",
        "loss_kind": args.loss_kind,
        "base_loss": base_loss,
        "initial_pose_error": _pose_error(pose_delta.detach()),
        "pose_grad": pose_stats,
        "pose_trans_update_cosine_to_gt_correction": trans_update_cosine_to_gt,
        "pose_trans_update_cosine_to_gt_correction_per_frame": per_frame_trans_cos,
        "gauss_grad": gauss_stats,
        "trial_losses": {
            "pose_sgd_standard_lr": pose_std_loss,
            "pose_sgd_high_trans_lr": pose_high_loss,
            "gauss_sgd_config_lr": gauss_loss,
            "joint_sgd_config_lr": joint_loss,
            "pose_normalized_trans": normalized_trans_losses,
            "pose_normalized_rot": normalized_rot_losses,
        },
        "trial_loss_delta": {
            "pose_sgd_standard_lr": pose_std_loss - base_loss,
            "pose_sgd_high_trans_lr": pose_high_loss - base_loss,
            "gauss_sgd_config_lr": gauss_loss - base_loss,
            "joint_sgd_config_lr": joint_loss - base_loss,
            "pose_normalized_trans": {k: v - base_loss for k, v in normalized_trans_losses.items()},
            "pose_normalized_rot": {k: v - base_loss for k, v in normalized_rot_losses.items()},
        },
    }

    text = json.dumps(result, indent=2, sort_keys=True)
    print(text)
    if args.out is not None:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(text + "\n", encoding="utf-8")
        print(f"[diagnose_pose_gradients] wrote {args.out}")


if __name__ == "__main__":
    main()
