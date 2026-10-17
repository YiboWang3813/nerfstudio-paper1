"""Pooled synthetic_liver FiMA PEM training.

This is the Level-A path: load full-resolution sweep images, do learnable early
downsampling inside the FiMA stem, and train on ordered train-frame subsequences
from all six synthetic_liver sweeps without exposing holdout test images.
"""
from __future__ import annotations

import argparse
import os
import random
import time
from dataclasses import asdict
from typing import Dict, List, Sequence, Tuple

import numpy as np
import torch
from tqdm import tqdm

from methods.fima.configs import FiMAConfig
from methods.fima.losses import heteroscedastic_pose_loss, pem_supervision_loss
from methods.fima.pose_estimator import FiMAPoseEstimator
from methods.fima.pose_geometry import (
    adjacent_frame_transform_gt,
    se3_from_6dof,
    six_dof_from_matrix,
    translation_rotation_errors,
)


DEFAULT_SWEEPS = ("l1", "l2", "l3", "r1", "r2", "r3")


def _holdout_indices(n: int, trim_each: int, num_samples: int) -> List[int]:
    picks = np.linspace(trim_each, n - 1 - trim_each, num_samples, dtype=np.float64)
    return [int(x) for x in np.rint(picks).astype(np.int64).tolist()]


def _load_image(path: str) -> torch.Tensor:
    arr = np.load(path).astype(np.float32, copy=False)
    t = torch.from_numpy(arr)
    if float(t.max()) > 2.0:
        t = t / 255.0
    return t


class PooledSyntheticLiverWindows:
    def __init__(
        self,
        root: str,
        sweeps: Sequence[str],
        *,
        image_name: str,
        window: int,
        trim_each: int,
        holdout_num_samples: int,
        device: torch.device,
    ) -> None:
        self.window = int(window)
        self.device = device
        self.images: List[torch.Tensor] = []
        self.poses: List[torch.Tensor] = []
        self.train_masks: List[torch.Tensor] = []
        self.test_masks: List[torch.Tensor] = []
        self.train_windows: List[Tuple[int, List[int]]] = []
        self.test_windows: List[Tuple[int, List[int]]] = []
        self.sweeps = list(sweeps)
        self.split_summary: Dict[str, Tuple[int, int, int]] = {}

        for si, sweep in enumerate(self.sweeps):
            d = os.path.join(root, sweep)
            images = _load_image(os.path.join(d, image_name)).to(device=device, dtype=torch.float32)
            poses = torch.from_numpy(np.load(os.path.join(d, "relative_poses.npy")).astype(np.float32, copy=False)).to(
                device=device, dtype=torch.float32
            )
            if images.shape[0] != poses.shape[0]:
                raise ValueError(f"{sweep}: image/pose length mismatch {images.shape[0]} vs {poses.shape[0]}")
            self.images.append(images)
            self.poses.append(poses)
            holdout = set(_holdout_indices(int(images.shape[0]), trim_each, holdout_num_samples))
            train_mask = torch.ones(int(images.shape[0]), device=device, dtype=torch.bool)
            if holdout:
                train_mask[torch.tensor(sorted(holdout), device=device, dtype=torch.long)] = False
            test_mask = ~train_mask
            self.train_masks.append(train_mask)
            self.test_masks.append(test_mask)
            train_count = int(train_mask.sum().item())
            test_count = int(test_mask.sum().item())
            self.split_summary[sweep] = (int(images.shape[0]), train_count, test_count)
            windows = self._ordered_windows(list(range(int(images.shape[0]))))
            self.train_windows.extend((si, ids) for ids in windows if self._edge_mask_for(si, ids, "train").any())
            self.test_windows.extend((si, ids) for ids in windows if self._edge_mask_for(si, ids, "test").any())
        if not self.train_windows:
            raise RuntimeError("No pooled FiMA train windows built.")

        edge_targets = []
        for si, ids in self.train_windows:
            target = self._target_for(si, ids)
            mask = self._edge_mask_for(si, ids, "train")
            edge_targets.append(target[mask].reshape(-1, 6))
        flat = torch.cat(edge_targets, dim=0)
        self.target_mean = flat.mean(dim=0)
        std = flat.std(dim=0, unbiased=False).clamp_min(1.0e-6)
        std[:3] = std[:3].clamp_min(1.0e-2)
        std[3:] = std[3:].clamp_min(1.0e-4)
        self.target_std = std

    def _ordered_windows(self, ids: Sequence[int]) -> List[List[int]]:
        if len(ids) < self.window:
            return []
        return [list(ids[pos - self.window + 1 : pos + 1]) for pos in range(self.window - 1, len(ids))]

    def _target_for(self, sweep_idx: int, ids: Sequence[int]) -> torch.Tensor:
        poses = self.poses[int(sweep_idx)]
        edges = []
        for k in range(1, len(ids)):
            edge = adjacent_frame_transform_gt(poses[int(ids[k - 1])], poses[int(ids[k])])
            edges.append(six_dof_from_matrix(edge))
        return torch.stack(edges, dim=0)

    def _edge_mask_for(self, sweep_idx: int, ids: Sequence[int], split: str) -> torch.Tensor:
        right = torch.tensor([int(x) for x in ids[1:]], device=self.device, dtype=torch.long)
        left = torch.tensor([int(x) for x in ids[:-1]], device=self.device, dtype=torch.long)
        if split == "train":
            m = self.train_masks[int(sweep_idx)]
            return m[left] & m[right]
        if split == "test":
            return self.test_masks[int(sweep_idx)][right]
        raise ValueError(f"Unknown split: {split}")

    def sample(self, split: str, batch_size: int) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        pool = self.train_windows if split == "train" else self.test_windows
        if not pool:
            pool = self.train_windows
        picks = random.choices(pool, k=int(batch_size))
        images = []
        targets = []
        masks = []
        for si, ids in picks:
            img = self.images[int(si)][torch.tensor(ids, device=self.device, dtype=torch.long)]
            if split == "train":
                keep = self.train_masks[int(si)][torch.tensor(ids, device=self.device, dtype=torch.long)]
                img = img * keep.to(dtype=img.dtype).reshape(-1, 1, 1)
            images.append(img.unsqueeze(1))
            targets.append(self._target_for(int(si), ids))
            masks.append(self._edge_mask_for(int(si), ids, split))
        return torch.stack(images, dim=0), torch.stack(targets, dim=0), torch.stack(masks, dim=0)

    def eval_windows(self, max_windows: int) -> List[Tuple[int, List[int]]]:
        pool = self.test_windows if self.test_windows else self.train_windows
        if len(pool) <= max_windows:
            return list(pool)
        idx = np.linspace(0, len(pool) - 1, int(max_windows), dtype=np.int64)
        return [pool[int(i)] for i in idx.tolist()]


def _make_cfg(args: argparse.Namespace) -> FiMAConfig:
    cfg = FiMAConfig()
    cfg.pem.input_mode = "multiframe"
    cfg.pem.train_protocol = "standard"
    cfg.pem.window_size = int(args.window_size)
    cfg.pem.pem_output_layout = "window_edges"
    cfg.pem.batch_size = int(args.batch_size)
    cfg.pem.d_model = int(args.d_model)
    cfg.pem.num_mixer_layers = int(args.layers)
    cfg.pem.stem_channels = [int(x) for x in str(args.stem_channels).split(",") if x]
    cfg.pem.input_downsample_stride = int(args.input_downsample_stride)
    cfg.pem.input_downsample_channels = int(args.input_downsample_channels)
    cfg.pem.prefer_mamba = True
    cfg.pem.require_mamba = True
    cfg.pem.predict_uncertainty = bool(getattr(args, "predict_uncertainty", False))
    cfg.pem.lr = float(args.lr)
    cfg.pem.weight_decay = float(args.weight_decay)
    cfg.pem.lr_decay_steps = int(args.lr_decay_steps)
    cfg.pem.lr_decay_gamma = float(args.lr_decay_gamma)
    cfg.pem_loss.use_l1 = True
    cfg.pem_loss.use_mse = False
    cfg.pem_loss.use_pearson = True
    cfg.pem_loss.lambda_l1 = 1.0
    cfg.pem_loss.lambda_pearson = float(args.lambda_pearson)
    cfg.pem_loss.pearson_mode = "per_channel"
    cfg.pem_loss.standardize_targets = True
    cfg.pem_loss.use_heteroscedastic_nll = bool(getattr(args, "predict_uncertainty", False))
    cfg.pem_loss.lambda_heteroscedastic = float(getattr(args, "lambda_heteroscedastic", 0.05))
    return cfg


def _init_head(model: FiMAPoseEstimator, mean: torch.Tensor) -> None:
    with torch.no_grad():
        model.head.weight.zero_()
        model.head.bias.copy_(mean.to(device=model.head.bias.device, dtype=model.head.bias.dtype))


@torch.no_grad()
def _evaluate(
    model: FiMAPoseEstimator,
    data: PooledSyntheticLiverWindows,
    cfg: FiMAConfig,
    target_mean: torch.Tensor,
    target_std: torch.Tensor,
    max_windows: int,
) -> Dict[str, float]:
    model.eval()
    losses = []
    trans_errs = []
    rot_errs = []
    for si, ids in data.eval_windows(max_windows):
        images = data.images[int(si)][torch.tensor(ids, device=data.device, dtype=torch.long)].unsqueeze(0).unsqueeze(2)
        target = data._target_for(int(si), ids).unsqueeze(0)
        mask = data._edge_mask_for(int(si), ids, "test").unsqueeze(0)
        out = model(images)
        loss, _ = pem_supervision_loss(
            out["pred_pose_6"],
            target,
            cfg.pem_loss,
            target_mean=target_mean,
            target_std=target_std,
            valid_mask=mask,
        )
        losses.append(float(loss.item()))
        pred = out["pred_pose_6"].squeeze(0)[mask.squeeze(0)]
        tgt = target.squeeze(0)[mask.squeeze(0)]
        pred_m = torch.stack([se3_from_6dof(v) for v in pred], dim=0)
        tgt_m = torch.stack([se3_from_6dof(v) for v in tgt], dim=0)
        dt, dr = translation_rotation_errors(pred_m, tgt_m)
        trans_errs.extend([float(x) for x in dt.detach().cpu().tolist()])
        rot_errs.extend([float(torch.rad2deg(x).item()) for x in dr.detach().cpu()])
    model.train()
    return {
        "loss": float(np.mean(losses)) if losses else float("nan"),
        "edge_trans_mm": float(np.mean(trans_errs)) if trans_errs else float("nan"),
        "edge_rot_deg": float(np.mean(rot_errs)) if rot_errs else float("nan"),
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default="/home/boywang25/ultrasound-scene-reconstruction/data/synthetic_liver")
    ap.add_argument("--sweeps", default="l1,l2,l3,r1,r2,r3")
    ap.add_argument("--image-name", default="images.npy")
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--iterations", type=int, default=40000)
    ap.add_argument("--window-size", type=int, default=32)
    ap.add_argument("--batch-size", type=int, default=1)
    ap.add_argument("--d-model", type=int, default=160)
    ap.add_argument("--layers", type=int, default=6)
    ap.add_argument("--stem-channels", default="32,64,128")
    ap.add_argument("--input-downsample-stride", type=int, default=4)
    ap.add_argument("--input-downsample-channels", type=int, default=32)
    ap.add_argument("--lr", type=float, default=2.0e-4)
    ap.add_argument("--weight-decay", type=float, default=1.0e-4)
    ap.add_argument("--lr-decay-steps", type=int, default=6000)
    ap.add_argument("--lr-decay-gamma", type=float, default=0.5)
    ap.add_argument("--lambda-pearson", type=float, default=1.0)
    ap.add_argument("--predict-uncertainty", action="store_true")
    ap.add_argument("--lambda-heteroscedastic", type=float, default=0.05)
    ap.add_argument("--log-every", type=int, default=200)
    ap.add_argument("--eval-every", type=int, default=2000)
    ap.add_argument("--save-every", type=int, default=10000)
    ap.add_argument("--eval-windows", type=int, default=120)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for pooled FiMA training.")
    device = torch.device("cuda")
    torch.manual_seed(int(args.seed))
    np.random.seed(int(args.seed))
    random.seed(int(args.seed))

    cfg = _make_cfg(args)
    sweeps = [x.strip() for x in str(args.sweeps).split(",") if x.strip()]
    data = PooledSyntheticLiverWindows(
        args.root,
        sweeps,
        image_name=str(args.image_name),
        window=int(args.window_size),
        trim_each=10,
        holdout_num_samples=40,
        device=device,
    )
    model = FiMAPoseEstimator(cfg).to(device)
    _init_head(model, data.target_mean)
    opt = torch.optim.AdamW(model.parameters(), lr=float(args.lr), weight_decay=float(args.weight_decay))
    scheduler = torch.optim.lr_scheduler.StepLR(
        opt,
        step_size=max(1, int(args.lr_decay_steps)),
        gamma=float(args.lr_decay_gamma),
    )
    target_mean = data.target_mean.detach()
    target_std = data.target_std.detach()

    print(f"[FiMA pooled] device={torch.cuda.get_device_name(0)} backend={model.mixer.backend_name}")
    print(f"[FiMA pooled] sweeps={sweeps} split={data.split_summary}")
    print(
        f"[FiMA pooled] train_windows={len(data.train_windows)} test_windows={len(data.test_windows)} "
        f"image={args.image_name} input_downsample_stride={args.input_downsample_stride}"
    )
    print(
        "[FiMA pooled] target standardization:"
        f" mean={target_mean.detach().cpu().tolist()} std={target_std.detach().cpu().tolist()}"
    )

    log_path = os.path.join(args.out_dir, "train_metrics.tsv")
    with open(log_path, "w", encoding="utf-8") as f:
        f.write("iter\tlr\ttrain_total\ttrain_l1_trans\ttrain_l1_rot\ttrain_pearson\tval_loss\tval_edge_trans_mm\tval_edge_rot_deg\tit_per_s\n")

    pbar = tqdm(range(1, int(args.iterations) + 1), dynamic_ncols=True)
    t0 = time.perf_counter()
    last_parts: Dict[str, torch.Tensor] = {}
    for it in pbar:
        images, target, edge_mask = data.sample("train", int(args.batch_size))
        out = model(images)
        if bool(getattr(cfg.pem_loss, "use_heteroscedastic_nll", False)):
            base_loss, base_parts = pem_supervision_loss(
                out["pred_pose_6"],
                target,
                cfg.pem_loss,
                target_mean=target_mean,
                target_std=target_std,
                valid_mask=edge_mask,
            )
            nll_loss, nll_parts = heteroscedastic_pose_loss(
                out["pred_pose_6"],
                out["pred_logvar_2"],
                target,
                cfg.pem_loss,
                target_mean=target_mean,
                target_std=target_std,
                valid_mask=edge_mask,
                logvar_min=float(cfg.pem.uncertainty_logvar_min),
                logvar_max=float(cfg.pem.uncertainty_logvar_max),
            )
            loss = base_loss + nll_loss
            parts = dict(base_parts)
            parts.update({k: v for k, v in nll_parts.items() if k != "total"})
            parts["total"] = loss
        else:
            loss, parts = pem_supervision_loss(
                out["pred_pose_6"],
                target,
                cfg.pem_loss,
                target_mean=target_mean,
                target_std=target_std,
                valid_mask=edge_mask,
            )
        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()
        scheduler.step()
        last_parts = parts

        if it % int(args.log_every) == 0 or it == 1:
            dt = time.perf_counter() - t0
            it_s = float(it) / max(dt, 1e-9)
            pbar.set_description(f"loss={float(parts['total'].item()):.4f} it/s={it_s:.1f}")
            val = {"loss": float("nan"), "edge_trans_mm": float("nan"), "edge_rot_deg": float("nan")}
            if it % int(args.eval_every) == 0 or it == 1:
                val = _evaluate(model, data, cfg, target_mean, target_std, int(args.eval_windows))
                print(
                    f"[EVAL] iter={it} loss={val['loss']:.5f} "
                    f"edge_trans_mm={val['edge_trans_mm']:.5f} edge_rot_deg={val['edge_rot_deg']:.5f}"
                )
            pearson = parts.get("pearson_mean", parts.get("pearson", torch.tensor(float("nan"))))
            with open(log_path, "a", encoding="utf-8") as f:
                f.write(
                    f"{it}\t{float(opt.param_groups[0]['lr']):.8e}\t{float(parts['total'].item()):.8f}\t"
                    f"{float(parts.get('l1_trans', torch.tensor(float('nan'))).item()):.8f}\t"
                    f"{float(parts.get('l1_rot', torch.tensor(float('nan'))).item()):.8f}\t"
                    f"{float(pearson.item()):.8f}\t{val['loss']:.8f}\t"
                    f"{val['edge_trans_mm']:.8f}\t{val['edge_rot_deg']:.8f}\t{it_s:.4f}\n"
                )

        if int(args.save_every) > 0 and it % int(args.save_every) == 0:
            ckpt_path = os.path.join(args.out_dir, f"checkpoint_{it:06d}.pt")
            torch.save(
                {
                    "iteration": int(it),
                    "model": model.state_dict(),
                    "optimizer": opt.state_dict(),
                    "target_mean": target_mean.detach().cpu(),
                    "target_std": target_std.detach().cpu(),
                    "args": vars(args),
                    "cfg": asdict(cfg),
                },
                ckpt_path,
            )
            print(f"[SAVE] {ckpt_path}")

    final_path = os.path.join(args.out_dir, "checkpoint_final.pt")
    torch.save(
        {
            "iteration": int(args.iterations),
            "model": model.state_dict(),
            "optimizer": opt.state_dict(),
            "target_mean": target_mean.detach().cpu(),
            "target_std": target_std.detach().cpu(),
            "args": vars(args),
            "cfg": asdict(cfg),
            "last_train_parts": {k: float(v.item()) for k, v in last_parts.items()},
        },
        final_path,
    )
    print(f"[DONE] final checkpoint: {final_path}")


if __name__ == "__main__":
    main()
