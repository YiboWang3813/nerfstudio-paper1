#!/usr/bin/env python3
"""加载 FiMA PEM checkpoint，导出链式 (N,4,4) probe→world，供 ``scene.loading.poses_abspath`` 使用。"""
from __future__ import annotations

import argparse
import os
import sys

import numpy as np


def _parse_args():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--experiment", default="preproduction/baseline_reproduction/fima/paper_edges", help="Hydra experiment alias used for FiMA training")
    p.add_argument(
        "--pem-output-dir",
        required=True,
        help="PEM 训练输出：末级目录（含 checkpoints/）或训练前传入的上一级目录（无 checkpoints 时再挂 exp_name）",
    )
    p.add_argument(
        "--resume-iter",
        type=int,
        required=True,
        help="与 engine.resume 一致的迭代号，用于加载 iteration_<N>_pose_estimator.pth",
    )
    p.add_argument("--out", required=True, help="输出 .npy 绝对路径 (N,4,4) float32")
    p.add_argument("--seed", type=int, default=None, help="覆盖 system.seed（可选）")
    p.add_argument(
        "--dotlist",
        action="append",
        default=None,
        help="FiMA 配置覆盖项（与 train 子进程一致）；可重复传入，如 --dotlist model.pem.occ_always_two_channels=true",
    )
    return p.parse_args()


def main() -> None:
    args = _parse_args()
    repo = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    if repo not in sys.path:
        sys.path.insert(0, repo)

    from configs.setup import load_hydra_config
    from omegaconf import OmegaConf

    from engine.engine import Engine

    import methods.fima.state  # noqa: F401 — 注册 FiMAOfflineState
    import methods.fima.callbacks  # noqa: F401 — 注册 callbacks

    overrides = [
        f"system.output_dir={args.pem_output_dir}",
        f"engine.resume={args.resume_iter}",
        f"engine.iterations={args.resume_iter}",
        "engine.task=train",
    ]
    if args.seed is not None:
        overrides.append(f"system.seed={int(args.seed)}")
    if args.dotlist:
        overrides.extend(list(args.dotlist))

    cfg = load_hydra_config(args.experiment, overrides)
    cfg.engine.needed_callbacks = OmegaConf.create(["FiMAOfflineSetupCallback", "FiMAOfflineFinishCallback"])

    os.makedirs(os.path.dirname(os.path.abspath(args.out)) or ".", exist_ok=True)

    engine = Engine(cfg)
    engine.trigger("on_setup")

    from methods.fima.pem_pose_export import export_adjacent_chain_poses_numpy

    arr = export_adjacent_chain_poses_numpy(engine.state)
    np.save(args.out, arr)
    print(f"[export_fima_adjacent_chain_poses] Saved (N,4,4) N={arr.shape[0]} -> {args.out}")
    engine.trigger("on_finish")


if __name__ == "__main__":
    main()
