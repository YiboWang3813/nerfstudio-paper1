#!/usr/bin/env python3
"""Warmup + timed ours offline train step (render + loss + backward + optimizer)."""
from __future__ import annotations

import os
import sys
import time
import argparse

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))
sys.path.insert(0, ROOT)
os.chdir(ROOT)
os.environ["WANDB_MODE"] = "online"

import torch

from configs.setup import load_hydra_config
from engine.engine import Engine

import methods.ours.gs_only_callbacks  # noqa: F401
import methods.ours.state  # noqa: F401

from methods.ours.render import render


def _one_step(state, iteration: int, *, measure: bool = False) -> dict:
    timings = {}
    state.iteration = iteration
    state.gauss_solver.step_lr(iteration)
    t0 = time.perf_counter()
    datum = state.scene.next_train_datum()
    timings["fetch_ms"] = (time.perf_counter() - t0) * 1000.0
    if measure:
        torch.cuda.synchronize()
    t0 = time.perf_counter()
    pred_image, depth_image, _ = render(
        state.gauss_container.model_cfg,
        state.scene.dataset_meta,
        datum["slice_meta"],
        state.gauss_container,
        is_jitter=True,
        return_intermediates=False,
    )
    if measure:
        torch.cuda.synchronize()
    timings["render_ms"] = (time.perf_counter() - t0) * 1000.0
    gt_image = datum["slice_meta"].image
    state.inputs["gt_image"] = gt_image
    state.outputs["pred_image"] = pred_image
    state.outputs["depth_image"] = depth_image
    state.outputs["gauss_scales"] = state.gauss_container.scales
    state.outputs["gauss_opacities"] = state.gauss_container.opacities
    if measure:
        torch.cuda.synchronize()
    t0 = time.perf_counter()
    loss_dict = state.loss_computer.compute(state.inputs, state.outputs)
    accum = int(state.cfg.engine.accum_steps)
    total_loss = loss_dict["total"] / accum
    if measure:
        torch.cuda.synchronize()
    timings["loss_ms"] = (time.perf_counter() - t0) * 1000.0
    t0 = time.perf_counter()
    total_loss.backward()
    if measure:
        torch.cuda.synchronize()
    timings["backward_ms"] = (time.perf_counter() - t0) * 1000.0
    t0 = time.perf_counter()
    if iteration % accum == 0:
        state.gauss_solver.step()
        state.gauss_solver.zero_grad()
    if measure:
        torch.cuda.synchronize()
    timings["opt_ms"] = (time.perf_counter() - t0) * 1000.0
    timings["total_ms"] = sum(timings.values())
    return timings


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--experiment", default="preproduction/ideas/ours/renderer_baseline/probe3d")
    parser.add_argument("--warmup", type=int, default=5)
    parser.add_argument("--reps", type=int, default=20)
    args, extra = parser.parse_known_args()
    cfg = load_hydra_config(args.experiment, extra if extra else None)
    engine = Engine(cfg)
    engine.trigger("on_setup")
    state = engine.state
    if not torch.cuda.is_available():
        print("[profile_ours_one_train_step] CUDA unavailable; skip GPU timing.")
        return

    warmup = args.warmup
    reps = args.reps
    for it in range(1, warmup + 1):
        _one_step(state, it)
    torch.cuda.synchronize()
    sums = {}
    t0 = time.perf_counter()
    for k in range(reps):
        timings = _one_step(state, warmup + 1 + k, measure=True)
        for name, value in timings.items():
            sums[name] = sums.get(name, 0.0) + value
    torch.cuda.synchronize()
    dt = time.perf_counter() - t0
    per = dt / reps
    print(
        f"[profile_ours_one_train_step] {reps} steps after {warmup} warmup: "
        f"total={dt:.4f}s mean_per_step={per*1000:.3f}ms"
    )
    for name in ("fetch_ms", "render_ms", "loss_ms", "backward_ms", "opt_ms", "total_ms"):
        if name in sums:
            print(f"[profile_ours_one_train_step] mean_{name}={sums[name] / reps:.3f}")
    engine.trigger("on_finish")


if __name__ == "__main__":
    main()
