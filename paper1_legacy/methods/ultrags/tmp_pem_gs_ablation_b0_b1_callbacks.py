"""
PEM ↔ UltraGS 消融编排（B0 / B1），与 ``offline_callbacks.py`` 解耦。

- **B0**（``model.pem_gs_ablation.mode=b0``）：子进程 FiMA PEM → 导出链式 pose → 写 ``scene.loading.poses_abspath``，随后正常 UltraGS 训练。
  **YAML 回调顺序**：``PemGsAblationB0PrepCallback`` 必须列在 ``UltraGSOfflineSetupCallback`` 之前（Engine 不按 ``priority`` 排序列表）。
- **B1**（``mode=b1``）：yaml **不要**注册 ``UltraGSOfflineTrainCallback``；由 ``PemGsAblationB1AlternatingTrainCallback``
  在 ``on_train`` 内交替：PEM ↔ GS（子进程），GS 后轮导出每帧 occupancy 供下一轮 PEM ``model.pem.gs_occ_cache_dir`` 使用。

使用前在 ``run.py`` 导入本模块已注册 callbacks。
"""

from __future__ import annotations

import glob
import os
import subprocess
import sys
from typing import Optional

import numpy as np
import torch
from omegaconf import OmegaConf

from engine.registry import register_callback
from methods.ultrags.render_integrated import render as render_integrated_render
from methods.ultrags.state import UltraGSOfflineState


# ---------------------------------------------------------------------------
# Repo / subprocess helpers
# ---------------------------------------------------------------------------


def _repo_root() -> str:
    return os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))


def _pem_occ_fusion_mode(repo: str, fima_yaml: str) -> str:
    """Read the FiMA Hydra alias and return ``model.pem.occ_fusion_mode``."""
    from configs.setup import load_hydra_config

    cfg = load_hydra_config(fima_yaml)
    return str(getattr(cfg.model.pem, "occ_fusion_mode", "concat")).lower().strip()


def _pem_save_every(mc) -> int:
    ab = mc.pem_gs_ablation
    v = int(ab.pem_save_every)
    return int(ab.pem_iterations) if v <= 0 else v


def _ablation_mode(mc) -> str:
    ab = getattr(mc, "pem_gs_ablation", None)
    if ab is None:
        return "off"
    return str(getattr(ab, "mode", "off")).lower()


def _find_pem_effective_root(pem_user_base: str) -> str:
    """``pem_user_base/* /checkpoints`` 的上一级为 FiMA ``system.output_dir`` 生效目录。"""
    patt = os.path.join(pem_user_base, "**", "checkpoints")
    hits = sorted(glob.glob(patt, recursive=True))
    if not hits:
        raise FileNotFoundError(f"[PemGsAblation] No PEM checkpoints under pattern: {patt}")
    return os.path.dirname(hits[0])


def _find_gs_effective_root(gs_user_base: str) -> str:
    patt = os.path.join(gs_user_base, "**", "checkpoints")
    hits = sorted(glob.glob(patt, recursive=True))
    if not hits:
        raise FileNotFoundError(f"[PemGsAblation] No GS checkpoints under pattern: {patt}")
    return os.path.dirname(hits[0])


def _run_subprocess(cmd: list[str], *, cwd: str, env: Optional[dict] = None) -> None:
    print("[PemGsAblation] exec:", " ".join(cmd))
    merged = {**os.environ, **(env or {})}
    subprocess.check_call(cmd, cwd=cwd, env=merged)


def _run_fima_pem_subprocess(
    *,
    repo: str,
    fima_yaml: str,
    user_output_base: str,
    iterations: int,
    save_every: int,
    seed: int,
    occ_cache_dir: str = "",
    exp_tag_suffix: str = "",
    occ_always_dual: bool = False,
) -> str:
    """
    FiMA PEM 训练；返回 **有效** PEM output（含 dataset_fima 子目录）。
    """
    tags = ("fima_abl_" + exp_tag_suffix) if exp_tag_suffix else "fima_abl"
    cmd = [
        sys.executable,
        "run.py",
        f"experiment={fima_yaml}",
        "task=train",
        f"system.output_root={user_output_base}",
        f"system.exp_tag={tags}",
        f"engine.iterations={iterations}",
        f"engine.interval.save={save_every}",
        f"system.seed={int(seed)}",
    ]
    if occ_cache_dir:
        cmd.append(f"model.pem.gs_occ_cache_dir={occ_cache_dir}")
    if occ_always_dual:
        cmd.append("model.pem.occ_always_two_channels=true")
    _run_subprocess(cmd, cwd=repo)
    return _find_pem_effective_root(user_output_base)


def _run_export_poses(
    repo: str,
    fima_yaml: str,
    pem_effective_root: str,
    resume_iter: int,
    out_npy: str,
    seed: int,
    *,
    pem_export_dotlist: Optional[list[str]] = None,
) -> None:
    cmd = [
        sys.executable,
        os.path.join(repo, "scripts", "methods", "fima", "export_fima_adjacent_chain_poses.py"),
        "--experiment",
        fima_yaml,
        "--pem-output-dir",
        pem_effective_root,
        "--resume-iter",
        str(int(resume_iter)),
        "--out",
        out_npy,
        "--seed",
        str(int(seed)),
    ]
    for item in pem_export_dotlist or []:
        cmd.extend(["--dotlist", item])
    _run_subprocess(cmd, cwd=repo)


def _run_ultrags_subprocess(
    *,
    repo: str,
    gs_yaml: str,
    user_output_base: str,
    poses_npy: str,
    iterations: int,
    save_every: int,
    seed: int,
    resume: str = "",
    exp_tag_suffix: str = "",
) -> str:
    tags = ("gs_abl_" + exp_tag_suffix) if exp_tag_suffix else "gs_abl"
    cmd = [
        sys.executable,
        "run.py",
        f"experiment={gs_yaml}",
        "task=train",
        f"system.output_root={user_output_base}",
        f"system.exp_tag={tags}",
        f"scene.loading.poses_abspath={poses_npy}",
        f"engine.iterations={iterations}",
        f"engine.interval.save={save_every}",
        f"system.seed={int(seed)}",
    ]
    if str(resume or "").strip():
        cmd.insert(-1, f"engine.resume={resume}")
    _run_subprocess(cmd, cwd=repo)
    return _find_gs_effective_root(user_output_base)


# ---------------------------------------------------------------------------
# Occupancy dump（本轮 GS ckpt）
# ---------------------------------------------------------------------------


def dump_gs_train_occupancy_maps(
    repo: str,
    gs_yaml: str,
    gs_user_output_base: str,
    *,
    poses_npy: str,
    gs_resume_iter: int,
    occ_out_dir: str,
    seed: int = 42,
) -> None:
    """
    在 **当前 PEM 预测的 poses** 下，用本轮 GS checkpoint 对每个全局帧索引 ``i`` 前向，
    将 ``occupancy_image`` 存 ``{occ_out_dir}/{i:06d}.npy``。
    """
    os.makedirs(occ_out_dir, exist_ok=True)
    repo = os.path.abspath(repo)

    from configs.setup import load_hydra_config
    from engine.engine import Engine

    overrides = [
        f"system.output_dir={gs_user_output_base}",
        f"engine.resume={int(gs_resume_iter)}",
        f"scene.loading.poses_abspath={poses_npy}",
        f"model.render.intermediates_level=full",
        f"system.seed={int(seed)}",
    ]
    cfg = load_hydra_config(gs_yaml, overrides)

    import methods.ultrags.state  # noqa: F401
    import methods.ultrags.offline_callbacks  # noqa: F401

    eng = Engine(cfg)
    eng.trigger("on_setup")

    st: UltraGSOfflineState = eng.state
    mc = st.gauss_container.model_cfg
    n = len(st.scene.train_container)
    H, W = int(st.scene.dataset_meta.H), int(st.scene.dataset_meta.W)
    print(f"[PemGsAblation][occ] dumping {n} frames -> {occ_out_dir}")

    with torch.no_grad():
        for i in range(n):
            sm = st.scene.train_container.get(i)
            _, _, intermediates = render_integrated_render(
                mc,
                st.scene.dataset_meta,
                sm,
                st.gauss_container,
                return_intermediates=True,
            )
            if intermediates is None or "occupancy_image" not in intermediates:
                raise RuntimeError("[PemGsAblation][occ] missing occupancy_image in intermediates")
            oim = intermediates["occupancy_image"]
            occ = (
                oim.detach().float().cpu().numpy()
                if torch.is_tensor(oim)
                else np.asarray(oim, dtype=np.float32)
            ).reshape(H, W)
            np.save(os.path.join(occ_out_dir, f"{i:06d}.npy"), occ.astype(np.float32))

    eng.trigger("on_finish")
    print(f"[PemGsAblation][occ] done.")


# ---------------------------------------------------------------------------
# Callbacks — B0
# ---------------------------------------------------------------------------


@register_callback("PemGsAblationB0PrepCallback")
class PemGsAblationB0PrepCallback:
    priority = 8

    def on_setup(self, state: UltraGSOfflineState):
        mc = state.cfg.model
        mode = _ablation_mode(mc)
        if mode != "b0":
            return

        ab = mc.pem_gs_ablation
        repo = _repo_root()
        seeds = int(state.cfg.system.seed) if state.cfg.system.seed is not None else 0
        pem_iter = int(ab.pem_iterations)
        save_iv = _pem_save_every(mc)

        work = state.cfg.system.output_dir
        pem_user = os.path.join(work, "_ablation_pem_train")
        os.makedirs(pem_user, exist_ok=True)

        fima_rel_cfg = str(ab.fima_config)
        pem_eff = _run_fima_pem_subprocess(
            repo=repo,
            fima_yaml=fima_rel_cfg,
            user_output_base=pem_user,
            iterations=pem_iter,
            save_every=save_iv,
            seed=seeds,
            occ_cache_dir="",
            exp_tag_suffix="b0",
        )

        pose_npy = os.path.join(work, "pred_chain_poses_b0.npy")
        _run_export_poses(repo, fima_rel_cfg, pem_eff, pem_iter, pose_npy, seeds, pem_export_dotlist=None)

        OmegaConf.update(state.cfg.scene.loading, "poses_abspath", pose_npy, merge=True)
        print(f"[PemGsAblation][B0] poses_abspath -> {pose_npy}")


# ---------------------------------------------------------------------------
# Callbacks — B1（整段 alternating 仅在 on_train）
# ---------------------------------------------------------------------------


@register_callback("PemGsAblationB1AlternatingTrainCallback")
class PemGsAblationB1AlternatingTrainCallback:
    """取代常规 UltraGS 训练循环：多轮子进程 PEM↔GS + occupancy 缓存。"""

    priority = 50

    def on_train(self, state: UltraGSOfflineState):
        mc = state.cfg.model
        if _ablation_mode(mc) != "b1":
            return

        ab = mc.pem_gs_ablation
        repo = _repo_root()
        seeds = int(state.cfg.system.seed) if state.cfg.system.seed is not None else 0
        fima_yaml_arg = str(ab.fima_config)
        udy = str(getattr(ab, "ultrags_driver_yaml", "") or "").strip()
        if not udy:
            raise RuntimeError(
                "[PemGsAblation][B1] 请设置 model.pem_gs_ablation.ultrags_driver_yaml 为 UltraGS Hydra experiment alias。"
            )
        ultrags_yaml = udy

        work = state.cfg.system.output_dir
        pem_it = int(ab.pem_iterations)
        gs_it = int(ab.alternating_gs_iterations)
        save_pem = _pem_save_every(mc)
        rounds = max(1, int(ab.num_rounds))
        fusion = _pem_occ_fusion_mode(repo, fima_yaml_arg)
        concat_dual = fusion != "weighted"

        prev_occ_dir = ""
        last_pose_np = ""

        for r in range(rounds):
            pem_user = os.path.join(work, f"_ablation_b1_pem_round_{r}")
            gs_user = os.path.join(work, f"_ablation_b1_gs_round_{r}")

            pem_eff = _run_fima_pem_subprocess(
                repo=repo,
                fima_yaml=fima_yaml_arg,
                user_output_base=pem_user,
                iterations=pem_it,
                save_every=save_pem,
                seed=seeds + r,
                occ_cache_dir=prev_occ_dir,
                exp_tag_suffix=f"b1_r{r}",
                occ_always_dual=concat_dual,
            )

            pose_np = os.path.join(work, f"pred_chain_poses_b1_round_{r}.npy")
            # concat：导出须带双通道标志与训练 stem 一致；weighted：单通道，禁止误开 occ_always_two_channels。
            _run_export_poses(
                repo,
                fima_yaml_arg,
                pem_eff,
                pem_it,
                pose_np,
                seeds + r,
                pem_export_dotlist=["model.pem.occ_always_two_channels=true"] if concat_dual else None,
            )
            last_pose_np = pose_np

            save_gs = gs_it if gs_it > 0 else 2500
            _ = _run_ultrags_subprocess(
                repo=repo,
                gs_yaml=ultrags_yaml,
                user_output_base=gs_user,
                poses_npy=pose_np,
                iterations=gs_it,
                save_every=save_gs,
                seed=seeds + 100 + r,
                resume="",
                exp_tag_suffix=f"b1_r{r}",
            )

            prev_occ_dir = os.path.join(work, f"_ablation_b1_occ_round_{r}")
            dump_gs_train_occupancy_maps(
                repo,
                ultrags_yaml,
                gs_user,
                poses_npy=pose_np,
                gs_resume_iter=gs_it,
                occ_out_dir=prev_occ_dir,
                seed=seeds + 200 + r,
            )

            print(f"[PemGsAblation][B1] round={r}/{rounds - 1}; next PEM occ_dir={prev_occ_dir}")

        print(f"[PemGsAblation][B1] finished. Last pose chain: {last_pose_np}")
