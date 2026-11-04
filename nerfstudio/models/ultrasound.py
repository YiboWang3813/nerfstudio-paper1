"""Native full-slice Gaussian model; all geometry is in world millimetres."""

from __future__ import annotations

from dataclasses import dataclass, field, replace
from types import SimpleNamespace
from typing import Type

import torch
import torch.nn.functional as F
from gsplat.strategy.ultrasound import UltrasoundStrategy
from gsplat.ultrasound import UltrasoundRenderConfig, build_slice_grid_pack, rasterization
from gsplat.ultrasound.diagnostics import contribution_residual_stats
from torch import nn

from nerfstudio.cameras.cameras import Cameras
from nerfstudio.cameras.ultrasound_poses import (
    UltrasoundCameraOptimizer,
    UltrasoundPoseConfig,
    UltrasoundUncertaintyConfig,
    _euler_xyz_to_matrix,
    frame_weights,
)
from nerfstudio.engine.callbacks import TrainingCallback, TrainingCallbackLocation
from nerfstudio.model_components.ultrasound_losses import UltrasoundSSIM
from nerfstudio.models.acoustic_slice import AcousticSliceModel, AcousticSliceModelConfig


@dataclass
class UltrasoundModelConfig(AcousticSliceModelConfig):
    _target: Type = field(default_factory=lambda: UltrasoundModel)
    enable_collider: bool = False
    num_gaussians: int = 1_000_000
    initial_scale_mm: float = 0.3
    initial_echo_alpha: float = 0.1
    initial_attenuation: float = 0.02
    scale_min_mm: float = 0.05
    scale_max_mm: float = 3.0
    echo_alpha_max: float = 0.2
    attenuation_max: float = 1.0
    l1_weight: float = 0.8
    ssim_weight: float = 0.2
    pose: UltrasoundPoseConfig = field(default_factory=UltrasoundPoseConfig)
    uncertainty: UltrasoundUncertaintyConfig = field(default_factory=UltrasoundUncertaintyConfig)
    density: UltrasoundStrategy = field(default_factory=UltrasoundStrategy)
    renderer: UltrasoundRenderConfig = field(
        default_factory=lambda: UltrasoundRenderConfig(
            H=1,
            W=1,
            lateral_spacing_mm_per_pix=1.0,
            axial_spacing_mm_per_pix=1.0,
            axial_near_mm=0.0,
            tile_size_px=8,
            k_sigma=2.5,
            shader_weight_mode="us_focus",
            local_conditioning_mode="fixed",
            us_focus_sigma_ele_min=0.8,
            us_focus_sigma_ele_max=5.0,
            mapper_sort_by_v=False,
        )
    )


class UltrasoundModel(AcousticSliceModel):
    config: UltrasoundModelConfig

    @torch.no_grad()
    def initialize_from_pixels(self, means, values, scales, quats):
        self.gaussians["means"].copy_(means)
        self.gaussians["scales"].copy_(
            torch.logit(
                ((scales - self.config.scale_min_mm) / (self.config.scale_max_mm - self.config.scale_min_mm)).clamp(
                    1e-8, 1 - 1e-8
                )
            )
        )
        self.gaussians["quats"].copy_(F.normalize(quats, dim=-1, eps=1e-8))
        for name, initial, maximum in (
            ("echo_alphas", self.config.initial_echo_alpha, self.config.echo_alpha_max),
            ("attenuations", self.config.initial_attenuation, self.config.attenuation_max),
        ):
            values = torch.empty(len(means), device=means.device).uniform_(initial * 0.8, initial * 1.2)
            self.gaussians[name].copy_(torch.logit((values / maximum).clamp(1e-8, 1 - 1e-8)))

    def populate_modules(self):
        super().populate_modules()
        cfg = self.config
        if not (0 < cfg.scale_min_mm < cfg.initial_scale_mm < cfg.scale_max_mm):
            raise ValueError("Initial scale must be strictly inside the physical scale bounds.")
        if not (0 < cfg.initial_echo_alpha < cfg.echo_alpha_max and 0 < cfg.initial_attenuation < cfg.attenuation_max):
            raise ValueError("Initial echo and attenuation must be strictly inside their bounds.")
        n = cfg.num_gaussians
        if n <= 0:
            raise ValueError("num_gaussians must be positive.")

        def raw(value):
            return torch.logit(torch.tensor(value))

        self.gaussians = nn.ParameterDict(
            {
                "means": nn.Parameter(torch.zeros(n, 3)),
                "scales": nn.Parameter(
                    raw((cfg.initial_scale_mm - cfg.scale_min_mm) / (cfg.scale_max_mm - cfg.scale_min_mm))
                    .expand(n, 3)
                    .clone()
                ),
                "quats": nn.Parameter(torch.tensor([1.0, 0.0, 0.0, 0.0]).expand(n, 4).clone()),
                "echo_alphas": nn.Parameter(raw(cfg.initial_echo_alpha / cfg.echo_alpha_max).expand(n).clone()),
                "attenuations": nn.Parameter(raw(cfg.initial_attenuation / cfg.attenuation_max).expand(n).clone()),
            }
        )
        self.geometry = self.kwargs["metadata"]["geometry"]
        metadata = self.kwargs["metadata"]
        import hashlib
        from pathlib import Path

        self.input_assets = {}
        for name, path in [
            ("pose_prior", cfg.pose.prior_poses_abspath if cfg.pose.enabled else ""),
            ("uncertainty", cfg.uncertainty.uncertainty_abspath if cfg.uncertainty.enabled else ""),
        ]:
            if path:
                asset = Path(path).expanduser().resolve()
                with asset.open("rb") as stream:
                    digest = hashlib.file_digest(stream, "sha256").hexdigest()
                self.input_assets[name] = {"path": str(asset), "sha256": digest}
        self.camera_optimizer = UltrasoundCameraOptimizer(cfg.pose, metadata["frame_ids"], metadata["probe_poses"])
        self.register_buffer("uncertainty_weights", frame_weights(cfg.uncertainty, metadata["frame_ids"]))
        g = self.geometry
        self.render_config = replace(
            cfg.renderer,
            H=g["height"],
            W=g["width"],
            lateral_spacing_mm_per_pix=g["lateral_spacing_mm"],
            axial_spacing_mm_per_pix=g["axial_spacing_mm"],
            axial_near_mm=g["near_mm"],
        )
        grid_meta = SimpleNamespace(
            H=g["height"],
            W=g["width"],
            delta_lateral_mm=g["lateral_spacing_mm"],
            delta_axial_mm=g["axial_spacing_mm"],
            near_mm=g["near_mm"],
            center_lateral=g["center_lateral"],
        )
        self.register_buffer("slice_grid", build_slice_grid_pack(grid_meta), persistent=False)
        self.ssim = UltrasoundSSIM()
        self.strategy = cfg.density
        self.strategy_state = self.strategy.initialize_state()
        self._info, self._step, self._gradient_hook = {}, 0, None
        if (
            self.strategy.enabled
            and self.strategy.pruning == "path_aware"
            and not self.render_config.ray_attenuation_enabled
        ):
            raise ValueError("Path-aware pruning requires global attenuation rendering.")

    def get_training_callbacks(self, training_callback_attributes):
        attributes = training_callback_attributes
        if attributes.grad_scaler is not None and attributes.grad_scaler.is_enabled():
            raise ValueError("Ultrasound topology control has not been validated with gradient scaling.")
        if attributes.trainer is not None and any(
            value != 1 for value in attributes.trainer.gradient_accumulation_steps.values()
        ):
            raise ValueError("Ultrasound topology control currently requires one optimizer update per slice.")
        self._optimizers = attributes.optimizers
        self.strategy.check_sanity(self.gaussians, {key: self._optimizers.optimizers[key] for key in self.gaussians})

        def before(step):
            self._step, self._info = step + 1, {}
            self.camera_optimizer.step = self._step
            p = self.config.pose
            self._pose_only = (
                self.camera_optimizer.is_optimizing()
                and p.pose_only_start_iter >= 0
                and p.pose_only_start_iter <= self._step <= p.pose_only_end_iter
            )
            for parameter in self.gaussians.values():
                parameter.requires_grad_(not self._pose_only)
            if self._gradient_hook is not None:
                self._gradient_hook.remove()
            if self.strategy.in_window(self._step) and not self._pose_only:
                self._gradient_hook = self.gaussians["means"].register_hook(
                    lambda grad: self._info.update(means_grad=grad.detach().clone() / self._info.get("dc_weight", 1.0))
                )

        def after(step):
            self.camera_optimizer.clamp_residuals()
            if self.strategy.in_window(step + 1) and not self._pose_only:
                self._info.update(
                    activated_scales=self.scales.detach(),
                    activated_echo=self.echo_alphas.detach(),
                    echo_max=self.config.echo_alpha_max,
                    scale_min=self.config.scale_min_mm,
                    scale_max=self.config.scale_max_mm,
                    activated_attenuation=self.attenuations.detach(),
                )
                self.strategy.step_post_backward(
                    self.gaussians, self._optimizers.optimizers, self.strategy_state, step + 1, self._info
                )
                # Keep Nerfstudio's clipping parameter registry synchronized too.
                for name, parameter in self.gaussians.items():
                    self._optimizers.parameters[name] = [parameter]
            self._info = {}

        return [
            TrainingCallback([TrainingCallbackLocation.BEFORE_TRAIN_ITERATION], before),
            TrainingCallback([TrainingCallbackLocation.AFTER_TRAIN_ITERATION], after),
        ]

    @property
    def scales(self):
        return (
            self.config.scale_min_mm
            + (self.config.scale_max_mm - self.config.scale_min_mm) * self.gaussians["scales"].sigmoid()
        )

    @property
    def echo_alphas(self):
        return self.config.echo_alpha_max * self.gaussians["echo_alphas"].sigmoid()

    @property
    def attenuations(self):
        return self.config.attenuation_max * self.gaussians["attenuations"].sigmoid()

    def get_param_groups(self):
        groups = {name: [parameter] for name, parameter in self.gaussians.items()}
        self.camera_optimizer.get_param_groups(groups)
        return groups

    def get_outputs(self, camera):
        return self._render(camera, self.render_config, self.slice_grid)

    def get_metrics_dict(self, outputs, batch):
        metrics = super().get_metrics_dict(outputs, batch)
        self.camera_optimizer.get_metrics_dict(metrics)
        return metrics

    def _render(self, camera, render_config, slice_grid):
        if not isinstance(camera, Cameras) or camera.size != 1:
            raise ValueError("UltrasoundModel expects one calibrated full-slice Cameras object.")
        if int(camera.height.item()) != render_config.H or int(camera.width.item()) != render_config.W:
            raise ValueError("Changing output resolution requires explicit physical resampling.")
        pose = torch.eye(4, device=self.device)
        pose[:3, :] = camera.camera_to_worlds.reshape(3, 4).to(self.device)
        pose[:3, 1:3] *= -1
        pose = self.camera_optimizer.probe_pose(camera, pose)
        if self.training:
            self._info["frame_index"] = self.camera_optimizer.indices[int(camera.metadata["frame_id"].item())]
        collect_path = self.training and self.strategy.collect_path(self._step)

        def render(current_pose):
            return rasterization(
                render_config,
                self.gaussians["means"],
                self.scales,
                F.normalize(self.gaussians["quats"], dim=-1),
                self.echo_alphas,
                torch.ones_like(self.echo_alphas),
                current_pose,
                slice_grid,
                self.attenuations if render_config.ray_attenuation_source == "attenuation" else None,
                collect_path,
            )

        u = self.config.uncertainty
        poses = [pose]
        if self.training and u.mc_enabled:
            import math

            for _ in range(max(1, u.mc_samples) - 1):
                translation = torch.randn(3, device=self.device) * u.mc_trans_sigma_mm
                rotation = torch.randn(3, device=self.device) * math.radians(u.mc_rot_sigma_deg)
                poses.append(pose @ _euler_xyz_to_matrix(torch.cat([rotation, translation])))
        samples = [render(current) for current in poses]
        image = (
            tuple(torch.stack(items).mean(0) for items in zip(*samples))
            if collect_path
            else torch.stack(samples).mean(0)
        )
        if collect_path:
            image, path, support = image
            self._info.update(path_contribution=path, path_support=support)
        if self.training:
            self._info["pose"] = pose.detach()
        gray = image.reshape(render_config.H, render_config.W, 1)
        return {"ultrasound": gray, "rgb": gray.expand(-1, -1, 3)}

    def get_outputs_for_camera(self, camera, obb_box=None):
        height, width = int(camera.height.item()), int(camera.width.item())
        if height == self.render_config.H and width == self.render_config.W:
            return self.get_outputs(camera)
        # Preview resampling preserves the calibrated physical field of view.
        # Training data resolution is immutable; viewer output resolution is not.
        config = replace(
            self.render_config,
            H=height,
            W=width,
            lateral_spacing_mm_per_pix=self.render_config.lateral_spacing_mm_per_pix * self.render_config.W / width,
            axial_spacing_mm_per_pix=self.render_config.axial_spacing_mm_per_pix * self.render_config.H / height,
        )
        meta = SimpleNamespace(
            H=height,
            W=width,
            delta_lateral_mm=config.lateral_spacing_mm_per_pix,
            delta_axial_mm=config.axial_spacing_mm_per_pix,
            near_mm=config.axial_near_mm,
            center_lateral=self.geometry["center_lateral"],
        )
        return self._render(camera, config, build_slice_grid_pack(meta, self.device))

    def get_loss_dict(self, outputs, batch, metrics_dict=None):
        pred, target = outputs["ultrasound"], batch["image"].to(self.device)
        if self.training and self.strategy.collect_local(self._step):
            with torch.no_grad():
                residual = (pred - target).abs().squeeze(-1)
                self._info["local_stats"] = contribution_residual_stats(
                    self.render_config,
                    self.gaussians["means"],
                    self.scales,
                    F.normalize(self.gaussians["quats"], dim=-1),
                    self.echo_alphas,
                    torch.ones_like(self.echo_alphas),
                    self._info["pose"],
                    self.slice_grid,
                    residual,
                    float(torch.quantile(residual, self.strategy.high_residual_quantile)),
                )
        p, t = pred.permute(2, 0, 1)[None], target.permute(2, 0, 1)[None]
        losses = {
            "l1": self.config.l1_weight * F.l1_loss(pred, target),
            "ssim": self.config.ssim_weight * (1 - self.ssim(p, t)),
        }
        if self.training:
            index = self._info["frame_index"]
            u = self.config.uncertainty
            if u.enabled:
                weight = self.uncertainty_weights[index]
                for name in set(u.image_loss_names) & losses.keys():
                    losses[name] *= weight
                if u.decouple_density_control:
                    self._info["dc_weight"] = max(float(weight), 1e-6)
            self.camera_optimizer.get_loss_dict(losses, index)
        return losses
