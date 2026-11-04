"""Native Nerfstudio pipeline with memory-bounded pixel-based initialization."""

from dataclasses import dataclass, field
from typing import Type

import torch

from nerfstudio.data.datamanagers.ultrasound_datamanager import UltrasoundDataManagerConfig
from nerfstudio.models.ultrasound import UltrasoundModelConfig
from nerfstudio.pipelines.base_pipeline import VanillaPipeline, VanillaPipelineConfig


@dataclass
class UltrasoundPipelineConfig(VanillaPipelineConfig):
    _target: Type = field(default_factory=lambda: UltrasoundPipeline)
    datamanager: UltrasoundDataManagerConfig = field(default_factory=UltrasoundDataManagerConfig)
    model: UltrasoundModelConfig = field(default_factory=UltrasoundModelConfig)
    uniform_ratio: float = 0.3
    intensity_threshold: float = 0.05
    jitter_radius_mm: float = 0.2


class UltrasoundPipeline(VanillaPipeline):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.initialize_gaussians()

    @torch.no_grad()
    def initialize_gaussians(self):
        dataset, cfg = self.datamanager.train_dataset, self.config
        if not 0 <= cfg.uniform_ratio <= 1 or cfg.jitter_radius_mm < 0:
            raise ValueError("Invalid Gaussian initialization sampling parameters.")
        g = dataset.metadata["geometry"]
        h, w = g["height"], g["width"]
        # Keep scalar candidate weights on CPU, never allocate full pixel XYZ,
        # quaternion and scale arrays for every frame on the GPU.
        weights = torch.cat([dataset.get_image_float32(i).flatten() for i in range(len(dataset))])
        n = len(self.model.gaussians["means"])
        count_uniform = int(n * cfg.uniform_ratio)
        uniform = torch.randint(len(weights), (count_uniform,))
        valid = torch.nonzero(weights > cfg.intensity_threshold).flatten()
        if len(valid) == 0:
            importance = torch.randint(len(weights), (n - count_uniform,))
        else:
            if len(valid) > 16_000_000:
                valid = valid[torch.randint(len(valid), (16_000_000,))]
            importance = valid[torch.multinomial(weights[valid], n - count_uniform, replacement=True)]
        indices = torch.cat([uniform, importance]).to(self.device)
        frame = indices // (h * w)
        pixel = indices % (h * w)
        offset = (w - 1) / 2 if g["center_lateral"] else 0.0
        local = torch.stack(
            [
                (pixel % w - offset) * g["lateral_spacing_mm"],
                pixel.div(w, rounding_mode="floor") * g["axial_spacing_mm"] + g["near_mm"],
                torch.zeros(n, device=self.device),
            ],
            dim=-1,
        )
        poses = dataset.cameras.camera_to_worlds[frame].clone()
        poses[:, :, 1:3] *= -1
        means = torch.bmm(poses[:, :, :3], local[:, :, None]).squeeze(-1) + poses[:, :, 3]
        means += (torch.rand_like(means) * 2 - 1) * cfg.jitter_radius_mm
        self.model.gaussians["means"].copy_(means)
        for name, initial, maximum in (
            ("echo_alphas", self.model.config.initial_echo_alpha, self.model.config.echo_alpha_max),
            ("attenuations", self.model.config.initial_attenuation, self.model.config.attenuation_max),
        ):
            values = torch.empty(n, device=self.device).uniform_(initial * 0.8, initial * 1.2)
            self.model.gaussians[name].copy_(torch.logit((values / maximum).clamp(1e-8, 1 - 1e-8)))
