"""Native Nerfstudio pipeline with memory-bounded pixel-based initialization."""

import csv
import json
from contextlib import contextmanager
from dataclasses import dataclass, field
from pathlib import Path
from typing import Type

import torch

from nerfstudio.data.datamanagers.ultrasound_datamanager import UltrasoundDataManagerConfig
from nerfstudio.models.base_model import ModelConfig
from nerfstudio.models.ultrasound import UltrasoundModelConfig
from nerfstudio.pipelines.base_pipeline import VanillaPipeline, VanillaPipelineConfig


@dataclass
class UltrasoundPipelineConfig(VanillaPipelineConfig):
    _target: Type = field(default_factory=lambda: UltrasoundPipeline)
    datamanager: UltrasoundDataManagerConfig = field(default_factory=UltrasoundDataManagerConfig)
    model: ModelConfig = field(default_factory=UltrasoundModelConfig)
    uniform_ratio: float = 0.3
    intensity_threshold: float = 0.05
    jitter_radius_mm: float = 0.2
    initial_scales_mm: tuple[float, float, float] = (0.3, 0.3, 0.3)
    initial_quats_wxyz: tuple[float, float, float, float] = (1.0, 0.0, 0.0, 0.0)
    evaluation_output: Path | None = None


class UltrasoundPipeline(VanillaPipeline):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        if hasattr(self.model, "camera_optimizer") and self.model.config.pose.enabled:
            poses = self.model.camera_optimizer.prior_poses[:, :3].clone()
            poses[:, :, 1:3] *= -1
            self.datamanager.train_dataset.cameras.camera_to_worlds = poses
        self.initialize_gaussians()
        from nerfstudio.utils.ultrasound_provenance import execution_identity

        self.execution_chain = [execution_identity()]

    def get_extra_state(self):
        return {"execution_chain": self.execution_chain}

    def set_extra_state(self, state):
        current = self.execution_chain[-1]
        self.execution_chain = list(state["execution_chain"])
        if current != self.execution_chain[-1]:
            self.execution_chain.append(current)

    @contextmanager
    def evaluation_context(self):
        # UltraGray's renderer draws perturbations even when their magnitude is
        # zero. Evaluation must not advance the checkpointed training RNG stream.
        devices = (
            [self.device.index if self.device.index is not None else torch.cuda.current_device()]
            if self.device.type == "cuda"
            else []
        )
        was_training = self.training
        with torch.random.fork_rng(devices=devices):
            self.eval()
            try:
                yield
            finally:
                self.train(was_training)

    @torch.no_grad()
    def get_eval_image_metrics_and_images(self, step):
        with self.evaluation_context():
            return super().get_eval_image_metrics_and_images(step)

    @torch.no_grad()
    def get_eval_loss_dict(self, step):
        with self.evaluation_context():
            return super().get_eval_loss_dict(step)

    @torch.no_grad()
    def get_average_eval_image_metrics(self, step=None, output_path=None, get_std=False):
        rows = []
        with self.evaluation_context():
            for camera, batch in self.datamanager.fixed_indices_eval_dataloader:
                outputs = self.model.get_outputs_for_camera(camera)
                metrics, images = self.model.get_image_metrics_and_images(outputs, batch)
                frame_id = int(camera.metadata["frame_id"].item())
                rows.append({"frame_id": frame_id, **metrics})
                if output_path is not None:
                    from torchvision.utils import save_image

                    output_path.mkdir(parents=True, exist_ok=True)
                    save_image(images["img"].permute(2, 0, 1), output_path / f"eval_{frame_id:06d}.png")
        result = {
            key: float(torch.tensor([row[key] for row in rows], dtype=torch.float64).mean())
            for key in rows[0]
            if key != "frame_id"
        }
        if get_std:
            result.update(
                {
                    key + "_std": float(
                        torch.tensor([row[key] for row in rows], dtype=torch.float64).std(unbiased=False)
                    )
                    for key in list(result)
                }
            )
        if self.config.evaluation_output is not None:
            from nerfstudio.utils.ultrasound_provenance import execution_identity

            dest = self.config.evaluation_output / f"step-{step if step is not None else 'loaded'}"
            dest.mkdir(parents=True, exist_ok=True)
            with (dest / "per-frame.csv").open("w", newline="") as stream:
                writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
                writer.writeheader()
                writer.writerows(rows)
            metadata = self.datamanager.eval_dataset.metadata
            receipt = {
                "step": step,
                "dataset_version": metadata["dataset_version"],
                "manifest_sha256": metadata["manifest_sha256"],
                "parser_identity_sha256": metadata["parser_identity_sha256"],
                "frame_ids": [row["frame_id"] for row in rows],
                "metrics": result,
                "metric_protocol": "pyiqa-0.1.7" if self.model.config.full_metrics else "engineering-only",
                "execution": execution_identity(),
                "training_execution_chain": self.execution_chain,
                "model_input_assets": getattr(self.model, "input_assets", {}),
            }
            (dest / "receipt.json").write_text(json.dumps(receipt, indent=2) + "\n")
        return result

    @torch.no_grad()
    def initialize_gaussians(self):
        if not hasattr(self.model, "initialize_from_pixels"):
            return
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
        self.model.initialize_from_pixels(
            means,
            weights[indices.cpu()].to(self.device),
            torch.tensor(cfg.initial_scales_mm, device=self.device).expand(n, 3),
            torch.tensor(cfg.initial_quats_wxyz, device=self.device).expand(n, 4),
        )
