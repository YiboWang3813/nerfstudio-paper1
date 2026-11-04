"""Versioned ultrasound sweeps with explicit splits and physical slice geometry."""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Literal, Type

import numpy as np
import torch

from nerfstudio.cameras.cameras import Cameras, CameraType
from nerfstudio.data.dataparsers.base_dataparser import DataParser, DataParserConfig, DataparserOutputs
from nerfstudio.data.scene_box import SceneBox


@dataclass
class UltrasoundDataParserConfig(DataParserConfig):
    _target: Type = field(default_factory=lambda: UltrasoundDataParser)
    data: Path = Path("/home/yibo/shared/datasets/paper1/manifest.json")
    verify_hashes: bool = True
    pose_mode: Literal["file", "canonical"] = "file"
    canonical_spacing_mm: float = 0.0
    canonical_axis: Literal["x", "y", "z"] = "z"
    canonical_reference_frame_id: int = 0
    pose_validation: Literal["rigid", "preserve_source_affine"] = "rigid"


class UltrasoundDataParser(DataParser):
    """No automatic orientation, scaling, image conversion or split generation."""

    def _generate_dataparser_outputs(self, split="train"):
        path = self.config.data.resolve()
        if self.config.pose_validation not in ("rigid", "preserve_source_affine"):
            raise ValueError("Unknown pose validation mode")
        manifest = json.loads(path.read_text())
        if manifest.get("schema_version") != 1 or manifest.get("pose_convention") != "probe_to_world_mm":
            raise ValueError("Expected schema_version=1 and pose_convention=probe_to_world_mm.")
        if not manifest.get("dataset_version"):
            raise ValueError("A frozen dataset_version is required.")
        geometry = manifest["geometry"]
        h, w = int(geometry["height"]), int(geometry["width"])
        dx, dy = float(geometry["lateral_spacing_mm"]), float(geometry["axial_spacing_mm"])
        near = float(geometry["near_mm"])
        if min(h, w, dx, dy, float(geometry["elevational_spacing_mm"])) <= 0:
            raise ValueError("Slice dimensions and spacings must be positive.")
        if geometry.get("probe_type") != "linear":
            raise ValueError("Only calibrated linear image planes are currently supported.")
        # Every split declares its source arrays and original frame IDs. This also
        # supports historical interpolation sets stored in different NPY files.
        sets = manifest["splits"]
        selected = sets["test" if split == "val" else split]
        root = path.parent
        arrays = {}
        for name in ("images", "poses"):
            spec = selected[name]
            file = (root / spec["path"]).resolve()
            if not file.is_relative_to(root):
                raise ValueError("Payload references must remain inside the dataset package.")
            if self.config.verify_hashes:
                with file.open("rb") as stream:
                    actual = hashlib.file_digest(stream, "sha256").hexdigest()
                if actual != spec["sha256"]:
                    raise ValueError(f"Dataset payload checksum mismatch: {file}")
            arrays[name] = np.load(file, mmap_mode="r", allow_pickle=False)
        images, poses = arrays["images"], arrays["poses"]
        if images.ndim != 3 or images.shape[1:] != (h, w) or poses.shape != (len(images), 4, 4):
            raise ValueError(f"Invalid image/pose shapes: {images.shape}, {poses.shape}.")
        indices = np.asarray(selected["indices"], dtype=np.int64)
        frame_ids = np.asarray(selected["frame_ids"], dtype=np.int64)
        if (
            len(indices) == 0
            or len(indices) != len(frame_ids)
            or len(np.unique(indices)) != len(indices)
            or np.any(indices < 0)
            or np.any(indices >= len(images))
        ):
            raise ValueError("Split indices must be unique, nonempty, in range and aligned with frame_ids.")
        # Detect leakage when train and test refer to the same source images.
        train, test = sets["train"], sets["test"]
        if train["images"]["sha256"] == test["images"]["sha256"]:
            if set(train["indices"]) & set(test["indices"]):
                raise ValueError("Train/test split leakage.")
        probe_poses = torch.from_numpy(np.array(poses[indices], dtype=np.float32))
        # Pose residual smoothness and uncertainty priors operate on the entire
        # temporal sweep, including held-out frames, as in the frozen protocol.
        # Image losses still see only the selected train split.
        domain_ids = {}
        same_source = all(train[name]["sha256"] == test[name]["sha256"] for name in ("images", "poses"))
        if same_source:
            for subset in (train, test):
                for index, frame_id in zip(subset["indices"], subset["frame_ids"]):
                    if index in domain_ids and domain_ids[index] != frame_id:
                        raise ValueError("Inconsistent source frame identity")
                    domain_ids[index] = frame_id
        complete_domain = same_source and set(domain_ids) == set(range(len(poses)))
        pose_frame_ids = [domain_ids[index] for index in range(len(poses))] if complete_domain else frame_ids.tolist()
        domain_poses = torch.from_numpy(np.array(poses, dtype=np.float32)) if complete_domain else probe_poses.clone()
        if self.config.pose_mode == "canonical":
            if self.config.canonical_spacing_mm <= 0:
                raise ValueError("Canonical sweeps require explicit positive calibrated spacing in mm.")
            probe_poses = torch.eye(4).repeat(len(indices), 1, 1)
            axis = {"x": 0, "y": 1, "z": 2}[self.config.canonical_axis]
            probe_poses[:, axis, 3] = (
                torch.from_numpy(frame_ids - self.config.canonical_reference_frame_id)
                * self.config.canonical_spacing_mm
            )
            domain_poses = torch.eye(4).repeat(len(pose_frame_ids), 1, 1)
            domain_poses[:, axis, 3] = (
                torch.tensor(pose_frame_ids) - self.config.canonical_reference_frame_id
            ) * self.config.canonical_spacing_mm
        rotation = probe_poses[:, :3, :3]
        if (
            not torch.isfinite(probe_poses).all()
            or not torch.allclose(probe_poses[:, 3], torch.tensor([0.0, 0.0, 0.0, 1.0]).expand(len(indices), 4))
            or (
                self.config.pose_validation == "rigid"
                and not torch.allclose(
                    rotation.transpose(-1, -2) @ rotation, torch.eye(3).expand(len(indices), 3, 3), atol=1e-4
                )
            )
            or (
                self.config.pose_validation == "rigid"
                and not torch.allclose(torch.linalg.det(rotation), torch.ones(len(indices)), atol=1e-4)
            )
            or torch.any(torch.linalg.det(rotation) <= 1e-6)
        ):
            raise ValueError("Poses must be finite rigid homogeneous probe-to-world transforms.")
        # Camera axes right/up/back correspond to probe lateral/-axial/-elevational.
        c2w = probe_poses[:, :3].clone()
        c2w[:, :, 1:3] *= -1
        offset = (w - 1) / 2 if geometry["center_lateral"] else 0.0
        cameras = Cameras(
            camera_to_worlds=c2w,
            fx=1 / dx,
            fy=1 / dy,
            cx=offset,
            cy=-near / dy,
            height=h,
            width=w,
            camera_type=CameraType.ORTHOPHOTO,
            metadata={"frame_id": torch.from_numpy(frame_ids).reshape(-1, 1)},
        )
        # Bounds include physical image corners rather than only probe origins.
        corners = torch.tensor(
            [
                [-offset * dx, near, 0.0],
                [(w - 1 - offset) * dx, near, 0.0],
                [-offset * dx, near + (h - 1) * dy, 0.0],
                [(w - 1 - offset) * dx, near + (h - 1) * dy, 0.0],
            ]
        )
        world = torch.einsum("nij,kj->nki", rotation, corners) + probe_poses[:, None, :3, 3]
        low, high = world.amin((0, 1)), world.amax((0, 1))
        padding = max(float(geometry["elevational_spacing_mm"]), 1e-3)
        metadata = {
            "probe_poses": probe_poses,
            "pose_domain_frame_ids": pose_frame_ids,
            "pose_domain_probe_poses": domain_poses,
            "pose_domain_complete": complete_domain,
            "pose_validation": self.config.pose_validation,
            "source_rotation_orthogonality_max_error": float(
                (rotation.transpose(-1, -2) @ rotation - torch.eye(3)).abs().max()
            ),
            "scene_radius_mm": float(
                (
                    (torch.ceil((high + (high - low) * 0.05) / 4) * 4)
                    - (torch.floor((low - (high - low) * 0.05) / 4) * 4)
                ).norm()
                / 2
            ),
            "geometry": geometry,
            "images_path": str((root / selected["images"]["path"]).resolve()),
            "array_indices": indices.tolist(),
            "frame_ids": frame_ids.tolist(),
            "image_value_mode": manifest["image_value_mode"],
            "dataset_version": manifest["dataset_version"],
            "manifest_sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
        }
        parser_identity = {
            "manifest_sha256": metadata["manifest_sha256"],
            "pose_mode": self.config.pose_mode,
            "canonical_spacing_mm": self.config.canonical_spacing_mm,
            "canonical_axis": self.config.canonical_axis,
            "canonical_reference_frame_id": self.config.canonical_reference_frame_id,
            "pose_validation": self.config.pose_validation,
        }
        metadata["parser_identity_sha256"] = hashlib.sha256(
            json.dumps(parser_identity, sort_keys=True).encode()
        ).hexdigest()
        return DataparserOutputs(
            image_filenames=[root / selected["images"]["path"]] * len(indices),
            cameras=cameras,
            scene_box=SceneBox(torch.stack([low - padding, high + padding])),
            metadata=metadata,
            dataparser_scale=1.0,
        )
