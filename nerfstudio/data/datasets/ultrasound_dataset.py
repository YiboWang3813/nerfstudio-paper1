"""Memory-mapped ultrasound frames; no duplicate PNG export or GPU-wide cache."""

import numpy as np
import torch

from nerfstudio.data.datasets.base_dataset import InputDataset


class UltrasoundDataset(InputDataset):
    def __init__(self, dataparser_outputs, scale_factor=1.0):
        if scale_factor != 1.0:
            raise ValueError("Resampling must be explicit in the versioned dataset package.")
        super().__init__(dataparser_outputs, scale_factor)
        self._images = np.load(self.metadata["images_path"], mmap_mode="r", allow_pickle=False)

    def get_image_float32(self, image_idx):
        image = np.array(self._images[self.metadata["array_indices"][image_idx]], dtype=np.float32)
        mode = self.metadata["image_value_mode"]
        if mode == "uint8_255":
            image = np.clip(np.nan_to_num(image, nan=0.0, posinf=255.0, neginf=0.0), 0.0, 255.0) / 255.0
        elif mode == "spine_phantom_shifted_float":
            image = np.clip(np.nan_to_num(image, nan=-0.125, posinf=1.0, neginf=-0.125) + 0.125, 0.0, 1.0)
        elif mode == "float_01":
            if not np.isfinite(image).all() or image.min() < 0 or image.max() > 1:
                raise ValueError("float_01 images must be finite and within [0,1].")
        else:
            raise ValueError(f"Unsupported image_value_mode: {mode}")
        return torch.from_numpy(image[..., None])

    def get_image_uint8(self, image_idx):
        return (self.get_image_float32(image_idx) * 255).round().to(torch.uint8)

    def get_metadata(self, data):
        return {"frame_id": self.metadata["frame_ids"][data["image_idx"]]}
