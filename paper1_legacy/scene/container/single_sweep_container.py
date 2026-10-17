import os
import json
from dataclasses import replace
from typing import Tuple
import numpy as np
import torch
from tqdm import tqdm

from ..meta.dataset_meta import DatasetMeta
from configs.base.scene import SceneConfig

from ..meta.slice_meta import SliceMeta
from ..meta.slice_meta.linear import LinearSliceMeta
from ..meta.slice_meta.convex import ConvexSliceMeta


class SingleSweepContainer:
    """ 单次扫描的数据容器 只能存储一个完整sweep的数据 """
    
    def __init__(
        self,
        meta: DatasetMeta,
        cfg: SceneConfig,
        device: torch.device,
        images_name: str = None,
        poses_name: str = None,
    ): 
        self.meta = meta
        self.cfg = cfg   
        self.device = device
        self.pose_mode = str(getattr(cfg.loading, "pose_mode", "file") or "file").lower()
        self.file_poses_loaded_for_length = False
        self.uses_file_poses_for_geometry = True

        # 1. 路径组装 (使用 cfg.loading)
        # 逻辑：baseroot/dataset_dirname/sweep_name
        dataset_dirname = cfg.loading.dataset_dirname or cfg.dataset_name
        self.sweep_dir = os.path.join(
            cfg.loading.baseroot, 
            dataset_dirname,
            cfg.loading.sweep_name
        )

        images_name = images_name or cfg.loading.images_name
        images_path = os.path.join(self.sweep_dir, images_name)
        
        # 根据 pose_type 选择加载不同类型的 pose
        # 绝对位姿：poses.npy 为传感器绝对 probe→world
        # 相对位姿：relative_poses.npy，与 data/synthetic_liver/gen_relative.py 一致：
        #   R_i = inv(pose[0]) @ pose[i]，R_0 = I，R_i 仍为 4×4 齐次矩阵供渲染/优化使用
        if poses_name is None:
            if cfg.loading.pose_type == "absolute":
                poses_name = "poses.npy"
            elif cfg.loading.pose_type == "relative":
                poses_name = "relative_poses.npy"
            else:
                raise NotImplementedError(f"Unsupported pose type: {cfg.loading.pose_type}")
        poses_override = str(getattr(cfg.loading, "poses_abspath", "") or "").strip()
        if poses_override:
            poses_path = os.path.abspath(os.path.expanduser(poses_override))
            print(f"[SingleSweepContainer] Loading poses from poses_abspath: {poses_path}")
        else:
            poses_path = os.path.join(self.sweep_dir, poses_name)

        if not os.path.isfile(images_path):
            raise FileNotFoundError(f"Missing file: {images_path}")
        if not os.path.isfile(poses_path):
            raise FileNotFoundError(f"Missing file: {poses_path}")

        print(
            f"[SingleSweepContainer] Loading '{meta.name}' from '{self.sweep_dir}' "
            f"(images='{images_name}', poses='{poses_name}')"
        )

        # 2. Load disk-backed data
        images_np = np.load(images_path)   # (N, H, W)
        poses_np = np.load(poses_path)     # (N, 4, 4)
        self.file_poses_loaded_for_length = True
        
        if images_np.ndim != 3:
            raise ValueError(f"Expected images shape (N,H,W), got {images_np.shape} from {images_path}")
        if poses_np.ndim != 3 or poses_np.shape[1:] != (4, 4):
            raise ValueError(f"Expected poses shape (N,4,4), got {poses_np.shape} from {poses_path}")
        if images_np.shape[0] != poses_np.shape[0]:
            raise ValueError(
                f"Frame count mismatch: images N={images_np.shape[0]} vs poses N={poses_np.shape[0]} "
                f"({images_path}, {poses_path})"
            )
        images_np, poses_np, frame_indices = self._apply_frame_window(images_np, poses_np)
        self.frame_indices = frame_indices

        images_np = self._prepare_images(images_np)

        # Keep runtime meta resolution aligned with actual file resolution so render kernels
        # and scene bookkeeping always use the loaded data size.
        loaded_h, loaded_w = int(images_np.shape[1]), int(images_np.shape[2])
        self.meta = self._resolve_runtime_meta(meta, loaded_h=loaded_h, loaded_w=loaded_w)

        images = torch.from_numpy(images_np).float()
        poses = torch.from_numpy(poses_np).float()

        # 3. Adjust Pose to ensure probe->world
        poses = self._adjust_poses(poses)
        self.file_poses = poses.detach().clone()
        poses = self._resolve_pose_mode(poses)

        # 4. Build SliceMeta
        self._slice_meta_list = []
        print(f"[SingleSweepContainer] Precomputing SliceMeta...")
        runtime_meta = self.meta
        if runtime_meta.probe_type == "linear":
            slice_meta_template = LinearSliceMeta(runtime_meta, device)
        elif runtime_meta.probe_type == "convex":
            slice_meta_template = ConvexSliceMeta(runtime_meta, device)
        else:
            raise ValueError(f"Unsupported probe type: {runtime_meta.probe_type}")
        for i in tqdm(range(images.shape[0]), unit="frame"):
            slice_meta = slice_meta_template.__class__(runtime_meta, device).initialize_from_image_pose(
                image=images[i], 
                pose=poses[i], 
                frame_id=int(frame_indices[i])
            )
            self._slice_meta_list.append(slice_meta)

        print(f"[SingleSweepContainer] Ready. Loaded {len(self._slice_meta_list)} frames.")

    def __len__(self) -> int:
        return len(self._slice_meta_list)

    def get(self, idx: int) -> SliceMeta:
        return self._slice_meta_list[int(idx)]

    def _apply_frame_window(
        self,
        images_np: np.ndarray,
        poses_np: np.ndarray,
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        n_frames = int(images_np.shape[0])
        start = int(getattr(self.cfg.loading, "frame_start", 0) or 0)
        end = int(getattr(self.cfg.loading, "frame_end", -1) or -1)
        stride = int(getattr(self.cfg.loading, "frame_stride", 1) or 1)
        if stride <= 0:
            raise ValueError(f"scene.loading.frame_stride must be positive, got {stride}")
        if start < 0:
            start = max(0, n_frames + start)
        if end < 0:
            end = n_frames
        else:
            end = min(end, n_frames)
        start = min(max(start, 0), n_frames)
        if end < start:
            raise ValueError(
                f"Invalid frame window: frame_start={start}, frame_end={end}, n_frames={n_frames}"
            )
        frame_indices = np.arange(start, end, stride, dtype=np.int64)
        if frame_indices.size <= 0:
            raise ValueError(
                f"Frame window selected zero frames: start={start}, end={end}, stride={stride}, n_frames={n_frames}"
            )
        if int(frame_indices.size) != n_frames or start != 0 or end != n_frames or stride != 1:
            print(
                "[SingleSweepContainer] Frame window:"
                f" original_n={n_frames}, start={start}, end={end}, stride={stride},"
                f" selected_n={int(frame_indices.size)}"
            )
            images_np = images_np[frame_indices]
            poses_np = poses_np[frame_indices]

        indices_path = str(getattr(self.cfg.loading, "frame_indices_path", "") or "").strip()
        if indices_path:
            explicit = self._load_frame_indices(indices_path)
            if explicit.size <= 0:
                raise ValueError(f"Frame indices file selected zero frames: {indices_path}")
            if np.any(explicit < 0) or np.any(explicit >= n_frames):
                raise ValueError(
                    f"Frame indices out of range for n_frames={n_frames}:"
                    f" min={int(explicit.min())}, max={int(explicit.max())}, path={indices_path}"
                )
            window_lookup = {int(idx): pos for pos, idx in enumerate(frame_indices.tolist())}
            missing = [int(idx) for idx in explicit.tolist() if int(idx) not in window_lookup]
            if missing:
                preview = ", ".join(str(x) for x in missing[:10])
                raise ValueError(
                    "Explicit frame indices must be within the selected frame window."
                    f" Missing examples: {preview}; path={indices_path}"
                )
            local = np.asarray([window_lookup[int(idx)] for idx in explicit.tolist()], dtype=np.int64)
            print(
                "[SingleSweepContainer] Explicit frame indices:"
                f" path={indices_path}, selected_n={int(explicit.size)},"
                f" first={int(explicit[0])}, last={int(explicit[-1])}"
            )
            images_np = images_np[local]
            poses_np = poses_np[local]
            frame_indices = explicit
        return images_np, poses_np, frame_indices

    def _load_frame_indices(self, path: str) -> np.ndarray:
        abs_path = os.path.abspath(os.path.expanduser(path))
        if not os.path.isfile(abs_path):
            raise FileNotFoundError(f"Missing frame indices file: {abs_path}")
        suffix = os.path.splitext(abs_path)[1].lower()
        if suffix == ".npy":
            values = np.load(abs_path)
        elif suffix == ".json":
            with open(abs_path, "r", encoding="utf-8") as f:
                data = json.load(f)
            values = data["indices"] if isinstance(data, dict) and "indices" in data else data
        else:
            values = np.loadtxt(abs_path, dtype=np.int64)
        values = np.asarray(values, dtype=np.int64).reshape(-1)
        if values.size > 1 and np.any(values[1:] <= values[:-1]):
            raise ValueError(f"Frame indices must be strictly increasing: {abs_path}")
        return values

    def _adjust_poses(self, poses: torch.Tensor) -> torch.Tensor:
        if self.meta.pose_init_type != "probe->world":
            print("[SingleSweepContainer] Pose init type is not 'probe->world', adjusting poses")
            return torch.inverse(poses)
        return poses

    def _resolve_pose_mode(self, poses: torch.Tensor) -> torch.Tensor:
        mode = str(getattr(self.cfg.loading, "pose_mode", "file") or "file").lower()
        if mode in ("file", "gt", "tracked"):
            self.uses_file_poses_for_geometry = True
            return poses
        if mode == "canonical_sweep":
            canonical = self._make_canonical_sweep_poses(
                file_poses=poses,
            )
            delta = self._canonical_sweep_delta_mm()
            spacing_mode = str(getattr(self.cfg.loading, "canonical_sweep_spacing_mode", "uniform") or "uniform")
            print(
                "[SingleSweepContainer] Pose mode canonical_sweep:"
                f" n={int(poses.shape[0])}, delta_mm={delta:.6f},"
                f" spacing='{spacing_mode}',"
                f" ref_frame={self._canonical_sweep_ref_frame(int(poses.shape[0]))},"
                f" axis='{str(getattr(self.cfg.loading, 'canonical_sweep_axis', 'z'))}'"
            )
            if spacing_mode == "uniform":
                self.uses_file_poses_for_geometry = False
                print(
                    "[SingleSweepContainer] Sensorless pose audit:"
                    " poses.npy loaded only for frame count; SliceMeta uses generated canonical poses."
                )
            else:
                self.uses_file_poses_for_geometry = True
                print(
                    "[SingleSweepContainer] Pose audit WARNING:"
                    f" canonical_sweep spacing='{spacing_mode}' uses file-pose spacing diagnostics."
                )
            return canonical
        raise ValueError(
            f"Unsupported scene.loading.pose_mode='{mode}'. "
            "Expected one of: file, gt, tracked, canonical_sweep."
        )

    def _canonical_sweep_delta_mm(self) -> float:
        delta = float(getattr(self.cfg.loading, "canonical_sweep_delta_mm", -1.0) or -1.0)
        if delta <= 0.0:
            delta = float(self.meta.delta_elevational_mm)
        scale = float(getattr(self.cfg.loading, "canonical_sweep_delta_scale", 1.0) or 1.0)
        return delta * scale

    def _canonical_sweep_ref_frame(self, n_frames: int) -> int:
        ref = int(getattr(self.cfg.loading, "canonical_sweep_ref_frame", 0) or 0)
        if ref < 0:
            ref = 0
        if ref >= n_frames:
            raise ValueError(
                f"canonical_sweep_ref_frame={ref} is outside frame range [0, {n_frames - 1}]"
            )
        return ref

    def _canonical_sweep_axis(self, dtype: torch.dtype) -> torch.Tensor:
        raw_axis = str(getattr(self.cfg.loading, "canonical_sweep_axis", "z") or "z").strip().lower()
        sign = -1.0 if raw_axis.startswith("-") else 1.0
        axis_name = raw_axis[1:] if raw_axis[:1] in ("-", "+") else raw_axis
        axis_map = {
            "x": (1.0, 0.0, 0.0),
            "y": (0.0, 1.0, 0.0),
            "z": (0.0, 0.0, 1.0),
        }
        if axis_name not in axis_map:
            raise ValueError(
                f"Unsupported canonical_sweep_axis='{raw_axis}'. "
                "Expected x, y, z, +x, +y, +z, -x, -y, or -z."
            )
        axis = torch.tensor(axis_map[axis_name], dtype=dtype) * sign
        return axis

    def _make_canonical_sweep_offsets(self, file_poses: torch.Tensor) -> torch.Tensor:
        n_frames = int(file_poses.shape[0])
        if n_frames <= 0:
            raise ValueError("Cannot build canonical sweep offsets for an empty sweep.")
        dtype = file_poses.dtype
        spacing_mode = str(getattr(self.cfg.loading, "canonical_sweep_spacing_mode", "uniform") or "uniform").lower()
        delta = self._canonical_sweep_delta_mm()
        ref = self._canonical_sweep_ref_frame(n_frames)
        if spacing_mode == "uniform":
            return (torch.arange(n_frames, dtype=dtype) - float(ref)) * float(delta)

        scale = float(getattr(self.cfg.loading, "canonical_sweep_delta_scale", 1.0) or 1.0)
        translations = file_poses[:, :3, 3].to(dtype=dtype)
        if n_frames == 1:
            return torch.zeros(1, dtype=dtype)
        step_vec = translations[1:] - translations[:-1]
        if spacing_mode == "file_step_norm":
            step = torch.linalg.norm(step_vec, dim=-1)
        elif spacing_mode == "file_axis_projection":
            axis = self._canonical_sweep_axis(dtype=dtype)
            step = step_vec @ axis
        else:
            raise ValueError(
                f"Unsupported canonical_sweep_spacing_mode='{spacing_mode}'. "
                "Expected one of: uniform, file_step_norm, file_axis_projection."
            )

        cumulative = torch.zeros(n_frames, dtype=dtype)
        cumulative[1:] = torch.cumsum(step, dim=0)
        return (cumulative - cumulative[ref]) * float(scale)

    def _make_canonical_sweep_poses(self, file_poses: torch.Tensor) -> torch.Tensor:
        n_frames = int(file_poses.shape[0])
        if n_frames <= 0:
            raise ValueError("Cannot build canonical sweep poses for an empty sweep.")
        dtype = file_poses.dtype
        axis = self._canonical_sweep_axis(dtype=dtype)
        poses = torch.eye(4, dtype=dtype).unsqueeze(0).repeat(n_frames, 1, 1)
        offsets = self._make_canonical_sweep_offsets(file_poses)
        poses[:, :3, 3] = offsets[:, None] * axis[None, :]
        return poses

    def _prepare_images(self, images_np: np.ndarray) -> np.ndarray:
        mode = str(getattr(self.cfg.loading, "image_value_mode", "uint8_255") or "uint8_255")
        nan_value = float(getattr(self.cfg.loading, "image_nan_value", 0.0) or 0.0)

        if mode == "uint8_255":
            images_np = np.nan_to_num(images_np, nan=nan_value, posinf=255.0, neginf=0.0)
            images_np = np.clip(images_np, 0.0, 255.0)
        elif mode == "spine_phantom_shifted_float":
            images_f = np.nan_to_num(images_np.astype(np.float32, copy=False), nan=-0.125, posinf=1.0, neginf=-0.125)
            images_np = np.clip(images_f + 0.125, 0.0, 1.0) * 255.0
        else:
            raise ValueError(
                f"Unsupported scene.loading.image_value_mode='{mode}'. "
                "Expected one of: uint8_255, spine_phantom_shifted_float."
            )

        image_factor = int(getattr(self.cfg.loading, "image_downsample_factor", 1) or 1)
        if image_factor < 1:
            raise ValueError(f"image_downsample_factor must be >= 1, got {image_factor}")
        if image_factor > 1:
            images_np = self._downsample_image(images_np, image_factor)

        axial_factor = int(getattr(self.cfg.loading, "axial_downsample_factor", 1) or 1)
        if axial_factor < 1:
            raise ValueError(f"axial_downsample_factor must be >= 1, got {axial_factor}")
        if axial_factor > 1:
            images_np = self._downsample_axial(images_np, axial_factor)
        return images_np.astype(np.float32, copy=False)

    def _downsample_image(self, images_np: np.ndarray, factor: int) -> np.ndarray:
        n, h, w = images_np.shape
        out_h = h // factor
        out_w = w // factor
        if out_h <= 0 or out_w <= 0:
            raise ValueError(f"Cannot downsample image shape {(h, w)} by factor={factor}")
        trimmed_h = out_h * factor
        trimmed_w = out_w * factor
        if trimmed_h != h or trimmed_w != w:
            print(
                "[SingleSweepContainer] Image downsample trims borders:"
                f" (H,W)=({h},{w}) -> ({trimmed_h},{trimmed_w}) before factor={factor}"
            )
            images_np = images_np[:, :trimmed_h, :trimmed_w]
        images_f = images_np.astype(np.float32, copy=False)
        out = images_f.reshape(n, out_h, factor, out_w, factor).mean(axis=(2, 4))
        print(f"[SingleSweepContainer] Image downsample: (H,W)=({h},{w}) -> ({out_h},{out_w}) (factor={factor})")
        return out

    def _downsample_axial(self, images_np: np.ndarray, factor: int) -> np.ndarray:
        n, h, w = images_np.shape
        out_h = h // factor
        if out_h <= 0:
            raise ValueError(f"Cannot downsample H={h} by factor={factor}")
        trimmed_h = out_h * factor
        if trimmed_h != h:
            print(
                "[SingleSweepContainer] Axial downsample trims bottom rows:"
                f" H={h} -> {trimmed_h} before factor={factor}"
            )
            images_np = images_np[:, :trimmed_h, :]
        out = images_np.reshape(n, out_h, factor, w).mean(axis=2)
        print(f"[SingleSweepContainer] Axial downsample: H={h} -> {out_h} (factor={factor})")
        return out

    def _resolve_runtime_meta(self, base_meta: DatasetMeta, loaded_h: int, loaded_w: int) -> DatasetMeta:
        if int(base_meta.H) == loaded_h and int(base_meta.W) == loaded_w:
            return base_meta

        old_h, old_w = int(base_meta.H), int(base_meta.W)
        new_delta_lateral = float(base_meta.delta_lateral_mm) * float(old_w) / float(loaded_w)
        new_delta_axial = float(base_meta.delta_axial_mm) * float(old_h) / float(loaded_h)
        runtime_meta = replace(
            base_meta,
            H=int(loaded_h),
            W=int(loaded_w),
            delta_lateral_mm=new_delta_lateral,
            delta_axial_mm=new_delta_axial,
        )
        print(
            "[SingleSweepContainer] Runtime resolution override:"
            f" ({old_h},{old_w}) -> ({loaded_h},{loaded_w})"
        )
        return runtime_meta
