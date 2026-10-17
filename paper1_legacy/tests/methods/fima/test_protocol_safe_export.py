from types import SimpleNamespace

import numpy as np
import pytest
import torch

from methods.fima.configs import FiMAConfig
from methods.fima.pem_pose_export import _compose_chain_from_edges, _export_initial_pose_anchor
from methods.fima.sequence_dataset import FiMASequenceSampler


class _FakeSlice:
    def __init__(self, frame_idx: int, pose: torch.Tensor):
        self.frame_id = frame_idx
        self.image = torch.full((2, 3), float(frame_idx), dtype=torch.float32)
        self.pose = pose


class _FakeContainer:
    def __init__(self, n_frames: int):
        self.meta = SimpleNamespace(H=2, W=3)
        self._slices = []
        for i in range(n_frames):
            pose = torch.eye(4, dtype=torch.float32)
            pose[0, 3] = float(i)
            pose[1, 3] = float(i) * 0.5
            self._slices.append(_FakeSlice(i, pose))

    def __len__(self):
        return len(self._slices)

    def get(self, idx: int):
        return self._slices[int(idx)]


def _cfg(input_mode: str = "multiframe", window_size: int = 4) -> FiMAConfig:
    cfg = FiMAConfig()
    cfg.pem.input_mode = input_mode
    cfg.pem.window_size = window_size
    cfg.pem.pem_output_layout = "window_edges"
    cfg.sequence.respect_scene_holdout = True
    cfg.sequence.pad_mode = "repeat_edge"
    return cfg


def test_compose_chain_uses_absolute_initial_pose():
    edge_pred = {}
    for i in range(2):
        T = torch.eye(4)
        T[0, 3] = 1.0
        edge_pred[i] = T
    anchor = torch.eye(4)
    anchor[0, 3] = 10.0

    out = _compose_chain_from_edges(edge_pred, 3, torch.device("cpu"), initial_pose=anchor)

    assert torch.allclose(out[0], anchor)
    assert torch.allclose(out[:, 0, 3], torch.tensor([10.0, 11.0, 12.0]))


def test_export_anchor_keeps_relative_identity_and_absolute_first_pose():
    container = _FakeContainer(3)
    state = SimpleNamespace(
        device=torch.device("cpu"),
        sequence_sampler=SimpleNamespace(container=container),
        cfg=SimpleNamespace(scene=SimpleNamespace(loading=SimpleNamespace(pose_type="relative"))),
    )
    rel_anchor = _export_initial_pose_anchor(state)
    assert torch.allclose(rel_anchor, torch.eye(4))

    state.cfg.scene.loading.pose_type = "absolute"
    abs_anchor = _export_initial_pose_anchor(state)
    assert torch.allclose(abs_anchor, container.get(0).pose)


def test_multiframe_holdout_rejects_compressed_fake_timeline():
    container = _FakeContainer(6)
    cfg = _cfg(input_mode="multiframe", window_size=4)

    with pytest.raises(RuntimeError, match="global-contiguous train windows"):
        FiMASequenceSampler(
            container,
            cfg,
            torch.device("cpu"),
            train_global_indices=[0, 1, 2, 4, 5],
            test_global_indices=[3],
        )


def test_multiframe_holdout_samples_only_global_consecutive_windows():
    container = _FakeContainer(6)
    cfg = _cfg(input_mode="multiframe", window_size=4)
    sampler = FiMASequenceSampler(
        container,
        cfg,
        torch.device("cpu"),
        train_global_indices=[0, 1, 2, 3, 5],
        test_global_indices=[4],
    )

    assert sampler._holdout_global_windows == [[0, 1, 2, 3]]

    batch = sampler.sample_train_batch(2, np.random.RandomState(0))
    assert batch.frame_idx_end.tolist() == [3, 3]
    assert torch.allclose(batch.T_gt_vec_edges[..., 0], torch.ones(2, 3))
    assert torch.allclose(batch.T_gt_vec_edges[..., 1], torch.full((2, 3), 0.5))


def test_adjacent_pair_holdout_edges_never_cross_test_frame():
    container = _FakeContainer(6)
    cfg = _cfg(input_mode="adjacent_pair", window_size=2)
    sampler = FiMASequenceSampler(
        container,
        cfg,
        torch.device("cpu"),
        train_global_indices=[0, 1, 2, 4, 5],
        test_global_indices=[3],
    )

    assert sampler._global_consecutive_train_edges == [(0, 1), (1, 2), (4, 5)]
    with pytest.raises(ValueError, match="global consecutive"):
        sampler.collate_adjacent_edges([(2, 4)])


def test_block_holdout_groups_val_frames_and_preserves_train_windows():
    container = _FakeContainer(30)
    cfg = _cfg(input_mode="multiframe", window_size=8)
    cfg.sequence.split_protocol = "block_holdout"
    cfg.sequence.respect_scene_holdout = False
    cfg.sequence.block_holdout_num_samples = 6
    cfg.sequence.block_holdout_block_size = 3
    cfg.sequence.block_holdout_trim_each = 2

    sampler = FiMASequenceSampler(container, cfg, torch.device("cpu"))

    assert sorted(sampler.val_indices) == [2, 3, 4, 25, 26, 27]
    assert all(2 not in w and 25 not in w for w in sampler._holdout_global_windows)
    assert sampler._holdout_global_windows


def test_block_holdout_keeps_more_phase_a_windows_than_interval_like_holdout():
    container = _FakeContainer(40)
    cfg_interval = _cfg(input_mode="adjacent_pair", window_size=2)
    interval_test = [4, 8, 12, 16, 20, 24, 28, 32]
    interval_train = [i for i in range(40) if i not in set(interval_test)]
    interval = FiMASequenceSampler(
        container,
        cfg_interval,
        torch.device("cpu"),
        train_global_indices=interval_train,
        test_global_indices=interval_test,
    )

    cfg_block = _cfg(input_mode="adjacent_pair", window_size=2)
    cfg_block.sequence.split_protocol = "block_holdout"
    cfg_block.sequence.respect_scene_holdout = False
    cfg_block.sequence.block_holdout_num_samples = 8
    cfg_block.sequence.block_holdout_block_size = 4
    cfg_block.sequence.block_holdout_trim_each = 4
    block = FiMASequenceSampler(container, cfg_block, torch.device("cpu"))

    assert len(block._train_global_windows) > len(interval._train_global_windows)
