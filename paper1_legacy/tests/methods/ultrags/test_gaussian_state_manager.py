"""
GaussianStateManager 与 online_callbacks 主 pipeline 的对应关系（摘要）：

- setup：`initialize(num_points, birth_frame_id)` 与空容器对齐；`membership_k<=0` 时关闭 GSM。
- `run_concat_seed`：`on_concat(n_new, frame_id)` 为新高斯首槽写入出生帧。
- `gs_optim_one_step`：可选 `gauss_indices_for_sub(i_right, W)`（`sub_locality_l>0`）筛子集；
  每步后 `_update_membership_from_render` 用 `seen_gauss_global_indices` 更新共视槽位。
- `run_pose_wash`：`get_aux_render_indices(frame_id, num_points)` 供辅助渲染排除「首槽=当前帧」的高斯。

本文件只测 GSM 纯逻辑，不依赖 scene/render。
"""

import pytest
import torch

from methods.ultrags.gaussian_state_manager import (
    MembershipBasedGaussianStateManager,
    SliceIdentityBasedGaussianStateManager,
    SpatialChunkBasedGaussianStateManager,
)


@pytest.fixture
def cpu():
    return torch.device("cpu")


def test_membership_k_zero_disables_gsm(cpu):
    gsm = MembershipBasedGaussianStateManager(device=cpu, membership_k=0, sub_locality_l=1)
    gsm.initialize(num_points=5, birth_frame_id=0)
    assert gsm.membership is None
    gsm.on_concat(n_new=3, frame_id=1)  # no-op
    assert gsm.membership is None
    assert gsm.gauss_indices_for_sub(i_right=5, W=3) is None
    assert torch.equal(gsm.get_aux_render_indices(frame_id=2, num_points=5), torch.arange(5))


def test_initialize_empty_then_concat(cpu):
    gsm = MembershipBasedGaussianStateManager(device=cpu, membership_k=3, sub_locality_l=0)
    gsm.initialize(num_points=0, birth_frame_id=0)
    assert gsm.membership.shape == (0, 3)

    gsm.on_concat(n_new=2, frame_id=7)
    assert gsm.membership.shape == (2, 3)
    assert torch.equal(gsm.membership[:, 0], torch.tensor([7, 7], dtype=torch.long))
    assert (gsm.membership[:, 1:] == -1).all()


def test_initialize_birth_column(cpu):
    gsm = MembershipBasedGaussianStateManager(device=cpu, membership_k=4, sub_locality_l=0)
    gsm.initialize(num_points=3, birth_frame_id=0)
    assert gsm.membership.shape == (3, 4)
    assert torch.equal(gsm.membership[:, 0], torch.zeros(3, dtype=torch.long))
    assert (gsm.membership[:, 1:] == -1).all()


def test_on_concat_requires_initialized_membership(cpu):
    gsm = MembershipBasedGaussianStateManager(device=cpu, membership_k=2, sub_locality_l=0)
    gsm.initialize(num_points=0, birth_frame_id=0)
    gsm.membership = None  # 非法：k>0 但未初始化
    with pytest.raises(RuntimeError, match="membership"):
        gsm.on_concat(n_new=1, frame_id=0)


def test_update_membership_fills_first_empty_slot(cpu):
    gsm = MembershipBasedGaussianStateManager(device=cpu, membership_k=3, sub_locality_l=0)
    gsm.initialize(num_points=2, birth_frame_id=0)
    # 行 0: [0,-1,-1] -> 见帧 1 后首空槽写 1
    gsm.update_membership_from_render(frame_id=1, seen_gauss_indices=torch.tensor([0], dtype=torch.long))
    assert int(gsm.membership[0, 1].item()) == 1
    assert int(gsm.membership[0, 2].item()) == -1


def test_update_membership_fifo_when_row_full(cpu):
    gsm = MembershipBasedGaussianStateManager(device=cpu, membership_k=3, sub_locality_l=0)
    gsm.initialize(num_points=1, birth_frame_id=0)
    gsm.membership[0] = torch.tensor([0, 1, 2], dtype=torch.long)
    gsm.update_membership_from_render(frame_id=3, seen_gauss_indices=torch.tensor([0], dtype=torch.long))
    assert torch.equal(gsm.membership[0], torch.tensor([1, 2, 3], dtype=torch.long))


def test_gauss_indices_for_sub_none_when_l_non_positive(cpu):
    gsm = MembershipBasedGaussianStateManager(device=cpu, membership_k=2, sub_locality_l=0)
    gsm.initialize(num_points=1, birth_frame_id=0)
    assert gsm.gauss_indices_for_sub(i_right=10, W=3) is None


def test_gauss_indices_for_sub_time_window(cpu):
    gsm = MembershipBasedGaussianStateManager(device=cpu, membership_k=2, sub_locality_l=1)
    gsm.initialize(num_points=4, birth_frame_id=0)
    # 行 i 首槽为帧 i，便于断言区间 [lo, hi]
    gsm.membership[:, 0] = torch.arange(4, dtype=torch.long)
    W = 3
    i_right = 2
    # lo = 2 - 3 + 1 - 1 + 1 = 0, hi = 2 -> 帧 0,1,2 有交
    idx = gsm.gauss_indices_for_sub(i_right=i_right, W=W)
    assert set(idx.tolist()) == {0, 1, 2}


def test_get_aux_render_indices_excludes_primary_frame(cpu):
    gsm = MembershipBasedGaussianStateManager(device=cpu, membership_k=2, sub_locality_l=0)
    gsm.initialize(num_points=4, birth_frame_id=0)
    gsm.membership[:, 0] = torch.tensor([0, 1, 2, 2], dtype=torch.long)
    ex = gsm.get_aux_render_indices(frame_id=2, num_points=4)
    # 行 2、3 首槽为 2 -> 排除；保留 0,1
    assert set(ex.tolist()) == {0, 1}


def test_on_remap(cpu):
    gsm = MembershipBasedGaussianStateManager(device=cpu, membership_k=2, sub_locality_l=0)
    gsm.initialize(num_points=3, birth_frame_id=0)
    keep = torch.tensor([True, False, True], dtype=torch.bool)
    gsm.on_remap(keep)
    assert gsm.membership.shape == (2, 2)
    assert torch.equal(gsm.membership[:, 0], torch.tensor([0, 0], dtype=torch.long))


def test_spatial_chunk_filter_only(cpu):
    gsm = SpatialChunkBasedGaussianStateManager(
        device=cpu,
        spatial_chunk_size_mm=1.0,
        spatial_active_half_width_mm=1.0,
        spatial_scale_margin_ratio=0.0,
    )
    means = torch.tensor(
        [
            [0.0, 0.0, -2.0],
            [0.0, 0.0, -0.5],
            [0.0, 0.0, 0.3],
            [0.0, 0.0, 2.2],
        ],
        dtype=torch.float32,
    )
    scales = torch.ones((4, 3), dtype=torch.float32) * 0.1
    idx = gsm.gauss_indices_for_active_set(
        i_right=0,
        W=3,
        means=means,
        scales=scales,
        slice_probe_origin=torch.zeros(3),
        slice_vec_elevational=torch.tensor([0.0, 0.0, 1.0]),
    )
    assert set(idx.tolist()) == {1, 2}


def test_slice_identity_basic_filter_and_cache(cpu):
    gsm = SliceIdentityBasedGaussianStateManager(device=cpu, exclude_num_ids=3)
    gsm.initialize(num_points=5, birth_frame_id=0)
    gsm.slice_ids = torch.tensor([0, 1, 3, 5, 8], dtype=torch.long)
    idx0 = gsm.gauss_indices_for_sub(i_right=5, W=3)
    idx1 = gsm.gauss_indices_for_sub(i_right=5, W=3)
    assert set(idx0.tolist()) == {2, 3}
    assert torch.equal(idx0, idx1)


def test_slice_identity_concat_inherit_parents(cpu):
    gsm = SliceIdentityBasedGaussianStateManager(device=cpu, exclude_num_ids=10)
    gsm.initialize(num_points=3, birth_frame_id=0)
    gsm.slice_ids = torch.tensor([2, 4, 7], dtype=torch.long)
    gsm.on_concat(
        n_new=3,
        frame_id=99,
        parent_indices=torch.tensor([2, 0, 2], dtype=torch.long),
    )
    assert torch.equal(gsm.slice_ids, torch.tensor([2, 4, 7, 7, 2, 7], dtype=torch.long))


def test_slice_identity_remap_and_aux(cpu):
    gsm = SliceIdentityBasedGaussianStateManager(device=cpu, exclude_num_ids=4)
    gsm.initialize(num_points=4, birth_frame_id=0)
    gsm.slice_ids = torch.tensor([1, 3, 3, 8], dtype=torch.long)
    keep = torch.tensor([True, False, True, True], dtype=torch.bool)
    gsm.on_remap(keep)
    assert torch.equal(gsm.slice_ids, torch.tensor([1, 3, 8], dtype=torch.long))
    ex = gsm.get_aux_render_indices(frame_id=3, num_points=3)
    assert set(ex.tolist()) == {0, 2}


