import math

import pytest

from methods.ultrags.local_window_manager import LocalWindowManager, build_budget_table


def test_build_budget_table_shape_and_first_window_values():
    n, t, w = 5, 100, 3
    c = build_budget_table(n, t, w)
    s = t / n

    assert len(c) == n
    assert len(c[0]) == n - w + 1

    # j=0: S/(i+1), i in [0, W-1]
    assert math.isclose(c[0][0], s / 1)
    assert math.isclose(c[1][0], s / 2)
    assert math.isclose(c[2][0], s / 3)
    assert math.isclose(c[3][0], 0.0)


def test_build_budget_table_w_equals_one():
    n, t, w = 4, 80, 1
    c = build_budget_table(n, t, w)
    s = t / n
    for i in range(n):
        for j in range(n):
            expected = s if i == j else 0.0
            assert math.isclose(c[i][j], expected)


def test_build_budget_table_invalid_args():
    with pytest.raises(ValueError):
        build_budget_table(5, 10, 0)
    with pytest.raises(ValueError):
        build_budget_table(5, 10, 6)
    with pytest.raises(ValueError):
        build_budget_table(5, -1, 3)


def test_k_ind_and_k_joint_follow_window_rules():
    # Pick ratios that make rounding behavior easy to inspect.
    lwm = LocalWindowManager(N=5, T=100, W=3, ratio_front=0.5, ratio_rear=0.5)

    # j=0: all frames in window have both ind/joint from split ratio.
    assert lwm.k_ind(0, 0) > 0
    assert lwm.k_joint(0, 0) > 0
    assert lwm.k_ind(1, 0) > 0
    assert lwm.k_joint(1, 0) > 0
    assert lwm.k_ind(3, 0) == 0
    assert lwm.k_joint(3, 0) == 0

    # j>=1: only rightmost frame has ind>0.
    j = 2
    t_new = lwm.t_new(j)  # 4
    assert lwm.k_ind(t_new, j) > 0
    assert lwm.k_ind(j, j) == 0

    # j>=1: joint has all old frames full ci, rightmost frame rear-split.
    assert lwm.k_joint(j, j) > 0
    assert lwm.k_joint(j + 1, j) > 0
    assert lwm.k_joint(t_new, j) >= 0


def test_joint_pool_and_total_joint_steps_consistent():
    lwm = LocalWindowManager(N=6, T=120, W=3, ratio_front=0.4, ratio_rear=0.6)
    j = 1
    pool = lwm.joint_pool(j)
    assert pool  # non-empty
    assert lwm.total_joint_steps(j) == sum(pool.values())
    # pool keys are in current window range
    assert set(pool).issubset(set(range(j, j + lwm.W)))


def test_lwm_handles_empty_stream_safely():
    lwm = LocalWindowManager(N=0, T=100, W=3, ratio_front=0.5, ratio_rear=0.5)
    assert lwm.M == 0
    assert lwm.c_ij(0, 0) == 0.0
    assert lwm.joint_pool(0) == {}
    assert lwm.total_joint_steps(0) == 0
