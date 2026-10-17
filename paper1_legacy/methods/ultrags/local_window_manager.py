"""Local Window Manager：仅预算表 C[i,j] 与 K_ind / K_joint（docs/methods/baselines/ultrags_baseline/online_update_spec.md §2.1）。"""
from __future__ import annotations

from typing import List


def build_budget_table(N: int, T: int, W: int) -> list[list[float]]:
    """
    按 docs/methods/baselines/ultrags_baseline/online_local_window_budget.md 构造每窗每帧预算 C[i][j]（浮点）。

    返回 C[i][j]：帧 i 在窗 j 内的步数预算；i ∉ L_j 时为 0。
    N: 帧数；T: 总优化步；W: 窗长；M = N - W + 1。
    """
    if N == 0:
        return []
    if W < 1 or W > N:
        raise ValueError(f"需要 1 <= W <= N，得到 W={W}, N={N}")
    if T < 0:
        raise ValueError(f"T 必须非负，得到 T={T}")

    S = T / N
    M = N - W + 1
    C = [[0.0 for _ in range(M)] for _ in range(N)]

    if W == 1:
        for j in range(M):
            C[j][j] = S
        return C

    for j in range(M):
        if j == 0:
            for i in range(W):
                C[i][0] = S / (i + 1)
        elif j == 1:
            C[1][1] = S / 2
            for i in range(2, W + 1):
                C[i][1] = S / W
        else:
            for i in range(j, j + W):
                C[i][j] = S / W
    return C


def _round_steps(x: float) -> int:
    return max(0, int(round(x)))


class LocalWindowManager:
    def __init__(
        self,
        N: int,
        T: int,
        W: int,
        ratio_front: float,
        ratio_rear: float,
    ):
        self.N = int(N)
        self.T = int(T)
        w = int(W)
        if self.N > 0:
            self.W = min(w, max(1, self.N))
        else:
            self.W = max(1, w)
        self.M = max(0, self.N - self.W + 1) if self.N > 0 else 0
        self.r_front = float(ratio_front)
        self.r_rear = float(ratio_rear)
        if self.N > 0 and self.M > 0:
            self._C: List[List[float]] = build_budget_table(self.N, self.T, self.W)
        else:
            self._C = []

    def c_ij(self, i: int, j: int) -> float:
        if self.M <= 0 or not self._C:
            return 0.0
        if i < 0 or i >= self.N or j < 0 or j >= self.M:
            return 0.0
        return float(self._C[i][j])

    def t_new(self, j: int) -> int:
        return j + self.W - 1

    def k_ind(self, i: int, j: int) -> int:
        """独立段子步数（整数）。"""
        ci = self.c_ij(i, j)
        if ci <= 0.0:
            return 0
        if j == 0:
            if i < 0 or i >= self.W:
                return 0
            return _round_steps(ci * self.r_front)
        tn = self.t_new(j)
        if i != tn:
            return 0
        return _round_steps(ci * self.r_front)

    def k_joint(self, i: int, j: int) -> int:
        """联合段中帧 i 的子步数（整数）。"""
        ci = self.c_ij(i, j)
        if ci <= 0.0:
            return 0
        if j == 0:
            if i < 0 or i >= self.W:
                return 0
            return _round_steps(ci * self.r_rear)
        tn = self.t_new(j)
        if i == tn:
            return _round_steps(ci * self.r_rear)
        if j <= i <= j + self.W - 2:
            return _round_steps(ci)
        return 0

    def joint_pool(self, j: int) -> dict[int, int]:
        """联合段：{帧 i: 剩余子步数}。"""
        if j < 0 or j >= self.M:
            return {}
        rem: dict[int, int] = {}
        for i in range(j, min(j + self.W, self.N)):
            k = self.k_joint(i, j)
            if k > 0:
                rem[i] = k
        return rem

    def total_joint_steps(self, j: int) -> int:
        return sum(self.joint_pool(j).values())

    def window_frame_ids(self, j: int) -> list[int]:
        """窗口 j 的成员帧下标列表 L_j。"""
        if self.N <= 0 or self.M <= 0:
            return []
        jj = max(0, min(int(j), self.M - 1))
        lo = jj
        hi = min(jj + self.W - 1, self.N - 1)
        return list(range(lo, hi + 1))
