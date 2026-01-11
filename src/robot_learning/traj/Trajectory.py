"""任务空间轨迹规划器。"""

from typing import Tuple

import numpy as np


class DecoupledQuinticTrajectory:
    """三轴解耦五次多项式轨迹规划器。

    提供 x/y/z 独立的五次多项式轨迹。
    """

    def __init__(self, start_pos: np.ndarray, target_pos: np.ndarray, duration: float):
        """初始化轨迹规划器。

        Args:
            start_pos: 初始位置 (x0, y0, z0)。
            target_pos: 目标位置 (xf, yf, zf)。
            duration: 轨迹持续时间（秒）。
        """
        assert start_pos.shape == (3,), "Start position must be 3D vector"
        assert target_pos.shape == (3,), "Target position must be 3D vector"
        assert duration > 0, "Duration must be positive"

        self.p0 = start_pos
        self.pf = target_pos
        self.T = duration

        self.ax = self._solve_quintic_coefficients(start_pos[0], target_pos[0])
        self.ay = self._solve_quintic_coefficients(start_pos[1], target_pos[1])
        self.az = self._solve_quintic_coefficients(start_pos[2], target_pos[2])

        self.coefficients = np.vstack([self.ax, self.ay, self.az])

    def _solve_quintic_coefficients(self, p0: float, pf: float) -> np.ndarray:
        """求解单轴五次多项式系数。

        Args:
            p0: 初始位置。
            pf: 目标位置。

        Returns:
            五次多项式系数向量。
        """
        A = np.array(
            [
                [0, 0, 0, 0, 0, 1],
                [self.T**5, self.T**4, self.T**3, self.T**2, self.T, 1],
                [0, 0, 0, 0, 1, 0],
                [5 * self.T**4, 4 * self.T**3, 3 * self.T**2, 2 * self.T, 1, 0],
                [0, 0, 0, 2, 0, 0],
                [20 * self.T**3, 12 * self.T**2, 6 * self.T, 2, 0, 0],
            ]
        )

        b = np.array([p0, pf, 0, 0, 0, 0])

        return np.linalg.solve(A, b)

    def get_state(self, t: float) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """获取时刻 t 的位置、速度与加速度。

        Args:
            t: 当前时间（秒）。

        Returns:
            位置、速度与加速度向量。
        """
        t = np.clip(t, 0, self.T)

        pos = np.zeros(3)
        vel = np.zeros(3)
        acc = np.zeros(3)

        t_pos = np.array([t**5, t**4, t**3, t**2, t, 1])
        t_vel = np.array([5 * t**4, 4 * t**3, 3 * t**2, 2 * t, 1, 0])
        t_acc = np.array([20 * t**3, 12 * t**2, 6 * t, 2, 0, 0])

        for i in range(3):
            pos[i] = self.coefficients[i] @ t_pos
            vel[i] = self.coefficients[i] @ t_vel
            acc[i] = self.coefficients[i] @ t_acc

        return pos, vel, acc

    def verify_boundary_conditions(self, tol: float = 1e-10) -> bool:
        """验证端点位置、速度与加速度约束。

        Args:
            tol: 比较容差。

        Returns:
            是否满足全部约束。
        """
        pos_start, vel_start, acc_start = self.get_state(0)
        pos_end, vel_end, acc_end = self.get_state(self.T)

        conditions = [
            np.allclose(pos_start, self.p0, atol=tol),
            np.allclose(pos_end, self.pf, atol=tol),
            np.allclose(vel_start, np.zeros(3), atol=tol),
            np.allclose(vel_end, np.zeros(3), atol=tol),
            np.allclose(acc_start, np.zeros(3), atol=tol),
            np.allclose(acc_end, np.zeros(3), atol=tol),
        ]

        return all(conditions)
