"""Pinocchio 任务空间控制器。

提供平动控制与含姿态的阻抗控制接口。
"""

from __future__ import annotations

from typing import Tuple

import numpy as np
import pinocchio as pin


class PinocchioTaskSpaceController:
    """基于 Pinocchio 的任务空间控制器。

    Args:
        model: Pinocchio 机器人模型。
        dt: 控制周期（秒）。
        ee_frame_name: 末端执行器 Frame 名称。
        initial_orientation: 期望初始姿态矩阵。
    """

    def __init__(
        self,
        model: pin.Model,
        dt: float,
        ee_frame_name: str = "cylinder_link",
        initial_orientation: np.ndarray | None = None,
    ):
        """初始化控制器。

        Args:
            model: Pinocchio 机器人模型。
            dt: 控制周期（秒）。
            ee_frame_name: 末端执行器 Frame 名称。
            initial_orientation: 期望初始姿态矩阵。

        Raises:
            ValueError: 末端 Frame 名称无效时抛出。
        """
        self.model = model
        self.data = self.model.createData()
        self.dt = dt

        self.ee_frame_name = ee_frame_name
        self.ee_frame_id = self.model.getFrameId(self.ee_frame_name)
        if self.ee_frame_id is None or self.ee_frame_id >= len(self.model.frames):
            raise ValueError(f"Frame '{self.ee_frame_name}' not found in model.")

        # 关节限位（若模型提供则使用）
        self.q_min = getattr(self.model, "lowerPositionLimit", None)
        self.q_max = getattr(self.model, "upperPositionLimit", None)
        self.v_max = getattr(self.model, "velocityLimit", None)

        if initial_orientation is None:
            self.initial_orientation = np.eye(3)
            # 固定的初始期望姿态
        else:
            self.initial_orientation = np.array(initial_orientation, dtype=float)

        self.J_prev = None
        self.J_dot = None

    def _update_kinematics(self, q: np.ndarray, v: np.ndarray) -> None:
        """更新运动学与雅可比缓存。"""
        pin.forwardKinematics(self.model, self.data, q, v)
        pin.updateFramePlacements(self.model, self.data)
        pin.computeJointJacobians(self.model, self.data, q)
        pin.computeJointJacobiansTimeVariation(self.model, self.data, q, v)

    def _compute_jacobian(self, update_prev: bool = True) -> np.ndarray:
        """计算末端雅可比及其数值导数。

        Args:
            update_prev: 是否更新上一帧雅可比缓存。

        Returns:
            世界坐标系下的 6xN 雅可比矩阵。
        """
        J = pin.getFrameJacobian(self.model, self.data, self.ee_frame_id, pin.ReferenceFrame.LOCAL_WORLD_ALIGNED)
        if update_prev:
            if self.J_prev is None:
                self.J_dot = np.zeros_like(J)
            else:
                self.J_dot = (J - self.J_prev) / self.dt
            self.J_prev = J.copy()
        return J

    def _log_map(self, R_current: np.ndarray, R_desired: np.ndarray) -> np.ndarray:
        """计算旋转误差的对数映射。"""
        return pin.log3(R_desired @ R_current.T)

    def _apply_limits(self, tau: np.ndarray, q: np.ndarray, v: np.ndarray) -> np.ndarray:
        """应用软限位与速度缩放。"""
        if self.q_min is None or self.q_max is None:
            return tau

        k_limit = 100.0
        tau_limit = np.zeros_like(tau)
        n = min(len(q), len(self.q_min))
        for i in range(n):
            if self.q_min[i] < self.q_max[i]:
                if q[i] < self.q_min[i]:
                    tau_limit[i] = k_limit * (self.q_min[i] - q[i])
                elif q[i] > self.q_max[i]:
                    tau_limit[i] = k_limit * (self.q_max[i] - q[i])

        if self.v_max is not None:
            v_scale = np.minimum(1.0, self.v_max[: len(v)] / (np.abs(v) + 1e-6))
            tau = tau * v_scale

        return tau + tau_limit

    def get_task_space_state(self, q: np.ndarray, v: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """返回末端位置与线速度。

        Args:
            q: 关节位置。
            v: 关节速度。

        Returns:
            末端位置与线速度。
        """
        self._update_kinematics(q, v)
        H = self.data.oMf[self.ee_frame_id]
        pos_cur = H.translation

        J = self._compute_jacobian(update_prev=False)
        J_pos = J[:3, :]
        vel_cur = J_pos @ v
        return pos_cur.copy(), vel_cur.copy()

    def get_task_space_state_with_orientation(
        self, q: np.ndarray, v: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """返回末端位置、速度与姿态误差。

        Args:
            q: 关节位置。
            v: 关节速度。

        Returns:
            位置、线速度、姿态误差与角速度。
        """
        self._update_kinematics(q, v)
        H = self.data.oMf[self.ee_frame_id]
        pos_cur = H.translation
        rot_cur = H.rotation

        ori_err = self._log_map(rot_cur, self.initial_orientation)

        J = self._compute_jacobian(update_prev=False)
        J_pos = J[:3, :]
        J_rot = J[3:, :]
        vel_pos = J_pos @ v
        vel_rot = J_rot @ v
        return pos_cur.copy(), vel_pos.copy(), ori_err.copy(), vel_rot.copy()

    def compute_control_task_space(
        self,
        q: np.ndarray,
        dq: np.ndarray,
        desired_ee_pos: np.ndarray,
        desired_ee_vel: np.ndarray,
        desired_ee_acc: np.ndarray,
    ) -> np.ndarray:
        """计算含重力补偿的任务空间控制。

        Args:
            q: 关节位置。
            dq: 关节速度。
            desired_ee_pos: 期望末端位置。
            desired_ee_vel: 期望末端速度。
            desired_ee_acc: 期望末端加速度。

        Returns:
            关节力矩。
        """
        self._update_kinematics(q, dq)

        pos_cur, vel_pos_cur, ori_err, vel_rot_cur = self.get_task_space_state_with_orientation(q, dq)
        vel_cur = np.concatenate([vel_pos_cur, vel_rot_cur])

        acc_des_full = np.concatenate([desired_ee_acc, np.zeros(3)])
        vel_des_full = np.concatenate([desired_ee_vel, np.zeros(3)])

        J = self._compute_jacobian(update_prev=True)

        M = pin.crba(self.model, self.data, q)
        M = 0.5 * (M + M.T)
        M_inv = np.linalg.inv(M + 1e-8 * np.eye(self.model.nv))

        J_dot = pin.getFrameJacobianTimeVariation(
            self.model, self.data, self.ee_frame_id, pin.ReferenceFrame.LOCAL_WORLD_ALIGNED
        )

        lambda_inv = J @ M_inv @ J.T
        omega = np.linalg.inv(lambda_inv + 1e-8 * np.eye(6))

        J_pinv = np.linalg.pinv(J)
        v_ref = J_pinv @ vel_des_full
        pin.computeCoriolisMatrix(self.model, self.data, q, dq)
        term1 = J @ M_inv @ (self.data.C @ v_ref)
        term2 = J_dot @ v_ref
        mu_vel_des = omega @ (term1 - term2)

        k_p = np.concatenate([np.full(3, 100.0), np.full(3, 50.0)])
        k_d = np.concatenate([np.full(3, 20.0), np.full(3, 10.0)])

        err_pos = np.concatenate([desired_ee_pos - pos_cur, ori_err])
        err_vel = vel_des_full - vel_cur

        f_cmd = (omega @ acc_des_full) + mu_vel_des + (k_d * err_vel) + (k_p * err_pos)
        tau = J.T @ f_cmd

        g = pin.computeGeneralizedGravity(self.model, self.data, q)
        tau = tau + g

        tau = self._apply_limits(tau, q, dq)
        return tau

    def compute_control_task_space_with_orientation_and_imp(
        self,
        q: np.ndarray,
        dq: np.ndarray,
        desired_ee_pos: np.ndarray,
        desired_ee_vel: np.ndarray,
        desired_ee_acc: np.ndarray,
        force_external: np.ndarray | None = None,
    ) -> np.ndarray:
        """计算含姿态跟踪的阻抗控制。

        Args:
            q: 关节位置。
            dq: 关节速度。
            desired_ee_pos: 期望末端位置。
            desired_ee_vel: 期望末端速度。
            desired_ee_acc: 期望末端加速度。
            force_external: 任务空间外力。

        Returns:
            关节力矩。
        """
        self._update_kinematics(q, dq)

        pos_cur, vel_pos_cur, ori_err, vel_rot_cur = self.get_task_space_state_with_orientation(q, dq)

        J = self._compute_jacobian(update_prev=True)
        J_pos = J[:3, :]
        J_rot = J[3:, :]

        M = pin.crba(self.model, self.data, q)
        M = 0.5 * (M + M.T)
        M_inv = np.linalg.inv(M + 1e-8 * np.eye(self.model.nv))
        h = pin.nonLinearEffects(self.model, self.data, q, dq)

        force_external = np.zeros(3) if force_external is None else np.array(force_external)

        m_pos = 10.0
        d_pos = 100.0
        k_pos = 1000.0

        m_rot = 1.0
        d_rot = 15.0
        k_rot = 30.0

        pos_err = pos_cur - desired_ee_pos
        vel_err = vel_pos_cur - desired_ee_vel
        u_pos = desired_ee_acc + (force_external - d_pos * vel_err - k_pos * pos_err) / m_pos

        vel_rot_err = -vel_rot_cur
        u_rot = (k_rot * ori_err + d_rot * vel_rot_err) / m_rot

        u_task = np.concatenate([u_pos, u_rot])

        lambda_inv = J @ M_inv @ J.T
        lambda_mat = np.linalg.inv(lambda_inv + 1e-8 * np.eye(6))
        J_bar = M_inv @ J.T @ lambda_mat
        N = np.eye(self.model.nv) - J_bar @ J

        D_null = 10.0 * np.eye(self.model.nv)
        tau_null = N @ (-D_null @ dq)

        if self.J_dot is None:
            J_dot_dq = np.zeros(6)
        else:
            J_dot_dq = self.J_dot @ dq

        tau = J.T @ lambda_mat @ (u_task - J_dot_dq + J @ M_inv @ h) + tau_null
        tau = self._apply_limits(tau, q, dq)
        return tau
