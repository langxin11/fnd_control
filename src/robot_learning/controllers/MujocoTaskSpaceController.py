"""MuJoCo 任务空间动力学控制器。

提供基于 MuJoCo API 的任务空间控制能力。
"""

import mujoco
import numpy as np


class MujocoTaskSpaceController:
    """MuJoCo 任务空间动力学控制器。

    Args:
        model: MuJoCo 模型实例。
        data: MuJoCo 数据实例。
        ee_site_name: 末端执行器 site 或 body 名称。
        dt: 仿真时间步长（秒）。
    """

    def __init__(self, model, data, ee_site_name="attachment_site", dt=0.002):
        """初始化控制器。

        Args:
            model: MuJoCo 模型实例。
            data: MuJoCo 数据实例。
            ee_site_name: 末端执行器 site 或 body 名称。
            dt: 仿真时间步长（秒）。
        """
        self.model = model
        self.data = data
        self.ee_site_name = ee_site_name
        self.dt = dt

        self.ee_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_SITE, self.ee_site_name)
        if self.ee_id != -1:
            self.obj_type = mujoco.mjtObj.mjOBJ_SITE
        else:
            print(f"Site {ee_site_name} not found, trying Body...")
            self.ee_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, self.ee_site_name)
            if self.ee_id == -1:
                raise ValueError(f"End-effector '{ee_site_name}' not found as site or body.")
            self.obj_type = mujoco.mjtObj.mjOBJ_BODY

        self.nv = self.model.nv
        self.nq = self.model.nq

        # 初始化用于数值差分的雅可比缓存
        self.J_prev = np.zeros((6, self.nv))

        # 机器人限位（iiwa14，可从模型读取）
        self.q_min = np.array([-2.96706, -2.0944, -2.96706, -2.0944, -2.96706, -2.0944, -3.05433])
        self.q_max = np.array([2.96706, 2.0944, 2.96706, 2.0944, 2.96706, 2.0944, 3.05433])
        self.v_max = np.array([1.4835, 1.4835, 1.7453, 1.3090, 2.2689, 2.3562, 2.3562])

        # 固定的初始期望姿态
        self.initial_orientation = np.array([[1, 0, 0], [0, -1, 0], [0, 0, -1]])

    def _get_jacobian(self, update_prev=True):
        """计算末端雅可比矩阵。

        Args:
            update_prev: 是否更新雅可比导数缓存。

        Returns:
            6xN 雅可比矩阵。
        """
        jacp = np.zeros((3, self.model.nv))
        jacr = np.zeros((3, self.model.nv))
        if self.obj_type == mujoco.mjtObj.mjOBJ_SITE:
            mujoco.mj_jacSite(self.model, self.data, jacp, jacr, self.ee_id)
        else:
            mujoco.mj_jacBodyCom(self.model, self.data, jacp, jacr, self.ee_id)
        J = np.vstack((jacp, jacr))

        if update_prev:
            # 数值计算 J_dot
            self.J_dot = (J - self.J_prev) / self.dt
            self.J_prev = J.copy()

        return J

    def _log_map(self, R_current, R_desired):
        """计算旋转矩阵的对数映射。

        Args:
            R_current: 当前旋转矩阵。
            R_desired: 期望旋转矩阵。

        Returns:
            3D 旋转向量。
        """
        # 计算相对旋转矩阵
        R_rel = R_desired @ R_current.T
        # 将旋转矩阵转换为轴角表示
        angle = np.arccos(np.clip((np.trace(R_rel) - 1) / 2, -1.0, 1.0))
        if angle < 1e-6:
            return np.zeros(3)
        axis = (1 / (2 * np.sin(angle))) * np.array(
            [R_rel[2, 1] - R_rel[1, 2], R_rel[0, 2] - R_rel[2, 0], R_rel[1, 0] - R_rel[0, 1]]
        )
        return angle * axis

    def _pseudo_inverse(self, J, damp=1e-4):
        """计算雅可比矩阵的阻尼伪逆。

        Args:
            J: 雅可比矩阵。
            damp: 阻尼系数。

        Returns:
            伪逆矩阵。
        """
        JT = J.T
        JJT = J @ JT
        if np.linalg.matrix_rank(JJT) < JJT.shape[0]:
            # 使用正则化的伪逆
            lambda_reg = damp
            J_pinv = JT @ np.linalg.inv(JJT + lambda_reg * np.eye(JJT.shape[0]))
        else:
            J_pinv = JT @ np.linalg.inv(JJT)
        return J_pinv

    def get_task_space_state_with_orientation(self, q: np.ndarray, v: np.ndarray):
        """返回包含姿态误差的任务空间状态。

        Args:
            q: 关节位置。
            v: 关节速度。

        Returns:
            位置、线速度、姿态误差与角速度。
        """
        # 1. Get current state
        if self.obj_type == mujoco.mjtObj.mjOBJ_SITE:
            pos_cur = self.data.site_xpos[self.ee_id].copy()
            rot_cur = self.data.site_xmat[self.ee_id].reshape(3, 3).copy()
        else:
            pos_cur = self.data.xpos[self.ee_id].copy()
            rot_cur = self.data.xmat[self.ee_id].reshape(3, 3).copy()

        # 2. Orientation error: log(Rd Rc^T)
        orientation_error = self._log_map(rot_cur, self.initial_orientation)

        # 3. Velocities
        J = self._get_jacobian(update_prev=False)
        J_pos = J[:3, :]
        J_rot = J[3:, :]

        current_vel_pos = J_pos @ v  # Linear velocity
        current_vel_rot = J_rot @ v  # Angular velocity

        return pos_cur, current_vel_pos, orientation_error, current_vel_rot

    def get_task_space_state(self, q: np.ndarray, v: np.ndarray):
        """返回末端位置与线速度。

        Args:
            q: 关节位置。
            v: 关节速度。

        Returns:
            末端位置与线速度。
        """
        # 1. Get current position
        if self.obj_type == mujoco.mjtObj.mjOBJ_SITE:
            pos_cur = self.data.site_xpos[self.ee_id].copy()
        else:
            pos_cur = self.data.xpos[self.ee_id].copy()

        # 2. Compute linear velocity via Jacobian
        J = self._get_jacobian(update_prev=False)
        J_pos = J[:3, :]
        current_vel = J_pos @ v

        return pos_cur, current_vel

    def _apply_limits(self, tau: np.ndarray, q: np.ndarray, v: np.ndarray) -> np.ndarray:
        """应用软限位与速度缩放。

        Args:
            tau: 关节力矩。
            q: 关节位置。
            v: 关节速度。

        Returns:
            调整后的关节力矩。
        """
        k_limit = 100.0
        tau_limit = np.zeros_like(tau)

        # Check dimensions
        n = min(len(q), len(self.q_min))

        for i in range(n):
            if q[i] < self.q_min[i]:
                tau_limit[i] = k_limit * (self.q_min[i] - q[i])
            elif q[i] > self.q_max[i]:
                tau_limit[i] = k_limit * (self.q_max[i] - q[i])

        # Velocity scaling
        v_scale = np.ones_like(v)
        if hasattr(self, "v_max"):
            v_scale = np.minimum(1.0, self.v_max[: len(v)] / (np.abs(v) + 1e-6))

        tau = tau * v_scale

        return tau + tau_limit

    def manipulability_gradient(self, q, delta=1e-6):
        """计算操控度梯度。

        Args:
            q: 关节位置。
            delta: 有限差分步长。

        Returns:
            梯度向量。
        """
        grad = np.zeros_like(q)
        # 保存当前状态
        q_save = self.data.qpos.copy()

        # 计算当前操控度
        # 假设已调用 mj_forward 或状态有效
        J_current = self._get_jacobian(update_prev=False)[:3, :]
        manipulability_current = np.sqrt(np.linalg.det(J_current @ J_current.T))

        for i in range(len(q)):
            q_delta = q.copy()
            q_delta[i] += delta

            # 设置扰动后的状态
            self.data.qpos[:] = q_delta
            mujoco.mj_forward(self.model, self.data)

            J_delta = self._get_jacobian(update_prev=False)[:3, :]
            manipulability_delta = np.sqrt(np.linalg.det(J_delta @ J_delta.T))

            grad[i] = (manipulability_delta - manipulability_current) / delta

        # 恢复状态
        self.data.qpos[:] = q_save
        mujoco.mj_forward(self.model, self.data)

        return grad

    def compute_control_task_space_with_orientation_and_imp(
        self,
        q: np.ndarray,
        dq: np.ndarray,
        desired_ee_pos: np.ndarray,
        desired_ee_vel: np.ndarray,
        desired_ee_acc: np.ndarray,
        force_external: np.ndarray = np.zeros(3),
    ) -> np.ndarray:
        """计算含姿态与阻抗的任务空间控制。

        Args:
            q: 关节位置。
            dq: 关节速度。
            desired_ee_pos: 期望末端位置。
            desired_ee_vel: 期望末端线速度。
            desired_ee_acc: 期望末端线加速度。
            force_external: 外部力。

        Returns:
            关节力矩。
        """
        # 1. 返回末端位置、线/角速度，以及相对于期望姿态的李代数姿态误差
        pos_cur, vel_pos_cur, ori_err, vel_rot_cur = self.get_task_space_state_with_orientation(q, dq)

        # 2. 计算雅可比矩阵 (J) 和 雅可比导数 (J_dot)
        J = self._get_jacobian(update_prev=True)
        J_pos = J[:3, :]
        J_rot = J[3:, :]

        # 3. 获取动力学参数
        M = np.zeros((self.nv, self.nv))
        mujoco.mj_fullM(self.model, M, self.data.qM)  # 质量矩阵
        M_inv = np.linalg.inv(M + 1e-6 * np.eye(self.nv))

        # 科氏力 + 重力项（MuJoCo的qfrc_bias包含两者）
        h = self.data.qfrc_bias.copy()

        # 4. 阻抗控制参数（参考实现）
        # 平动阻抗
        m_pos = 10.0  # 虚拟质量
        d_pos = 100.0  # 虚拟阻尼
        k_pos = 1000.0  # 虚拟刚度

        # 旋转阻抗
        m_rot = 1.0  # 虚拟质量
        d_rot = 15.0  # 虚拟阻尼
        k_rot = 30.0  # 虚拟刚度

        force_desired = np.zeros(3)  # 期望外力

        # 5. 计算控制输入 u（阻抗控制公式）
        # 平动：Md*(xdd - xdd_des) + Dd*(xd - xd_des) + Kd*(x - x_des) = F_ext - F_des
        # 整理：u_pos = xdd_des + (F_ext - F_des - Dd*vel_err - Kd*pos_err) / Md
        pos_err = pos_cur - desired_ee_pos
        vel_err = vel_pos_cur - desired_ee_vel

        u_pos = desired_ee_acc + (force_external - force_desired - d_pos * vel_err - k_pos * pos_err) / m_pos

        # 旋转：类似PD控制（期望角速度为0）
        vel_rot_err = -vel_rot_cur
        u_rot = (k_rot * ori_err + d_rot * vel_rot_err) / m_rot

        # 组合6维任务空间输入
        u_task = np.concatenate([u_pos, u_rot])

        # 6. 动力学一致映射（操作空间控制）
        # Lambda = (J * M^-1 * J^T)^-1（操作空间惯性矩阵）
        Lambda_inv = J @ M_inv @ J.T
        Lambda = np.linalg.inv(Lambda_inv + 1e-6 * np.eye(6))

        # 动力学一致的雅可比伪逆：J_bar = M^-1 * J^T * Lambda
        J_bar = M_inv @ J.T @ Lambda

        # 7. 零空间控制（抑制未约束自由度的运动）
        # N = I - J_bar * J（零空间投影矩阵）
        N = np.eye(self.nv) - J_bar @ J

        # 零空间阻尼
        D_null = 10.0 * np.eye(self.nv)
        tau_null = -N @ D_null @ dq

        # 8. 合成关节力矩
        # tau = J^T * Lambda * (u - J_dot*dq + J*M_inv*h) + tau_null
        # 其中 J^T * Lambda 是动力学映射矩阵

        if not hasattr(self, "J_dot"):
            J_dot_dq = np.zeros(6)
        else:
            J_dot_dq = self.J_dot @ dq

        # 使用 J.T @ Lambda 作为映射矩阵
        lambda_map = J.T @ Lambda

        # 主任务力矩：前馈（acc） + 反馈（PD） + 补偿（J_dot, h）
        tau = lambda_map @ (u_task - J_dot_dq + J @ M_inv @ h) + tau_null

        # 9. 应用关节限位
        # tau = self._apply_limits(tau, q, dq)

        return tau

    def compute_control_task_space(
        self,
        q: np.ndarray,
        dq: np.ndarray,
        desired_ee_pos: np.ndarray,
        desired_ee_vel: np.ndarray,
        desired_ee_acc: np.ndarray,
    ) -> np.ndarray:
        """计算含姿态的任务空间控制。"""
        pos_cur, vel_pos_cur, ori_err, vel_rot_cur = self.get_task_space_state_with_orientation(q, dq)

        pos_err = desired_ee_pos - pos_cur
        vel_err = desired_ee_vel - vel_pos_cur

        k_p_pos = 100.0
        k_d_pos = 20.0
        k_p_rot = 50.0
        k_d_rot = 10.0

        u_pos = desired_ee_acc + k_d_pos * vel_err + k_p_pos * pos_err
        vel_rot_err = -vel_rot_cur
        u_rot = k_p_rot * ori_err + k_d_rot * vel_rot_err
        u_task = np.concatenate([u_pos, u_rot])

        J = self._get_jacobian(update_prev=True)

        M = np.zeros((self.nv, self.nv))
        mujoco.mj_fullM(self.model, M, self.data.qM)
        M_inv = np.linalg.inv(M + 1e-6 * np.eye(self.nv))

        h = self.data.qfrc_bias.copy()

        Lambda_inv = J @ M_inv @ J.T
        Lambda = np.linalg.inv(Lambda_inv + 1e-6 * np.eye(6))

        J_bar = M_inv @ J.T @ Lambda
        N = np.eye(self.nv) - J_bar @ J

        grad_m = self.manipulability_gradient(q)
        k_manip = 0.2
        ddq_desired = k_manip * grad_m
        tau_null_manip = M @ ddq_desired

        D_null = 1.2 * np.eye(self.nv)
        tau_null_damp = -D_null @ dq

        tau_null = N @ (tau_null_manip + tau_null_damp)

        if not hasattr(self, "J_dot"):
            J_dot_dq = np.zeros(6)
        else:
            J_dot_dq = self.J_dot @ dq

        tau_main = J.T @ Lambda @ (u_task - J_dot_dq + J @ M_inv @ h)
        tau = tau_main + tau_null

        tau = self._apply_limits(tau, q, dq)

        return tau
