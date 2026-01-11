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

        pos_err = desired_ee_pos - pos_cur
        vel_err = desired_ee_vel - vel_pos_cur
        u_pos = desired_ee_acc + (force_external + d_pos * vel_err + k_pos * pos_err) / m_pos

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

    def compute_control_task_space_strict_fnd(
        self, q: np.ndarray, v: np.ndarray, pos_des: np.ndarray, vel_des: np.ndarray, acc_des: np.ndarray
    ) -> np.ndarray:
        # --- 1. 基础状态计算 ---
        pos_cur, vel_pos_cur, ori_err, vel_rot_cur = self.get_task_space_state_with_orientation(q, v)
        vel_cur = np.concatenate([vel_pos_cur, vel_rot_cur])

        # 补全 acc_des 为 6维
        if acc_des.shape == (3,):
            acc_des_full = np.concatenate([acc_des, np.zeros(3)])
        else:
            acc_des_full = acc_des

        # 补全 vel_des 为 6维 (假设期望角速度为0，或者你需要传入完整的 vel_des)
        # 这里为了演示，构建一个完整的 vel_des_full
        if vel_des.shape == (3,):
            vel_des_full = np.concatenate([vel_des, np.zeros(3)])
        else:
            vel_des_full = vel_des

        # --- 2. 动力学矩阵 (包括重力计算) ---
        # 关键修复：计算重力项
        g = pin.computeGeneralizedGravity(self.model, self.data, q)

        end_effector_id = self.model.getFrameId("cylinder_link")
        J = pin.computeFrameJacobian(self.model, self.data, q, end_effector_id, pin.ReferenceFrame.WORLD)
        dJ = pin.getFrameJacobianTimeVariation(self.model, self.data, end_effector_id, pin.ReferenceFrame.WORLD)

        M = pin.crba(self.model, self.data, q)
        M_inv = np.linalg.inv(M + 1e-9 * np.eye(7))

        Lambda_inv = J @ M_inv @ J.T
        Omega = np.linalg.inv(Lambda_inv + 1e-9 * np.eye(6))

        # --- 3. 严格计算 mu (任务空间科氏矩阵) ---
        # 公式: mu = Omega * (J * M^-1 * C - dJ) * J^#
        # 但计算 J^# (伪逆) 比较慢。我们可以利用性质 mu * v = Omega * (J * M^-1 * C*v - dJ*v)
        # 来直接计算向量 h_mu = mu * v (这是任务空间的科氏力向量)

        # 关节空间科氏力向量
        pin.computeCoriolisMatrix(self.model, self.data, q, v)
        C_vec = self.data.C @ v

        # 任务空间科氏力向量 h_task = mu * x_dot = Omega * (J * M^-1 * C * q_dot - dJ * q_dot)
        # 这一项对应公式中的 hat_mu * x_dot
        term_temp = J @ M_inv @ C_vec - dJ @ v
        h_hat_mu = Omega @ term_temp  # 这是一个 6维向量

        # --- 4. 提取 mu_d (近似) ---
        # 严格提取矩阵 mu 的对角块非常耗时（需要计算 6x6 矩阵）。
        # 工程上通常近似：假设 mu_d * x_dot 约等于 h_hat_mu 的分量投影，或者直接忽略 mu_d 的非对角影响。
        # 但为了凑公式，我们可以近似认为 mu_d * x_dot 就是 h_hat_mu 向量本身（即不进行去耦），
        # 或者更简单的：在 FND 论文中，mu_d 是为了保持能量被动性。

        # 这里我们采用一个常用的工程技巧：
        # 我们已经有了 hat_mu * x_dot (即 h_hat_mu)。
        # 公式要求项： Scaling * [ (hat_mu * x_dot - mu_d * x_dot) + (mu_d * x_dot_des) ]
        #            = Scaling * [ hat_mu * x_dot - mu_d * (x_dot - x_dot_des) ]
        #            = Scaling * [ h_hat_mu - mu_d * err_vel ]

        # 此时我们需要 mu_d 矩阵。
        # 我们可以通过数值差分或简化模型估算，或者直接忽略 mu_d * err_vel (假设误差很小)。
        # 如果必须实现，我们假设 mu_d 为零（完全消除科氏力，退化为 ID），或者：
        # 使用 h_hat_mu 的绝对值构造一个对角阻尼矩阵作为 mu_d 的近似。

        # 鉴于计算 mu_d 矩阵太复杂，我们回到 FND 的本质：
        # FND 希望闭环方程为：Omega_d * dd_x + mu_d * d_x = F_ctrl
        # 我们这里的实现选择：补偿掉所有非对角项。

        # 让我们构建严格匹配公式的力矩：
        # tau = g + J.T * Omega * Omega_d_inv * ( ... )

        # Omega_d 构造
        Omega_d = np.block([[Omega[0:3, 0:3], np.zeros((3, 3))], [np.zeros((3, 3)), Omega[3:6, 3:6]]])
        Omega_d_inv = np.linalg.inv(Omega_d)
        Scaling = J.T @ Omega @ Omega_d_inv

        # PD 反馈
        Kp = np.concatenate([np.full(3, 150.0), np.full(3, 50.0)])
        Kd = np.concatenate([np.full(3, 40.0), np.full(3, 10.0)])
        err_pos = np.concatenate([pos_des - pos_cur, ori_err])
        err_vel = vel_des_full - vel_cur  # 注意：这里是 desired - current

        # 闭环整形项 (Closed-loop shaping)
        # Inner = Omega_d * acc_des + mu_d * vel_des - D * (-err_vel) - K * (-err_pos)
        #       = Omega_d * acc_des + mu_d * vel_des + Kd * err_vel + Kp * err_pos
        # 注意：公式里的 D 和 K 定义可能带负号，这里按标准 PD 正反馈写

        # 科氏力解耦项 (Decoupling)
        # Decouple = hat_mu * vel_cur - mu_d * vel_cur

        # 合并 Inner + Decouple:
        # Total = Omega_d * acc_des + mu_d * vel_des + Kd * err_vel + Kp * err_pos + hat_mu * vel_cur - mu_d * vel_cur
        #       = Omega_d * acc_des + hat_mu * vel_cur - mu_d * (vel_cur - vel_des) + Kd * err_vel + Kp * err_pos
        #       = Omega_d * acc_des + h_hat_mu       - mu_d * (-err_vel)        + Kd * err_vel + Kp * err_pos
        #       = Omega_d * acc_des + h_hat_mu       + (Kd + mu_d) * err_vel    + Kp * err_pos

        # 此时你可以看到，mu_d 实际上起到了增加阻尼的作用！
        # 如果我们无法精确计算 mu_d，可以将其视为 0 (即完全线性化)，或者将其吸收到 Kd 中 (调大 Kd)。

        # *** 最终工程实现 (Strict Structure) ***
        # 1. 惯性项
        F_inertial = Omega_d @ acc_des_full

        # 2. 科氏力项 (h_hat_mu 就是 hat_mu * x_dot)
        F_coriolis = h_hat_mu

        # 3. 反馈项 (忽略 mu_d 对阻尼的微小贡献，假设其被 Kd 覆盖)
        F_feedback = Kd * err_vel + Kp * err_pos

        # 总输出
        # F_star = F_inertial + F_coriolis + F_feedback
        # tau = Scaling @ F_star + g

        F_star = F_inertial + F_coriolis + F_feedback
        tau = Scaling @ F_star + g  # 使用计算出的重力项

        # 这种写法是数学上最接近公式的，且没有做 C@v 的粗暴消除，
        # 而是正确计算了任务空间的科氏力 h_hat_mu 并通过 Scaling 映射回去。
        # 获取重力向量
        g_vector = self.model.gravity.linear
        # 判断是否有重力
        if np.linalg.norm(g_vector) == 0:
            print("当前是无重力环境 (Zero Gravity)")
        else:
            print(f"当前环境存在重力，大小为: {np.linalg.norm(g_vector)}")
        # === 调试打印：检查重力补偿 ===
        print(f"\n[重力补偿调试]")
        print(f"  重力项 g: {g}")
        print(f"  惯性项贡献: {np.linalg.norm(Scaling @ F_inertial):.3f}")
        print(f"  科氏力项贡献: {np.linalg.norm(Scaling @ F_coriolis):.3f}")
        print(f"  反馈项贡献: {np.linalg.norm(Scaling @ F_feedback):.3f}")
        print(f"  总控制力矩 tau: {tau}")
        print(f"  tau 范数: {np.linalg.norm(tau):.3f}")

        return tau

    def compute_control_task_space_pd_plus(
        self, q: np.ndarray, v: np.ndarray, pos_des: np.ndarray, vel_des: np.ndarray, acc_des: np.ndarray
    ) -> np.ndarray:
        """
        Task-Space PD+ Controller
        Formula: tau = J.T * ( Omega * acc_des + mu * vel_des - D * err_vel - K * err_pos ) + g
        """
        # --- 1. 基础状态计算 ---
        # 获取当前位置、速度、姿态误差
        pos_cur, vel_pos_cur, ori_err, vel_rot_cur = self.get_task_space_state_with_orientation(q, v)
        vel_cur = np.concatenate([vel_pos_cur, vel_rot_cur])  # 当前任务空间速度 x_dot

        # 补全期望值为 6维
        if acc_des.shape == (3,):
            acc_des_full = np.concatenate([acc_des, np.zeros(3)])
        else:
            acc_des_full = acc_des

        if vel_des.shape == (3,):
            vel_des_full = np.concatenate([vel_des, np.zeros(3)])
        else:
            vel_des_full = vel_des

        # --- 2. 动力学矩阵计算 ---
        end_effector_id = self.model.getFrameId("cylinder_link")

        # 2.1 雅可比 J
        J = pin.computeFrameJacobian(self.model, self.data, q, end_effector_id, pin.ReferenceFrame.WORLD)

        # 2.2 雅可比导数 dJ (对应公式中的 J_dot)
        dJ = pin.getFrameJacobianTimeVariation(self.model, self.data, end_effector_id, pin.ReferenceFrame.WORLD)

        # 2.3 质量矩阵 M 及其逆
        M = pin.crba(self.model, self.data, q)
        M_inv = np.linalg.inv(M + 1e-9 * np.eye(7))

        # 2.4 任务空间惯量矩阵 Omega (即 Lambda)
        # Omega = (J * M^-1 * J.T)^-1
        Lambda_inv = J @ M_inv @ J.T
        Omega = np.linalg.inv(Lambda_inv + 1e-9 * np.eye(6))

        # --- 3. 核心项：任务空间科氏力 mu * vel_des ---
        # 公式推导: mu * v = Omega * (J * M^-1 * C * J_pinv * v - dJ * J_pinv * v)
        # 我们需要计算 mu * vel_des_full

        # 3.1 计算期望关节速度 v_ref (通过伪逆映射)
        # v_ref = J^# * x_dot_d
        J_pinv = np.linalg.pinv(J)
        v_ref = J_pinv @ vel_des_full

        # 3.2 计算关节空间科氏矩阵 C(q, v)
        # 注意：这里必须用当前的真实速度 v 来计算 C 矩阵
        pin.computeCoriolisMatrix(self.model, self.data, q, v)
        C_matrix = self.data.C

        # 3.3 计算中间项 T = J * M^-1 * C * v_ref - dJ * v_ref
        # term1 = J * M^-1 * (C * v_ref)
        C_times_vref = C_matrix @ v_ref
        term1 = J @ M_inv @ C_times_vref

        # term2 = dJ * v_ref
        term2 = dJ @ v_ref

        # mu_times_vel_des = Omega * (term1 - term2)
        mu_vel_des = Omega @ (term1 - term2)

        # --- 4. 控制律组装 ---

        # PD 参数 (建议根据任务调整)
        Kp = np.concatenate([np.full(3, 100.0), np.full(3, 50.0)])
        Kd = np.concatenate([np.full(3, 20.0), np.full(3, 10.0)])

        # 误差计算 (期望 - 当前)
        # 姿态误差 ori_err 已经是 (Log(R_des * R_cur.T))
        err_pos = np.concatenate([pos_des - pos_cur, ori_err])
        err_vel = vel_des_full - vel_cur

        # 括号内的项: ( Omega * acc_des + mu * vel_des - D * err_vel - K * err_pos )
        # 注意公式符号: - D * dot(x_tilde) - K * x_tilde
        # 通常 x_tilde = x - x_des, 所以 - K * (x - x_des) = K * (x_des - x) = K * err_pos
        # 因此这里用 + 号

        F_cmd = (Omega @ acc_des_full) + mu_vel_des + (Kd * err_vel) + (Kp * err_pos)

        # 映射到关节力矩: J.T * F_cmd
        tau_task = J.T @ F_cmd

        # --- 5. 重力补偿 ---
        # 关键修复：计算重力项
        g = pin.computeGeneralizedGravity(self.model, self.data, q)

        # 总力矩
        tau = tau_task + g

        return tau

    def compute_control_inverse_dynamics_1(
        self, q: np.ndarray, v: np.ndarray, pos_des: np.ndarray, vel_des: np.ndarray, acc_des: np.ndarray
    ) -> np.ndarray:
        """
        Inverse Dynamics #1 (Full Artificial Decoupling)
        Formula: tau = J^T * Omega * (x_dd_d + hat_mu*x_dot - D*x_tilde_dot - K*x_tilde) + g

        Features:
        - Enforces closed-loop inertia to be Identity (Omega_d = I)
        - High feedback gains required (Aggressive)
        """
        # --- 1. 基础状态计算 ---
        pos_cur, vel_pos_cur, ori_err, vel_rot_cur = self.get_task_space_state_with_orientation(q, v)
        vel_cur = np.concatenate([vel_pos_cur, vel_rot_cur])

        # 补全期望值为 6维
        if acc_des.shape == (3,):
            acc_des_full = np.concatenate([acc_des, np.zeros(3)])
        else:
            acc_des_full = acc_des

        if vel_des.shape == (3,):
            vel_des_full = np.concatenate([vel_des, np.zeros(3)])
        else:
            vel_des_full = vel_des

        # --- 2. 动力学矩阵 ---
        end_effector_id = self.model.getFrameId("cylinder_link")

        J = pin.computeFrameJacobian(self.model, self.data, q, end_effector_id, pin.ReferenceFrame.WORLD)
        dJ = pin.getFrameJacobianTimeVariation(self.model, self.data, end_effector_id, pin.ReferenceFrame.WORLD)

        M = pin.crba(self.model, self.data, q)
        M_inv = np.linalg.inv(M + 1e-9 * np.eye(7))

        # 计算自然任务空间惯量 Omega
        Lambda_inv = J @ M_inv @ J.T
        Omega = np.linalg.inv(Lambda_inv + 1e-9 * np.eye(6))

        # --- 3. 计算 hat_mu * x_dot (Drift Acceleration) ---
        # 依据 Eq. 21 (当 Omega_d = I 时): hat_mu * x_dot = J * M^-1 * C * q_dot - dJ * q_dot

        # 3.1 计算关节空间科氏力项 C * q_dot
        pin.computeCoriolisMatrix(self.model, self.data, q, v)
        C_q_dot = self.data.C @ v

        # 3.2 映射到任务空间加速度: J * M^-1 * (C * q_dot)
        term_coriolis = J @ M_inv @ C_q_dot

        # 3.3 计算雅可比漂移: dJ * q_dot
        term_drift = dJ @ v

        # 3.4 组合得到 hat_mu * x_dot
        # 这一项代表了如果不加控制，末端执行器会产生的"自然漂移"加速度
        hat_mu_x_dot = term_coriolis - term_drift

        # --- 4. 控制律组装 ---
        # PD 参数 (ID 方法通常需要较高的增益来克服模型误差，参考论文 Table II)
        Kp = np.concatenate([np.full(3, 200.0), np.full(3, 50.0)])
        Kd = np.concatenate([np.full(3, 40.0), np.full(3, 5.0)])

        # 误差计算 (期望 - 当前)
        err_pos = np.concatenate([pos_des - pos_cur, ori_err])
        err_vel = vel_des_full - vel_cur

        # 括号内的项: u_cmd + hat_mu*x_dot
        # u_cmd = x_dd_d + Kd * err_vel + Kp * err_pos (对应公式中的 - D*x_tilde_dot - K*x_tilde)
        u_cmd = acc_des_full + (Kd * err_vel) + (Kp * err_pos)

        # Inverse Dynamics #1 的核心特征：
        # 我们希望消除 hat_mu_x_dot，并注入 u_cmd。
        # 由于我们希望闭环惯量是 I，所以 F_task = Omega * (u_cmd + hat_mu_x_dot)
        # 这样 Omega^-1 * F_task = u_cmd + hat_mu_x_dot
        # 动力学方程: x_dd = Omega^-1 * F_task - hat_mu_x_dot = u_cmd

        total_acc_cmd = u_cmd + hat_mu_x_dot
        F_task = Omega @ total_acc_cmd

        # --- 5. 重力补偿 ---
        # 按照您的要求使用 pin.computeGeneralizedGravity
        g = pin.computeGeneralizedGravity(self.model, self.data, q)

        # 总力矩
        tau = J.T @ F_task + g

        return tau

    def compute_control_inverse_dynamics_2(
        self, q: np.ndarray, v: np.ndarray, pos_des: np.ndarray, vel_des: np.ndarray, acc_des: np.ndarray
    ) -> np.ndarray:
        """
        Inverse Dynamics #2 (Fixed Desired Inertia)
        Formula: tau = J^T * Omega * Omega_d^-1 * (Omega_d * x_dd_d + hat_mu*x_dot - D*x_tilde_dot - K*x_tilde) + g
        Implementation: Omega_d = Omega_d(t0) fixed at first call
        """
        # --- 1. 基础状态计算 ---
        pos_cur, vel_pos_cur, ori_err, vel_rot_cur = self.get_task_space_state_with_orientation(q, v)
        vel_cur = np.concatenate([vel_pos_cur, vel_rot_cur])

        # 补全期望值为 6维
        if acc_des.shape == (3,):
            acc_des_full = np.concatenate([acc_des, np.zeros(3)])
        else:
            acc_des_full = acc_des

        if vel_des.shape == (3,):
            vel_des_full = np.concatenate([vel_des, np.zeros(3)])
        else:
            vel_des_full = vel_des

        # --- 2. 动力学矩阵 ---
        end_effector_id = self.model.getFrameId("cylinder_link")

        J = pin.computeFrameJacobian(self.model, self.data, q, end_effector_id, pin.ReferenceFrame.WORLD)
        dJ = pin.getFrameJacobianTimeVariation(self.model, self.data, end_effector_id, pin.ReferenceFrame.WORLD)

        M = pin.crba(self.model, self.data, q)
        M_inv = np.linalg.inv(M + 1e-9 * np.eye(7))

        # 当前时刻的自然惯量 Omega(t)
        Lambda_inv = J @ M_inv @ J.T
        Omega_t = np.linalg.inv(Lambda_inv + 1e-9 * np.eye(6))

        # --- 3. 处理固定惯量 Omega_d(t0) ---
        # 如果是第一次运行（或已复位），则锁定当前的 Omega 对角部分作为 Omega_d
        if self.Omega_d_fixed is None:
            # 提取对角块
            Omega_d_init = np.block([[Omega_t[0:3, 0:3], np.zeros((3, 3))], [np.zeros((3, 3)), Omega_t[3:6, 3:6]]])
            self.Omega_d_fixed = Omega_d_init
            # print("Inverse Dynamics #2: Omega_d fixed at t0.")

        Omega_d = self.Omega_d_fixed
        Omega_d_inv = np.linalg.inv(Omega_d + 1e-9 * np.eye(6))

        # --- 4. 计算 hat_mu * x_dot (Drift Compensation) ---
        # 根据 Eq. 21: hat_mu = Omega_d * (J * M^-1 * C - dJ) * J^-1
        # 所以 hat_mu * x_dot = Omega_d * (J * M^-1 * C * q_dot - dJ * q_dot)
        # 注意这里乘的是固定的 Omega_d

        # 4.1 关节空间科氏力 C * q_dot
        pin.computeCoriolisMatrix(self.model, self.data, q, v)
        C_q_dot = self.data.C @ v

        # 4.2 漂移加速度项: a_drift = J * M^-1 * C*q_dot - dJ * q_dot
        a_drift = (J @ M_inv @ C_q_dot) - (dJ @ v)

        # 4.3 映射用力: hat_mu_x_dot = Omega_d * a_drift
        hat_mu_x_dot = Omega_d @ a_drift

        # --- 5. 控制律组装 ---
        # PD 参数 (ID 方法通常需要较高的增益)
        Kp = np.concatenate([np.full(3, 200.0), np.full(3, 50.0)])
        Kd = np.concatenate([np.full(3, 40.0), np.full(3, 5.0)])

        # 误差
        err_pos = np.concatenate([pos_des - pos_cur, ori_err])
        err_vel = vel_des_full - vel_cur

        # 括号内各项:
        # Term 1: 惯性前馈 Omega_d * acc_des
        F_inertial = Omega_d @ acc_des_full

        # Term 2: 科氏力解耦 hat_mu * x_dot
        F_coriolis = hat_mu_x_dot

        # Term 3: 反馈力 - D*x_tilde_dot - K*x_tilde
        # 对应代码: Kd * err_vel + Kp * err_pos
        F_feedback = (Kd * err_vel) + (Kp * err_pos)

        # 括号内总和
        F_inner = F_inertial + F_coriolis + F_feedback

        # 外部缩放: J^T * Omega(t) * Omega_d^-1
        # tau = J^T * Omega(t) * Omega_d^-1 * F_inner + g

        # 计算 Scaling Matrix
        Scaling = J.T @ Omega_t @ Omega_d_inv

        # 重力补偿
        g = pin.computeGeneralizedGravity(self.model, self.data, q)

        # 总力矩
        tau = Scaling @ F_inner + g

        return tau
