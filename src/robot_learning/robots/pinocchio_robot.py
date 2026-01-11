"""Pinocchio 纯仿真环境。"""

from typing import Dict, Optional, Tuple

import numpy as np
import pinocchio as pin


class PinocchioRobot:
    """Pinocchio 仿真环境（数值积分）。

    Attributes:
        model: Pinocchio 机器人模型。
        data: Pinocchio 数据结构。
        dt: 仿真时间步长（秒）。
        sim_time: 当前仿真时间（秒）。
        q: 关节位置 (nq,)。
        dq: 关节速度 (nv,)。
        ee_frame_id: 末端执行器 frame id。
        integrator: 积分器名称。
    """

    def __init__(
        self,
        model: pin.Model,
        dt: float,
        ee_frame_name: str = "ee_frame",
        integrator: str = "semi-implicit-euler",
    ):
        """
        初始化 Pinocchio 仿真环境

        Args:
            model: Pinocchio 机器人模型
            dt: 仿真时间步长（秒）
            ee_frame_name: 末端执行器 frame 名称
            integrator: 积分器类型
                - "semi-implicit-euler": 半隐式欧拉（默认，稳定）
                - "rk4": 四阶 Runge-Kutta（更精确，但计算量大）
        """
        self.model = model
        self.data = model.createData()
        self.dt = dt
        self.sim_time = 0.0
        self.integrator = integrator

        # 初始状态
        self.q = pin.neutral(model)
        self.dq = np.zeros(model.nv)

        # EE frame
        self.ee_frame_id = model.getFrameId(ee_frame_name)
        if self.ee_frame_id >= len(model.frames):
            raise ValueError(f"Frame '{ee_frame_name}' not found in model.")

        # 关节限位
        self.q_min = getattr(model, "lowerPositionLimit", None)
        self.q_max = getattr(model, "upperPositionLimit", None)
        self.v_max = getattr(model, "velocityLimit", None)

        # 更新运动学
        pin.forwardKinematics(self.model, self.data, self.q, self.dq)
        pin.updateFramePlacements(self.model, self.data)

    # ==================== 仿真控制 ====================

    def reset(self, q: Optional[np.ndarray] = None, dq: Optional[np.ndarray] = None) -> None:
        """
        重置仿真状态

        Args:
            q: 初始关节位置（可选）
            dq: 初始关节速度（可选）
        """
        if q is None:
            self.q = pin.neutral(self.model)
        else:
            self.q = q.copy()

        if dq is None:
            self.dq = np.zeros(self.model.nv)
        else:
            self.dq = dq.copy()

        self.sim_time = 0.0

        # 更新运动学
        pin.forwardKinematics(self.model, self.data, self.q, self.dq)
        pin.updateFramePlacements(self.model, self.data)

    def step(self, tau: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        仿真步进（单步）

        Args:
            tau: 控制输入扭矩 (nv,)

        Returns:
            q: 关节位置 (nq,)
            dq: 关节速度 (nv,)
            ee_pos: 末端执行器位置 (3,)
        """
        if self.integrator == "semi-implicit-euler":
            self._step_semi_implicit_euler(tau)
        elif self.integrator == "rk4":
            self._step_rk4(tau)
        else:
            raise ValueError(f"Unknown integrator: {self.integrator}")

        # 更新运动学
        pin.forwardKinematics(self.model, self.data, self.q, self.dq)
        pin.updateFramePlacements(self.model, self.data)

        # 更新时间
        self.sim_time += self.dt

        # 返回 EE 位置
        ee_pos = self.data.oMf[self.ee_frame_id].translation.copy()

        return self.q.copy(), self.dq.copy(), ee_pos

    def _step_semi_implicit_euler(self, tau: np.ndarray) -> None:
        """
        半隐式欧拉积分

        优点：数值稳定，计算高效
        缺点：一阶精度

        算法：
            1. 计算加速度：ddq = M^{-1} (tau - h)
            2. 更新速度：dq <- dq + ddq * dt
            3. 更新位置：q <- q + dq * dt
        """
        # 前向动力学
        ddq = self._compute_forward_dynamics(self.q, self.dq, tau)

        # 半隐式欧拉
        self.dq += ddq * self.dt
        self.q = pin.integrate(self.model, self.q, self.dq * self.dt)

        # 关节限位（位置）
        if self.q_min is not None and self.q_max is not None:
            self.q = np.clip(self.q, self.q_min, self.q_max)

        # 速度限位
        if self.v_max is not None:
            self.dq = np.clip(self.dq, -self.v_max, self.v_max)

    def _step_rk4(self, tau: np.ndarray) -> None:
        """
        四阶 Runge-Kutta 积分

        优点：高精度（四阶）
        缺点：计算量大（每步需要 4 次前向动力学）

        算法：
            k1 = f(t, y)
            k2 = f(t + dt/2, y + dt*k1/2)
            k3 = f(t + dt/2, y + dt*k2/2)
            k4 = f(t + dt, y + dt*k3)
            y <- y + dt*(k1 + 2*k2 + 2*k3 + k4)/6

        其中 y = (q, dq)，f(y) = (dq, ddq)
        """
        q0 = self.q.copy()
        dq0 = self.dq.copy()
        dt = self.dt

        def dynamics(q, dq):
            """动力学函数：返回 (dq, ddq)"""
            ddq = self._compute_forward_dynamics(q, dq, tau)
            return dq, ddq

        # k1
        k1_q, k1_dq = dynamics(q0, dq0)

        # k2
        k2_q, k2_dq = dynamics(
            pin.integrate(self.model, q0, k1_q * dt / 2),
            dq0 + k1_dq * dt / 2,
        )

        # k3
        k3_q, k3_dq = dynamics(
            pin.integrate(self.model, q0, k2_q * dt / 2),
            dq0 + k2_dq * dt / 2,
        )

        # k4
        k4_q, k4_dq = dynamics(
            pin.integrate(self.model, q0, k3_q * dt),
            dq0 + k3_dq * dt,
        )

        # 更新状态
        dq_new = dq0 + (dt / 6) * (k1_dq + 2 * k2_dq + 2 * k3_dq + k4_dq)
        q_new = pin.integrate(self.model, q0, (dt / 6) * (k1_q + 2 * k2_q + 2 * k3_q + k4_q))

        # 关节限位
        if self.q_min is not None and self.q_max is not None:
            q_new = np.clip(q_new, self.q_min, self.q_max)
        if self.v_max is not None:
            dq_new = np.clip(dq_new, -self.v_max, self.v_max)

        self.q = q_new
        self.dq = dq_new

    # ==================== 动力学计算 ====================

    def _compute_forward_dynamics(self, q: np.ndarray, dq: np.ndarray, tau: np.ndarray) -> np.ndarray:
        """
        计算前向动力学：M(q)ddq + h(q,dq) = tau

        Args:
            q: 关节位置 (nq,)
            dq: 关节速度 (nv,)
            tau: 控制扭矩 (nv,)

        Returns:
            ddq: 关节加速度 (nv,)
        """
        # 计算质量矩阵
        M = pin.crba(self.model, self.data, q)
        M = 0.5 * (M + M.T)  # 确保对称

        # 计算非线性项（科氏力 + 重力）
        h = pin.nonLinearEffects(self.model, self.data, q, dq)

        # 求解 ddq
        ddq = np.linalg.solve(M, tau - h)

        return ddq

    def compute_mass_matrix(self) -> np.ndarray:
        """
        计算质量矩阵 M(q)

        Returns:
            M: 质量矩阵 (nv, nv)
        """
        M = pin.crba(self.model, self.data, self.q)
        return 0.5 * (M + M.T)

    def compute_coriolis_gravity(self) -> np.ndarray:
        """
        计算科氏力和重力项 h(q, dq) = C(q, dq) * dq + g(q)

        Returns:
            h: 非线性项向量 (nv,)
        """
        return pin.nonLinearEffects(self.model, self.data, self.q, self.dq)

    def compute_inverse_dynamics(self, ddq: np.ndarray) -> np.ndarray:
        """
        计算逆动力学：给定加速度计算所需力矩

        Args:
            ddq: 关节加速度 (nv,)

        Returns:
            tau: 所需的广义力 (nv,)
        """
        return pin.aba(self.model, self.data, self.q, self.dq, ddq)

    # ==================== 运动学 ====================

    def compute_forward_kinematics(self) -> None:
        """更新正运动学"""
        pin.forwardKinematics(self.model, self.data, self.q, self.dq)
        pin.updateFramePlacements(self.model, self.data)

    def get_ee_state(self) -> Tuple[np.ndarray, np.ndarray]:
        """
        获取末端执行器状态（位置和线速度）

        Returns:
            pos: 末端位置 (3,)
            vel: 末端线速度 (3,)
        """
        # 位置
        H = self.data.oMf[self.ee_frame_id]
        pos = H.translation.copy()

        # 速度（通过雅可比）
        J = pin.getFrameJacobian(
            self.model,
            self.data,
            self.ee_frame_id,
            pin.ReferenceFrame.LOCAL_WORLD_ALIGNED,
        )
        J_pos = J[:3, :]
        vel = J_pos @ self.dq

        return pos, vel

    def get_ee_pose(self) -> Tuple[np.ndarray, np.ndarray]:
        """
        获取末端执行器位姿（位置和旋转矩阵）

        Returns:
            pos: 末端位置 (3,)
            rot: 末端旋转矩阵 (3, 3)
        """
        H = self.data.oMf[self.ee_frame_id]
        pos = H.translation.copy()
        rot = H.rotation.copy()
        return pos, rot

    def compute_jacobian(self) -> Tuple[np.ndarray, np.ndarray]:
        """
        计算末端雅可比矩阵

        Returns:
            jacp: 线速度雅可比 (3, nv)
            jacr: 角速度雅可比 (3, nv)
        """
        J = pin.getFrameJacobian(
            self.model,
            self.data,
            self.ee_frame_id,
            pin.ReferenceFrame.LOCAL_WORLD_ALIGNED,
        )
        return J[:3, :], J[3:, :]

    # ==================== IK 求解 ====================

    def solve_ik(
        self,
        target_pose: pin.SE3,
        initial_q: Optional[np.ndarray] = None,
        max_iters: int = 3000,
        eps: float = 1e-7,
        damp: float = 1e-8,
    ) -> Tuple[np.ndarray, bool]:
        """
        阻尼最小二乘 IK 求解

        Args:
            target_pose: 目标末端位姿（Pinocchio SE3）
            initial_q: 初始关节位置（可选）
            max_iters: 最大迭代次数
            eps: 收敛阈值（位置误差）
            damp: 阻尼系数（用于奇异点处理）

        Returns:
            q: 求解得到的关节位置
            success: 是否收敛
        """
        if initial_q is None:
            q = pin.neutral(self.model)
        else:
            q = initial_q.copy()

        for _ in range(max_iters):
            pin.forwardKinematics(self.model, self.data, q)
            pin.updateFramePlacements(self.model, self.data)

            current_pose = self.data.oMf[self.ee_frame_id]

            # 位置误差
            error_pos = target_pose.translation - current_pose.translation

            # 姿态误差（轴角表示）
            error_rot = pin.log3(target_pose.rotation @ current_pose.rotation.T)

            error = np.concatenate([error_pos, error_rot])

            # 检查收敛
            if np.linalg.norm(error) < eps:
                return q, True

            # 计算雅可比
            pin.computeJointJacobians(self.model, self.data, q)
            J = pin.getFrameJacobian(
                self.model,
                self.data,
                self.ee_frame_id,
                pin.ReferenceFrame.LOCAL_WORLD_ALIGNED,
            )

            # 阻尼最小二乘
            Jt = J.T
            JJt = J @ Jt
            lambda_eye = damp * np.eye(6)

            v = np.linalg.solve(JJt + lambda_eye, error)
            dq = Jt @ v
            q = pin.integrate(self.model, q, dq)

            # 关节限位
            if self.q_min is not None and self.q_max is not None:
                q = np.clip(q, self.q_min, self.q_max)

        return q, False

    # ==================== 工具方法 ====================

    def get_joint_states(self) -> Tuple[np.ndarray, np.ndarray]:
        """
        获取当前关节状态

        Returns:
            q: 关节位置 (nq,)
            dq: 关节速度 (nv,)
        """
        return self.q.copy(), self.dq.copy()

    def check_joint_limits(self) -> Dict[str, bool]:
        """
        检查关节是否在限位内

        Returns:
            limit_status: 关节名称 -> 是否在限位内
        """
        limit_status = {}
        if self.q_min is None or self.q_max is None:
            return limit_status

        for i in range(self.model.njnt):
            joint_name = self.model.names[i]
            if joint_name:
                in_limit = self.q_min[i] <= self.q[i] <= self.q_max[i]
                limit_status[joint_name] = in_limit

        return limit_status

    def get_manipulability(self) -> float:
        """
        计算操作度（Yoshikawa 可操作度）

        w = sqrt(det(J * J^T))

        Returns:
            w: 操作度值（越大表示离奇异点越远）
        """
        J_pos, _ = self.compute_jacobian()
        JJt = J_pos @ J_pos.T
        det = np.linalg.det(JJt)
        return np.sqrt(max(0, det))

    # ==================== 上下文管理器 ====================

    def __enter__(self):
        """上下文管理器入口"""
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """上下文管理器退出"""
        pass


# ==================== 测试代码 ====================


def test_pinocchio_robot():
    """测试 PinocchioRobot 的基本功能"""
    urdf_path = "./assets/kuka_xml_urdf/urdf/iiwa14_dock.urdf"

    print("=== Testing PinocchioRobot ===\n")

    # 创建模型
    model = pin.buildModelFromUrdf(urdf_path)

    # 创建仿真环境
    robot = PinocchioRobot(
        model=model,
        dt=0.001,
        ee_frame_name="cylinder_link",
        integrator="semi-implicit-euler",
    )

    print(f"DOF: nq={model.nq}, nv={model.nv}")
    print(f"Integrator: {robot.integrator}")
    print(f"Timestep: {robot.dt}s\n")

    # 测试初始状态
    q, dq, ee_pos = robot.step(np.zeros(model.nv))
    print(f"Initial EE pos: {ee_pos}")

    # 简单的重力补偿控制
    g = pin.computeGeneralizedGravity(model, robot.data, q)

    print("\n=== Running Simulation ===")
    for step in range(1000):
        # 重力补偿 + PD 控制
        k_p = 10.0
        k_d = 5.0
        q_des = q
        dq_des = np.zeros_like(dq)

        tau = g + k_p * (q_des - q) + k_d * (dq_des - dq)

        q, dq, ee_pos = robot.step(tau)

        if step % 200 == 0:
            print(f"Step {step}: EE pos = {ee_pos}")

    # 测试 IK
    print("\n=== Testing IK ===")
    ee_pos_current, _ = robot.get_ee_state()
    target_pos = ee_pos_current + np.array([0.0, 0.1, 0.0])
    ee_rot = robot.get_ee_pose()[1]
    target_pose = pin.SE3(ee_rot, target_pos)

    q_sol, success = robot.solve_ik(target_pose, initial_q=q, max_iters=100)
    print(f"IK success: {success}")
    print(f"Target pos: {target_pos}")
    print(f"Solved pos: {robot.get_ee_state()[0]}")

    # 测试操作度
    print("\n=== Testing Manipulability ===")
    manip = robot.get_manipulability()
    print(f"Manipulability: {manip:.6f}")

    print("\n=== Test Complete ===")


def test_integrators():
    """比较不同积分器的精度"""
    urdf_path = "./assets/kuka_xml_urdf/urdf/iiwa14_dock.urdf"

    print("=== Comparing Integrators ===\n")

    integrators = ["semi-implicit-euler", "rk4"]
    results = {}

    for integrator in integrators:
        model = pin.buildModelFromUrdf(urdf_path)
        robot = PinocchioRobot(model=model, dt=0.001, ee_frame_name="cylinder_link", integrator=integrator)

        # 简单的摆动控制
        q_final = np.zeros(model.nv)
        q_final[0] = 0.5  # 移动第一个关节

        # 记录轨迹
        trajectory = []

        for _ in range(1000):
            # PD 控制到目标
            k_p = 100.0
            k_d = 20.0
            g = pin.computeGeneralizedGravity(model, robot.data, robot.q)

            tau = g + k_p * (q_final - robot.q) + k_d * (0 - robot.dq)
            q, dq, ee_pos = robot.step(tau)

            trajectory.append(robot.q[0])

        results[integrator] = trajectory
        print(f"{integrator}: final q[0] = {q[0]:.6f}")

    print("\n=== Test Complete ===")


if __name__ == "__main__":
    test_pinocchio_robot()
    print("\n" + "=" * 40 + "\n")
    test_integrators()
