"""MuJoCo 与 Pinocchio 的混合仿真环境。"""

from typing import Dict, List, Optional, Tuple

import mujoco
import numpy as np
import pinocchio as pin

from .mujoco_robot import MujocoRobot


class HybridRobot(MujocoRobot):
    """混合仿真环境：MuJoCo 仿真 + Pinocchio 动力学。

    Attributes:
        pin_model: Pinocchio 机器人模型。
        pin_data: Pinocchio 数据结构。
        pin_joint_map: 关节索引映射 (mj_idx, pin_q_idx, pin_v_idx)。
        pin_ee_frame_name: Pinocchio 末端执行器 frame 名称。
        pin_ee_frame_id: Pinocchio 末端执行器 frame id。
    """

    def __init__(
        self,
        mujoco_model_path: str,
        pinocchio_urdf_path: str,
        pin_joint_map: List[Tuple[int, int, int]],
        render: bool = True,
        dt: Optional[float] = None,
        ee_name: str = "attachment_site",
        ee_type: str = "site",
        camera_name: str = "track_cam",
        init_keyframe: Optional[str] = None,
        pin_ee_frame_name: str = "cylinder_link",
    ):
        """
        初始化混合仿真环境

        Args:
            mujoco_model_path: MuJoCo XML 模型文件路径
            pinocchio_urdf_path: Pinocchio URDF 模型文件路径
            pin_joint_map: 关节索引映射 [(mj_idx, pin_q_idx, pin_v_idx), ...]
                - mj_idx: MuJoCo 关节索引
                - pin_q_idx: Pinocchio 位置索引
                - pin_v_idx: Pinocchio 速度索引
            render: 是否启用渲染
            dt: 仿真时间步长
            ee_name: MuJoCo 末端执行器名称
            ee_type: MuJoCo 末端执行器类型 ("site" 或 "body")
            camera_name: 相机名称
            init_keyframe: 初始关键帧
            pin_ee_frame_name: Pinocchio 末端执行器 frame 名称
        """
        # 初始化 MuJoCo 环境
        super().__init__(
            model_path=mujoco_model_path,
            render=render,
            dt=dt,
            ee_name=ee_name,
            ee_type=ee_type,
            camera_name=camera_name,
            init_keyframe=init_keyframe,
        )

        # 初始化 Pinocchio 模型
        self.pin_model = pin.buildModelFromUrdf(pinocchio_urdf_path)
        self.pin_data = self.pin_model.createData()

        # 关节映射
        self.pin_joint_map = pin_joint_map

        # Pinocchio EE frame
        self.pin_ee_frame_name = pin_ee_frame_name
        self.pin_ee_frame_id = self.pin_model.getFrameId(pin_ee_frame_name)
        if self.pin_ee_frame_id >= len(self.pin_model.frames):
            raise ValueError(f"Pinocchio frame '{pin_ee_frame_name}' not found.")

        # 同步重力
        self.pin_model.gravity.linear = self.model.opt.gravity.copy()

        # 构建速度映射字典（加速访问）
        self._v_map_dict = {pin_v_idx: mj_idx for mj_idx, _, pin_v_idx in pin_joint_map}

    # ==================== 状态同步 ====================

    def sync_mujoco_to_pinocchio(self) -> None:
        """
        将 MuJoCo 状态同步到 Pinocchio

        更新 Pinocchio 的 q 和 dq，并计算前向运动学。
        """
        q_ctrl = np.zeros(self.pin_model.nq)
        dq_ctrl = np.zeros(self.pin_model.nv)

        for mj_idx, pin_q_idx, pin_v_idx in self.pin_joint_map:
            q_ctrl[pin_q_idx] = self.data.qpos[mj_idx]
            dq_ctrl[pin_v_idx] = self.data.qvel[mj_idx]

        pin.forwardKinematics(self.pin_model, self.pin_data, q_ctrl, dq_ctrl)
        pin.updateFramePlacements(self.pin_model, self.pin_data)

    def sync_pinocchio_to_mujoco(self, q_pin: np.ndarray, dq_pin: Optional[np.ndarray] = None) -> None:
        """
        将 Pinocchio 状态同步到 MuJoCo（用于 IK 结果等）

        Args:
            q_pin: Pinocchio 关节位置
            dq_pin: Pinocchio 关节速度（可选）
        """
        for mj_idx, pin_q_idx, _ in self.pin_joint_map:
            self.data.qpos[mj_idx] = q_pin[pin_q_idx]

        if dq_pin is not None:
            for mj_idx, _, pin_v_idx in self.pin_joint_map:
                self.data.qvel[mj_idx] = dq_pin[pin_v_idx]

        # 更新 MuJoCo 运动学
        mujoco.mj_forward(self.model, self.data)

    # ==================== 末端状态（Pinocchio） ====================

    def get_ee_state_pinocchio(self) -> Tuple[np.ndarray, np.ndarray]:
        """
        使用 Pinocchio 获取末端执行器状态（位置和线速度）

        Returns:
            pos: 末端位置 (3,)
            vel: 末端线速度 (3,)
        """
        self.sync_mujoco_to_pinocchio()

        H = self.pin_data.oMf[self.pin_ee_frame_id]
        pos = H.translation.copy()

        J = pin.getFrameJacobian(
            self.pin_model,
            self.pin_data,
            self.pin_ee_frame_id,
            pin.ReferenceFrame.LOCAL_WORLD_ALIGNED,
        )
        J_pos = J[:3, :]

        # 提取 MuJoCo 速度
        dq_pin = np.zeros(self.pin_model.nv)
        for mj_idx, _, pin_v_idx in self.pin_joint_map:
            dq_pin[pin_v_idx] = self.data.qvel[mj_idx]

        vel = J_pos @ dq_pin

        return pos, vel

    def get_ee_pose_pinocchio(self) -> Tuple[np.ndarray, np.ndarray]:
        """
        使用 Pinocchio 获取末端执行器位姿（位置和旋转矩阵）

        Returns:
            pos: 末端位置 (3,)
            rot: 末端旋转矩阵 (3, 3)
        """
        self.sync_mujoco_to_pinocchio()

        H = self.pin_data.oMf[self.pin_ee_frame_id]
        pos = H.translation.copy()
        rot = H.rotation.copy()

        return pos, rot

    # ==================== 动力学计算（Pinocchio） ====================

    def compute_mass_matrix_pinocchio(self) -> np.ndarray:
        """
        使用 Pinocchio 计算质量矩阵 M(q)

        Returns:
            M: 质量矩阵 (nv, nv) - 映射到 MuJoCo 维度
        """
        self.sync_mujoco_to_pinocchio()

        q_ctrl = np.zeros(self.pin_model.nq)
        for mj_idx, pin_q_idx, _ in self.pin_joint_map:
            q_ctrl[pin_q_idx] = self.data.qpos[mj_idx]

        M = pin.crba(self.pin_model, self.pin_data, q_ctrl)
        M = 0.5 * (M + M.T)  # 确保对称

        # 映射到 MuJoCo 维度
        M_mj = np.zeros((self.model.nv, self.model.nv))
        for i, (_, _, pin_v_idx_i) in enumerate(self.pin_joint_map):
            for j, (_, pin_q_idx_j, _) in enumerate(self.pin_joint_map):
                M_mj[i, j] = M[pin_v_idx_i, pin_q_idx_j]

        return M_mj

    def compute_jacobian_pinocchio(self) -> Tuple[np.ndarray, np.ndarray]:
        """
        使用 Pinocchio 计算末端雅可比矩阵

        Returns:
            jacp: 线速度雅可比 (3, nv) - 映射到 MuJoCo 维度
            jacr: 角速度雅可比 (3, nv) - 映射到 MuJoCo 维度
        """
        self.sync_mujoco_to_pinocchio()

        J = pin.getFrameJacobian(
            self.pin_model,
            self.pin_data,
            self.pin_ee_frame_id,
            pin.ReferenceFrame.LOCAL_WORLD_ALIGNED,
        )

        J_pos = J[:3, :]  # 线速度雅可比 (3, pin_nv)
        J_rot = J[3:, :]  # 角速度雅可比 (3, pin_nv)

        # 映射到 MuJoCo 维度
        jacp = np.zeros((3, self.model.nv))
        jacr = np.zeros((3, self.model.nv))

        for i, (_, _, pin_v_idx) in enumerate(self.pin_joint_map):
            jacp[:, i] = J_pos[:, pin_v_idx]
            jacr[:, i] = J_rot[:, pin_v_idx]

        return jacp, jacr

    # ==================== IK 求解（Pinocchio） ====================

    def solve_ik(
        self,
        target_pose: pin.SE3,
        initial_q: Optional[np.ndarray] = None,
        max_iters: int = 3000,
        eps: float = 1e-7,
        damp: float = 1e-8,
    ) -> Tuple[np.ndarray, bool]:
        """
        使用 Pinocchio 进行阻尼最小二乘 IK 求解

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
            q = pin.neutral(self.pin_model)
        else:
            q = initial_q.copy()

        for _ in range(max_iters):
            pin.forwardKinematics(self.pin_model, self.pin_data, q)
            pin.updateFramePlacements(self.pin_model, self.pin_data)

            current_pose = self.pin_data.oMf[self.pin_ee_frame_id]

            # 位置误差
            error_pos = target_pose.translation - current_pose.translation

            # 姿态误差（轴角表示）
            error_rot = pin.log3(target_pose.rotation @ current_pose.rotation.T)

            error = np.concatenate([error_pos, error_rot])

            # 检查收敛
            if np.linalg.norm(error) < eps:
                return q, True

            # 计算雅可比
            pin.computeJointJacobians(self.pin_model, self.pin_data, q)
            J = pin.getFrameJacobian(
                self.pin_model,
                self.pin_data,
                self.pin_ee_frame_id,
                pin.ReferenceFrame.LOCAL_WORLD_ALIGNED,
            )

            # 阻尼最小二乘
            Jt = J.T
            JJt = J @ Jt
            lambda_eye = damp * np.eye(6)

            v = np.linalg.solve(JJt + lambda_eye, error)
            dq = Jt @ v
            q = pin.integrate(self.pin_model, q, dq)

            # 关节限位
            if (
                hasattr(self.pin_model, "lowerPositionLimit")
                and hasattr(self.pin_model, "upperPositionLimit")
                and self.pin_model.lowerPositionLimit is not None
                and self.pin_model.upperPositionLimit is not None
            ):
                q = np.clip(
                    q,
                    self.pin_model.lowerPositionLimit,
                    self.pin_model.upperPositionLimit,
                )

        return q, False

    # ==================== 重写 reset 方法 ====================

    def reset(self, keyframe: Optional[str] = None) -> None:
        """
        重置仿真环境

        Args:
            keyframe: 关键帧名称（可选）
        """
        super().reset(keyframe)

        # 同步到 Pinocchio
        self.sync_mujoco_to_pinocchio()

    # ==================== 调试工具 ====================

    def debug_sync(self) -> Dict[str, np.ndarray]:
        """
        调试 MuJoCo 和 Pinocchio 的同步状态

        Returns:
            包含各种差值的字典
        """
        # MuJoCo EE 状态
        ee_pos_mj, _ = self.get_ee_state()
        ee_rot_mj = self.data.site(self.ee_name).xmat.copy().reshape(3, 3)

        # Pinocchio EE 状态
        ee_pos_pin, ee_rot_pin = self.get_ee_pose_pinocchio()

        # 雅可比差值
        jacp_mj, jacr_mj = self.compute_jacobian()
        jacp_pin, jacr_pin = self.compute_jacobian_pinocchio()

        # 质量矩阵差值
        M_mj = self.compute_mass_matrix()
        M_pin = self.compute_mass_matrix_pinocchio()

        return {
            "ee_pos_diff": ee_pos_mj - ee_pos_pin,
            "ee_rot_diff": ee_rot_mj - ee_rot_pin,
            "jacp_diff": jacp_mj - jacp_pin,
            "jacr_diff": jacr_mj - jacr_pin,
            "M_diff": M_mj - M_pin,
            "ee_pos_mj": ee_pos_mj,
            "ee_pos_pin": ee_pos_pin,
        }


# ==================== 测试代码 ====================


def test_hybrid_robot():
    """测试 HybridRobot 的基本功能"""
    import os
    import sys

    # 添加 mujoco 导入（避免运行时错误）
    import mujoco

    model_path = "./assets/kuka_xml_urdf/iiwa14_dock.xml"
    urdf_path = "./assets/kuka_xml_urdf/urdf/iiwa14_dock.urdf"

    print("=== Testing HybridRobot ===\n")

    # 构建关节映射
    env = MujocoRobot(model_path=model_path, render=False)
    env.reset(keyframe="home")

    mj_joint_names = []
    for i in range(env.model.njnt):
        name = mujoco.mj_id2name(env.model, mujoco.mjtObj.mjOBJ_JOINT, i)
        if name:
            mj_joint_names.append(name)

    name_map = {
        "joint1": "iiwa_joint_1",
        "joint2": "iiwa_joint_2",
        "joint3": "iiwa_joint_3",
        "joint4": "iiwa_joint_4",
        "joint5": "iiwa_joint_5",
        "joint6": "iiwa_joint_6",
        "joint7": "iiwa_joint_7",
    }

    # 重新创建 Pinocchio 模型（仅用于获取索引）
    pin_model = pin.buildModelFromUrdf(urdf_path)

    pin_joint_map = []
    for mj_idx, mj_name in enumerate(mj_joint_names):
        pin_name = name_map.get(mj_name)
        if pin_name is None:
            raise ValueError(f"No Pinocchio joint mapping for {mj_name}")
        pin_id = pin_model.getJointId(pin_name)
        pin_joint = pin_model.joints[pin_id]
        pin_joint_map.append((mj_idx, pin_joint.idx_q, pin_joint.idx_v))

    env.close()

    # 创建混合机器人
    with HybridRobot(
        mujoco_model_path=model_path,
        pinocchio_urdf_path=urdf_path,
        pin_joint_map=pin_joint_map,
        render=False,
        pin_ee_frame_name="cylinder_link",
    ) as robot:
        print(f"MuJoCo DOF: nq={robot.nq}, nv={robot.nv}")
        print(f"Pinocchio DOF: nq={robot.pin_model.nq}, nv={robot.pin_model.nv}")
        print(f"Joint mapping: {len(pin_joint_map)} joints\n")

        # 测试同步
        debug_info = robot.debug_sync()
        print("=== Sync Debug ===")
        print(f"EE pos diff norm: {np.linalg.norm(debug_info['ee_pos_diff']):.6f}")
        print(f"EE rot diff norm: {np.linalg.norm(debug_info['ee_rot_diff']):.6f}")
        print(f"Jacobian pos diff norm: {np.linalg.norm(debug_info['jacp_diff']):.6f}")
        print(f"Mass matrix diff norm: {np.linalg.norm(debug_info['M_diff']):.6f}")

        # 测试 IK
        print("\n=== Testing IK ===")
        ee_pos_pin, ee_rot_pin = robot.get_ee_pose_pinocchio()
        target_pos = ee_pos_pin + np.array([0.0, 0.1, 0.0])
        target_rot = ee_rot_pin.copy()
        target_pose = pin.SE3(target_rot, target_pos)

        initial_q = np.zeros(robot.pin_model.nq)
        for mj_idx, pin_q_idx, _ in pin_joint_map:
            initial_q[pin_q_idx] = robot.data.qpos[mj_idx]

        q_sol, success = robot.solve_ik(target_pose, initial_q, max_iters=100)
        print(f"IK success: {success}")
        print(f"Target pos: {target_pos}")
        print(f"Solved pos: {robot.get_ee_pose_pinocchio()[0]}")

        # 简单仿真循环测试
        print("\n=== Testing Simulation ===")
        for step in range(100):
            # 简单的重力补偿控制
            tau = np.zeros(robot.nu)
            q, dq, ee_pos = robot.step(tau)

            if step % 50 == 0:
                print(f"Step {step}: EE pos = {ee_pos}")

    print("\n=== Test Complete ===")


if __name__ == "__main__":
    test_hybrid_robot()
