"""Pinocchio 逆运动学工具函数。"""

from __future__ import annotations

from typing import Optional, Tuple

import numpy as np
import pinocchio as pin


def compute_ik(
    pin_model: pin.Model,
    pin_data: pin.Data,
    target_pose: pin.SE3,
    ee_frame_name: str,
    initial_q: Optional[np.ndarray] = None,
    max_iters: int = 3000,
    eps: float = 1e-7,
    damp: float = 1e-8,
) -> Tuple[np.ndarray, bool]:
    """使用 Pinocchio 进行阻尼最小二乘 IK。

    Args:
        pin_model: Pinocchio 机器人模型。
        pin_data: Pinocchio 数据结构。
        target_pose: 目标末端位姿。
        ee_frame_name: 末端执行器 frame 名称。
        initial_q: 迭代初值（未提供则使用 neutral）。
        max_iters: 最大迭代次数。
        eps: 收敛阈值。
        damp: 阻尼系数。

    Returns:
        (q, success) 关节解与是否收敛。
    """
    if initial_q is None:
        q = pin.neutral(pin_model)
    else:
        q = initial_q.copy()

    ee_frame_id = pin_model.getFrameId(ee_frame_name)
    if ee_frame_id < 0:
        raise ValueError(f"Frame '{ee_frame_name}' not found in Pinocchio model.")

    for _ in range(max_iters):
        pin.forwardKinematics(pin_model, pin_data, q)
        pin.updateFramePlacements(pin_model, pin_data)

        current_pose = pin_data.oMf[ee_frame_id]
        error_pos = target_pose.translation - current_pose.translation
        error_rot = pin.log3(target_pose.rotation @ current_pose.rotation.T)
        error = np.concatenate([error_pos, error_rot])

        if np.linalg.norm(error) < eps:
            return q, True

        pin.computeJointJacobians(pin_model, pin_data, q)
        J = pin.getFrameJacobian(pin_model, pin_data, ee_frame_id, pin.ReferenceFrame.LOCAL_WORLD_ALIGNED)
        Jt = J.T
        JJt = J @ Jt
        lambda_eye = damp * np.eye(6)

        v = np.linalg.solve(JJt + lambda_eye, error)
        dq = Jt @ v
        q = pin.integrate(pin_model, q, dq)

        if hasattr(pin_model, "lowerPositionLimit") and hasattr(pin_model, "upperPositionLimit"):
            lower = pin_model.lowerPositionLimit
            upper = pin_model.upperPositionLimit
            if lower is not None and upper is not None and len(lower) == len(q):
                q = np.clip(q, lower, upper)

    return q, False
