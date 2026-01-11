"""
Author: Your Name
Date: 2026-01-03 11:39:24
LastEditors: Your Name
LastEditTime: 2026-01-11 01:51:30
Description:
"""

import argparse
import os
import sys
import time
from typing import Optional

import mujoco
import numpy as np
import pinocchio as pin
from tqdm import tqdm

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "src")))


from robot_learning import (
    DataLogger,
    DecoupledQuinticTrajectory,
    MujocoRobot,
    MujocoTaskSpaceController,
    PinocchioTaskSpaceController,
)
from robot_learning.utils import compute_ik


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run MuJoCo/Pinocchio task-space control simulation.")
    parser.add_argument("--model-path", default="./assets/kuka_xml_urdf/iiwa14_dock.xml")
    parser.add_argument("--urdf-path", default="./assets/kuka_xml_urdf/urdf/iiwa14_dock.urdf")
    parser.add_argument("--duration", type=float, default=5.0)
    parser.add_argument("--dt", type=float, default=0.001)
    parser.add_argument("--ee-name", default="attachment_site")
    parser.add_argument("--ee-type", default="site")
    parser.add_argument("--pin-ee-frame", default="cylinder_link")
    parser.add_argument(
        "--use-pinocchio",
        action=argparse.BooleanOptionalAction,
        default=os.environ.get("USE_PINOCCHIO", "0") == "1",
    )
    parser.add_argument("--use-external-comp", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--render", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--force-sensor-name", default=None)
    parser.add_argument("--torque-sensor-name", default=None)
    return parser.parse_args()


def simulate(
    mj_robot: MujocoRobot,
    controller,
    trajectory_planner: DecoupledQuinticTrajectory,
    log: DataLogger,
    duration: float,
    dt: float,
    q_init: Optional[np.ndarray] = None,
    use_external_comp: bool = False,
    force_sensor_name: Optional[str] = None,
    torque_sensor_name: Optional[str] = None,
    pin_joint_map: Optional[list[tuple[int, int, int]]] = None,
):
    """仿真主循环

    Args:
        mj_robot (MujocoRobot): 机器人实例
        controller: 机器人控制器实例
        trajectory_planner (DecoupledQuinticTrajectory): 轨迹规划器实例
        log (DataLogger):   数据记录器实例
        duration (float):   仿真总时长
        dt (float): 仿真时间步长
        q_init (np.ndarray): 初始关节角
        use_external_comp (bool): 是否启用外力补偿
        force_sensor_name (str): 外力传感器名称（MuJoCo sensor）
        torque_sensor_name (str): 外力矩传感器名称（MuJoCo sensor）
        pin_joint_map: Pinocchio 关节索引映射（mujoco_idx, pin_q_idx, pin_v_idx）
    """
    # 仿真初始化已在 main() 中完成 / Initialization completed in main()

    n_steps = int(duration / dt)
    sim_time = 0.0
    force_external = np.zeros(3)
    torque_external = np.zeros(3)

    with tqdm(total=n_steps) as pbar:
        for step in range(n_steps):
            # 1. 获取机器人状态 / Get robot state
            q = mj_robot.data.qpos.copy()
            dq = mj_robot.data.qvel.copy()

            if pin_joint_map is not None:
                q_ctrl = np.zeros(controller.model.nq)
                dq_ctrl = np.zeros(controller.model.nv)
                for mj_idx, pin_q_idx, pin_v_idx in pin_joint_map:
                    q_ctrl[pin_q_idx] = q[mj_idx]
                    dq_ctrl[pin_v_idx] = dq[mj_idx]
            else:
                q_ctrl = q
                dq_ctrl = dq

            # 2. 获取期望轨迹点 / Get desired trajectory point
            desired_pos, desired_vel, desired_acc = trajectory_planner.get_state(sim_time)

            # 3. 计算控制输入 / Compute control input
            if use_external_comp:
                tau = controller.compute_control_task_space_with_orientation_and_imp(
                    q=q_ctrl,
                    dq=dq_ctrl,
                    desired_ee_pos=desired_pos,
                    desired_ee_vel=desired_vel,
                    desired_ee_acc=desired_acc,
                    force_external=force_external,
                )
            else:
                tau = controller.compute_control_task_space_with_orientation_and_imp(
                    q=q_ctrl,
                    dq=dq_ctrl,
                    desired_ee_pos=desired_pos,
                    desired_ee_vel=desired_vel,
                    desired_ee_acc=desired_acc,
                )

            if pin_joint_map is not None:
                tau_mj = np.zeros(mj_robot.model.nu)
                for mj_idx, _, pin_v_idx in pin_joint_map:
                    tau_mj[mj_idx] = tau[pin_v_idx]
            else:
                tau_mj = tau

            # 4. 应用控制输入并推进仿真 / Apply control input and step simulation
            q, dq, _ = mj_robot.step(tau_mj)

            # 5. 读取传感器并更新外力估计 / Read sensors and update external force
            if use_external_comp:
                if force_sensor_name is not None:
                    try:
                        force_external = -mj_robot.get_sensor_data(force_sensor_name)
                    except Exception as e:
                        if step < 5:
                            print(f"Force sensor read failed: {e}")
                if torque_sensor_name is not None:
                    try:
                        torque_external = -mj_robot.get_sensor_data(torque_sensor_name)
                    except Exception as e:
                        if step < 5:
                            print(f"Torque sensor read failed: {e}")

            # 获取实际末端状态用于记录 / Get actual EE state for logging
            # Note: controller.get_task_space_state_with_orientation returns (pos, vel_pos, ori_err, vel_rot)
            act_pos, act_vel, _, _ = controller.get_task_space_state_with_orientation(q_ctrl, dq_ctrl)

            # 调试：打印每隔一定步数的信息
            if step % 1000 == 0:
                print(f"\n=== Step {step} ===")
                print(f"sim_time: {sim_time:.4f}")
                print(f"desired_pos: {desired_pos}")
                print(f"actual_pos: {act_pos}")
                print(f"pos_error: {np.linalg.norm(desired_pos - act_pos):.6f} m")
                print(f"tau: {tau_mj}")
                print(f"tau_norm: {np.linalg.norm(tau_mj):.6f}")
                if use_external_comp:
                    print(f"force_external: {force_external}")
                    if torque_sensor_name is not None:
                        print(f"torque_external: {torque_external}")

            # 6. 记录数据 / Log data
            log.record(
                sim_time,
                desired_pos,
                desired_vel,
                desired_acc,
                act_pos,
                act_vel,
                tau,
                q,
                dq,
                external_force=force_external if use_external_comp else None,
                external_torque=torque_external if use_external_comp else None,
            )

            # 更新时间和进度条 / Update time and progress bar
            sim_time += dt
            pbar.update(1)

            # 渲染 (如果启用) / Render (if enabled)
            if mj_robot.render:
                # 控制渲染帧率 / Control render frame rate
                if sim_time - mj_robot.data.time < mj_robot.dt:
                    time.sleep(mj_robot.dt - (sim_time - mj_robot.data.time))

    pass


def main():
    args = parse_args()
    use_pinocchio_controller = args.use_pinocchio
    print(f"Using Pinocchio controller: {use_pinocchio_controller}")

    env = MujocoRobot(
        model_path=args.model_path,
        render=args.render,
        dt=args.dt,
        ee_name=args.ee_name,
        ee_type=args.ee_type,
    )

    sim_duration = args.duration
    dt = args.dt
    use_external_comp = args.use_external_comp
    force_sensor_name = args.force_sensor_name
    torque_sensor_name = args.torque_sensor_name

    # B. 规划器 (计算起点和终点)
    # 重置到 home 关键帧获取起点位置（reset() 已调用 mj_forward）
    env.reset(keyframe="home")
    start_pos = env.get_ee_pos()
    goal_pos = start_pos + np.array([0.00, -0.00, -0.15])

    print("\n========== 轨迹规划信息 ==========")
    print(f"Start position: {start_pos}")
    print(f"Goal position: {goal_pos}")
    print(f"Distance: {np.linalg.norm(goal_pos - start_pos):.4f} m")
    print(f"Duration: {sim_duration:.1f} s")
    print("================================\n")

    planner = DecoupledQuinticTrajectory(start_pos, goal_pos, duration=sim_duration)

    # C. 控制器
    if use_pinocchio_controller:
        pin_model = pin.buildModelFromUrdf(args.urdf_path)
        pin_data = pin_model.createData()
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

        pin_joint_map = []
        for mj_idx, mj_name in enumerate(mj_joint_names):
            pin_name = name_map.get(mj_name)
            if pin_name is None:
                raise ValueError(f"No Pinocchio joint mapping for {mj_name}")
            pin_id = pin_model.getJointId(pin_name)
            if pin_id <= 0:
                raise ValueError(f"Pinocchio joint not found: {pin_name}")
            pin_joint = pin_model.joints[pin_id]
            pin_joint_map.append((mj_idx, pin_joint.idx_q, pin_joint.idx_v))

        ee_frame_name = args.pin_ee_frame
        init_pos = start_pos.copy()
        init_ori = np.array(
            [
                [1, 0, 0],
                [0, -1, 0],
                [0, 0, -1],
            ]
        )
        init_pose = pin.SE3(init_ori, init_pos)
        initial_guess = np.zeros(pin_model.nq)
        qpos_home = env.data.qpos.copy()
        for mj_idx, pin_q_idx, _ in pin_joint_map:
            initial_guess[pin_q_idx] = qpos_home[mj_idx]
        q_init, success = compute_ik(
            pin_model,
            pin_data,
            init_pose,
            ee_frame_name,
            initial_q=initial_guess,
            max_iters=5000,
        )
        if not success:
            print("Warning: IK did not converge perfectly, using current MuJoCo state.")
        else:
            qpos_mj = env.data.qpos.copy()
            qvel_mj = env.data.qvel.copy()
            for mj_idx, pin_q_idx, _ in pin_joint_map:
                qpos_mj[mj_idx] = q_init[pin_q_idx]
            env.data.qpos[:] = qpos_mj
            env.data.qvel[:] = qvel_mj * 0.0
            mujoco.mj_forward(env.model, env.data)

        pin.forwardKinematics(pin_model, pin_data, q_init)
        pin.updateFramePlacements(pin_model, pin_data)
        start_pos_ctrl = pin_data.oMf[pin_model.getFrameId(ee_frame_name)].translation
        align_error = np.linalg.norm(start_pos_ctrl - init_pos)
        print(f"IK alignment error: {align_error:.6f} m")

        start_pos = start_pos_ctrl
        goal_pos = start_pos + np.array([0.00, -0.00, -0.15])
        planner = DecoupledQuinticTrajectory(start_pos, goal_pos, duration=sim_duration)

        controller = PinocchioTaskSpaceController(
            pin_model,
            dt=dt,
            ee_frame_name=ee_frame_name,
            initial_orientation=np.array(
                [
                    [1, 0, 0],
                    [0, -1, 0],
                    [0, 0, -1],
                ]
            ),
        )
    else:
        pin_joint_map = None
        controller = MujocoTaskSpaceController(env.model, env.data, ee_site_name=args.ee_name, dt=dt)

    # D. 记录器
    logger = DataLogger()

    # 3. 运行仿真
    try:
        with env:  # 使用上下文管理器自动处理关闭
            simulate(
                mj_robot=env,
                controller=controller,
                trajectory_planner=planner,
                log=logger,
                duration=sim_duration,
                dt=dt,
                q_init=None,
                use_external_comp=use_external_comp,
                force_sensor_name=force_sensor_name,
                torque_sensor_name=torque_sensor_name,
                pin_joint_map=pin_joint_map,
            )

        # 4. 仿真结束后统计
        if logger.desired_pos and logger.actual_pos:
            desired = np.array(logger.desired_pos)
            actual = np.array(logger.actual_pos)
            errors = np.linalg.norm(desired - actual, axis=1)
            print("\n========== Tracking Error Stats ==========")
            print(f"mean_error: {np.mean(errors):.6f} m")
            print(f"max_error: {np.max(errors):.6f} m")
            print(f"final_error: {errors[-1]:.6f} m")
            print("=========================================\n")

        # 5. 仿真结束后保存与绘图
        # logger.save()
        logger.plot(show=False, save=False)

    except Exception as e:
        print(f"Error during simulation: {e}")
        import traceback

        traceback.print_exc()


if __name__ == "__main__":
    main()
