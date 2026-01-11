"""
Pinocchio controller entrypoint.
"""

import argparse
import os
import sys
import time

import mujoco
import numpy as np
import pinocchio as pin
from tqdm import tqdm

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "src")))


from robot_learning import (
    DataLogger,
    DecoupledQuinticTrajectory,
    MujocoRobot,
    MujocoTaskSpaceController,
    PinocchioTaskSpaceController,
)
from robot_learning.utils import compute_ik


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run Pinocchio task-space control simulation.")
    parser.add_argument("--model-path", default="./assets/kuka_xml_urdf/iiwa14_dock.xml")
    parser.add_argument("--urdf-path", default="./assets/kuka_xml_urdf/urdf/iiwa14_dock.urdf")
    parser.add_argument("--duration", type=float, default=5.0)
    parser.add_argument("--dt", type=float, default=0.001)
    parser.add_argument("--ee-name", default="link7_site")
    parser.add_argument("--ee-type", default="site")
    parser.add_argument("--pin-ee-frame", default="iiwa_link_7")
    parser.add_argument("--use-external-comp", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--render", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--force-sensor-name", default=None)
    parser.add_argument("--torque-sensor-name", default=None)
    return parser.parse_args()


def simulate(
    mj_robot: MujocoRobot,
    controller: PinocchioTaskSpaceController,
    trajectory_planner: DecoupledQuinticTrajectory,
    log: DataLogger,
    duration: float,
    dt: float,
    pin_joint_map,
    use_external_comp: bool = False,
    force_sensor_name: str | None = None,
    torque_sensor_name: str | None = None,
):
    """仿真主循环（Pinocchio 控制器版本）。"""
    n_steps = int(duration / dt)
    sim_time = 0.0
    force_external = np.zeros(3)
    torque_external = np.zeros(3)

    with tqdm(total=n_steps) as pbar:
        for step in range(n_steps):
            q = mj_robot.data.qpos.copy()
            dq = mj_robot.data.qvel.copy()

            q_ctrl = np.zeros(controller.model.nq)
            dq_ctrl = np.zeros(controller.model.nv)
            for mj_idx, pin_q_idx, pin_v_idx in pin_joint_map:
                q_ctrl[pin_q_idx] = q[mj_idx]
                dq_ctrl[pin_v_idx] = dq[mj_idx]

            desired_pos, desired_vel, desired_acc = trajectory_planner.get_state(sim_time)

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

            tau_mj = np.zeros(mj_robot.model.nu)
            for mj_idx, _, pin_v_idx in pin_joint_map:
                tau_mj[mj_idx] = tau[pin_v_idx]

            q, dq, _ = mj_robot.step(tau_mj)

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

            act_pos, act_vel, _, _ = controller.get_task_space_state_with_orientation(q_ctrl, dq_ctrl)

            if step % 1000 == 0:
                print(f"\n=== Step {step} ===")
                print(f"sim_time: {sim_time:.4f}")
                print(f"desired_pos: {desired_pos}")
                print(f"actual_pos: {act_pos}")
                print(f"pos_error: {np.linalg.norm(desired_pos - act_pos):.6f} m")
                print(f"tau_norm: {np.linalg.norm(tau_mj):.6f}")

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

            sim_time += dt
            pbar.update(1)

            if mj_robot.render:
                if sim_time - mj_robot.data.time < mj_robot.dt:
                    time.sleep(mj_robot.dt - (sim_time - mj_robot.data.time))


def main():
    args = parse_args()
    sim_duration = args.duration
    dt = args.dt
    use_external_comp = args.use_external_comp
    force_sensor_name = args.force_sensor_name
    torque_sensor_name = args.torque_sensor_name

    env = MujocoRobot(
        model_path=args.model_path,
        render=args.render,
        dt=dt,
        ee_name=args.ee_name,
        ee_type=args.ee_type,
    )

    env.reset(keyframe="home")
    start_pos = env.get_ee_pos()
    goal_pos = start_pos + np.array([0.00, -0.00, -0.18])

    print("\n========== 轨迹规划信息 ==========")
    print(f"Start position: {start_pos}")
    print(f"Goal position: {goal_pos}")
    print(f"Distance: {np.linalg.norm(goal_pos - start_pos):.4f} m")
    print(f"Duration: {sim_duration:.1f} s")
    print("================================\n")

    pin_model = pin.buildModelFromUrdf(args.urdf_path)
    pin_model.gravity.linear = np.array([0.0, 0.0, 0.0])
    pin_data = pin_model.createData()

    name_map = {
        "joint1": "iiwa_joint_1",
        "joint2": "iiwa_joint_2",
        "joint3": "iiwa_joint_3",
        "joint4": "iiwa_joint_4",
        "joint5": "iiwa_joint_5",
        "joint6": "iiwa_joint_6",
        "joint7": "iiwa_joint_7",
    }

    mj_joint_names = []
    for i in range(env.model.njnt):
        name = mujoco.mj_id2name(env.model, mujoco.mjtObj.mjOBJ_JOINT, i)
        if name:
            mj_joint_names.append(name)

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

    q_init, success = compute_ik(pin_model, pin_data, init_pose, ee_frame_name, initial_q=initial_guess, max_iters=5000)
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
    goal_pos = start_pos + np.array([0.00, -0.00, -0.18])
    planner = DecoupledQuinticTrajectory(start_pos, goal_pos, duration=sim_duration)

    controller = PinocchioTaskSpaceController(
        pin_model,
        dt=dt,
        ee_frame_name=ee_frame_name,
        initial_orientation=init_ori,
    )

    debug_sync = os.environ.get("DEBUG_SYNC", "0") == "1"
    if debug_sync:
        q_mj = env.data.qpos.copy()
        dq_mj = env.data.qvel.copy()
        q_ctrl = np.zeros(pin_model.nq)
        dq_ctrl = np.zeros(pin_model.nv)
        for mj_idx, pin_q_idx, pin_v_idx in pin_joint_map:
            q_ctrl[pin_q_idx] = q_mj[mj_idx]
            dq_ctrl[pin_v_idx] = dq_mj[mj_idx]

        pin.forwardKinematics(pin_model, pin_data, q_ctrl, dq_ctrl)
        pin.updateFramePlacements(pin_model, pin_data)
        ee_pin = pin_data.oMf[pin_model.getFrameId(ee_frame_name)].translation
        ee_mj = env.get_ee_pos()
        print("\n========== Sync Check ==========")
        print(f"EE pos mujo: {ee_mj}")
        print(f"EE pos pin : {ee_pin}")
        print(f"EE pos diff norm: {np.linalg.norm(ee_pin - ee_mj):.6f} m")

        J_pin = pin.getFrameJacobian(pin_model, pin_data, pin_model.getFrameId(ee_frame_name), pin.ReferenceFrame.WORLD)
        Jp_pin = J_pin[:3, :]
        Jp_pin_mj = np.zeros((3, env.model.nv))
        for mj_idx, _, pin_v_idx in pin_joint_map:
            Jp_pin_mj[:, mj_idx] = Jp_pin[:, pin_v_idx]

        jacp = np.zeros((3, env.model.nv))
        jacr = np.zeros((3, env.model.nv))
        site_id = mujoco.mj_name2id(env.model, mujoco.mjtObj.mjOBJ_SITE, "link7_site")
        mujoco.mj_jacSite(env.model, env.data, jacp, jacr, site_id)
        diff = Jp_pin_mj - jacp
        col_norms = np.linalg.norm(diff, axis=0)
        print(f"Jacobian pos diff norm: {np.linalg.norm(diff):.6f}")
        print(f"Jacobian col diff norms: {col_norms}")

        g_pin = pin.computeGeneralizedGravity(pin_model, pin_data, q_ctrl)
        print(f"MuJoCo gravity: {env.model.opt.gravity}")
        print(f"Pinocchio gravity: {pin_model.gravity.linear}")
        print(f"g_pin norm: {np.linalg.norm(g_pin):.6f}")
        print("================================\n")

    logger = DataLogger()

    try:
        with env:
            simulate(
                mj_robot=env,
                controller=controller,
                trajectory_planner=planner,
                log=logger,
                duration=sim_duration,
                dt=dt,
                pin_joint_map=pin_joint_map,
                use_external_comp=use_external_comp,
                force_sensor_name=force_sensor_name,
                torque_sensor_name=torque_sensor_name,
            )

        if logger.desired_pos and logger.actual_pos:
            desired = np.array(logger.desired_pos)
            actual = np.array(logger.actual_pos)
            errors = np.linalg.norm(desired - actual, axis=1)
            print("\n========== Tracking Error Stats ==========")
            print(f"mean_error: {np.mean(errors):.6f} m")
            print(f"max_error: {np.max(errors):.6f} m")
            print(f"final_error: {errors[-1]:.6f} m")
            print("=========================================\n")

        logger.save()
        logger.plot(show=True, save=False)

    except Exception as e:
        print(f"Error during simulation: {e}")
        import traceback

        traceback.print_exc()


if __name__ == "__main__":
    main()
