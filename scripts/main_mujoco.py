"""
MuJoCo controller entrypoint.
"""

import argparse
import os
import sys
import time

import mujoco
import numpy as np
from tqdm import tqdm

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "src")))


from robot_learning import (
    DataLogger,
    DecoupledQuinticTrajectory,
    MujocoRobot,
    MujocoTaskSpaceController,
    PinocchioTaskSpaceController,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run MuJoCo task-space control simulation.")
    parser.add_argument("--model-path", default="./assets/kuka_xml_urdf/iiwa14_dock.xml")
    parser.add_argument("--duration", type=float, default=5.0)
    parser.add_argument("--dt", type=float, default=0.001)
    parser.add_argument("--ee-name", default="link7_site")
    parser.add_argument("--ee-type", default="site")
    parser.add_argument("--use-external-comp", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--render", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--force-sensor-name", default=None)
    parser.add_argument("--torque-sensor-name", default=None)
    return parser.parse_args()


def simulate(
    mj_robot: MujocoRobot,
    controller: MujocoTaskSpaceController,
    trajectory_planner: DecoupledQuinticTrajectory,
    log: DataLogger,
    duration: float,
    dt: float,
    use_external_comp: bool = False,
    force_sensor_name: str | None = None,
    torque_sensor_name: str | None = None,
):
    """仿真主循环（MuJoCo 控制器版本）。"""
    n_steps = int(duration / dt)
    sim_time = 0.0
    force_external = np.zeros(3)
    torque_external = np.zeros(3)

    with tqdm(total=n_steps) as pbar:
        for step in range(n_steps):
            q = mj_robot.data.qpos.copy()
            dq = mj_robot.data.qvel.copy()

            desired_pos, desired_vel, desired_acc = trajectory_planner.get_state(sim_time)

            if use_external_comp:
                tau = controller.compute_control_task_space_with_orientation_and_imp(
                    q=q,
                    dq=dq,
                    desired_ee_pos=desired_pos,
                    desired_ee_vel=desired_vel,
                    desired_ee_acc=desired_acc,
                    force_external=force_external,
                )
            else:
                tau = controller.compute_control_task_space(
                    q=q,
                    dq=dq,
                    desired_ee_pos=desired_pos,
                    desired_ee_vel=desired_vel,
                    desired_ee_acc=desired_acc,
                )

            q, dq, _ = mj_robot.step(tau)

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

            act_pos, act_vel, _, _ = controller.get_task_space_state_with_orientation(q, dq)

            if step % 1000 == 0:
                print(f"\n=== Step {step} ===")
                print(f"sim_time: {sim_time:.4f}")
                print(f"desired_pos: {desired_pos}")
                print(f"actual_pos: {act_pos}")
                print(f"pos_error: {np.linalg.norm(desired_pos - act_pos):.6f} m")
                print(f"tau_norm: {np.linalg.norm(tau):.6f}")

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

    planner = DecoupledQuinticTrajectory(start_pos, goal_pos, duration=sim_duration)

    controller = MujocoTaskSpaceController(env.model, env.data, ee_site_name=args.ee_name, dt=dt)

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
