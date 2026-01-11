"""仿真数据记录工具。"""

import os
from datetime import datetime

import matplotlib.pyplot as plt
import numpy as np


class DataLogger:
    """仿真数据记录与绘图。

    记录期望/实际状态、力矩以及可选外力数据。
    """

    def __init__(self, log_dir="outputs/logs"):
        self.log_dir = log_dir
        os.makedirs(self.log_dir, exist_ok=True)
        self.reset()

    def reset(self):
        """重置所有记录缓存。"""
        self.times = []
        self.desired_pos = []
        self.desired_vel = []
        self.desired_acc = []
        self.actual_pos = []
        self.actual_vel = []
        self.control_torques = []
        self.joint_positions = []
        self.joint_velocities = []
        self.external_forces = []
        self.external_torques = []
        self.sensor_raw = []

    def record(
        self,
        time,
        des_pos,
        des_vel,
        des_acc,
        act_pos,
        act_vel,
        tau,
        q,
        dq,
        external_force=None,
        external_torque=None,
        sensor_raw=None,
    ):
        """记录单步仿真数据。

        Args:
            time: 仿真时间。
            des_pos: 期望末端位置。
            des_vel: 期望末端速度。
            des_acc: 期望末端加速度。
            act_pos: 实际末端位置。
            act_vel: 实际末端速度。
            tau: 关节力矩。
            q: 关节位置。
            dq: 关节速度。
            external_force: 外部力（可选）。
            external_torque: 外部力矩（可选）。
            sensor_raw: 原始传感器数据（可选）。
        """
        self.times.append(time)
        self.desired_pos.append(des_pos.copy())
        self.desired_vel.append(des_vel.copy())
        self.desired_acc.append(des_acc.copy())
        self.actual_pos.append(act_pos.copy())
        self.actual_vel.append(act_vel.copy())
        self.control_torques.append(tau.copy())
        self.joint_positions.append(q.copy())
        self.joint_velocities.append(dq.copy())
        if external_force is not None:
            self.external_forces.append(np.array(external_force).copy())
        if external_torque is not None:
            self.external_torques.append(np.array(external_torque).copy())
        if sensor_raw is not None:
            self.sensor_raw.append(sensor_raw)

    def save(self, filename=None):
        """保存记录数据到 npz 文件。

        Args:
            filename: 可选的日志文件名。
        """
        if filename is None:
            filename = f"log_{datetime.now().strftime('%Y%m%d_%H%M%S')}.npz"

        filepath = os.path.join(self.log_dir, filename)
        np.savez(
            filepath,
            times=np.array(self.times),
            desired_pos=np.array(self.desired_pos),
            desired_vel=np.array(self.desired_vel),
            desired_acc=np.array(self.desired_acc),
            actual_pos=np.array(self.actual_pos),
            actual_vel=np.array(self.actual_vel),
            control_torques=np.array(self.control_torques),
            joint_positions=np.array(self.joint_positions),
            joint_velocities=np.array(self.joint_velocities),
            external_forces=np.array(self.external_forces),
            external_torques=np.array(self.external_torques),
            sensor_raw=np.array(self.sensor_raw, dtype=object),
        )
        print(f"Data saved to {filepath}")

    def plot(self, show=True, save=False):
        """绘制记录的信号。

        Args:
            show: 是否显示图形。
            save: 是否保存图形到磁盘。
        """
        times = np.array(self.times)
        des_pos = np.array(self.desired_pos)
        act_pos = np.array(self.actual_pos)
        tau = np.array(self.control_torques)

        # 1. Position Tracking
        fig1, axes1 = plt.subplots(3, 1, figsize=(10, 12), sharex=True)
        labels = ["X", "Y", "Z"]
        for i in range(3):
            axes1[i].plot(times, des_pos[:, i], "r--", label="Desired")
            axes1[i].plot(times, act_pos[:, i], "b-", label="Actual")
            axes1[i].set_ylabel(f"{labels[i]} Position [m]")
            axes1[i].legend()
            axes1[i].grid(True)
        axes1[-1].set_xlabel("Time [s]")
        fig1.suptitle("End-Effector Position Tracking")

        # 2. Tracking Error
        fig2, ax2 = plt.subplots(figsize=(10, 6))
        error = des_pos - act_pos
        for i in range(3):
            ax2.plot(times, error[:, i], label=f"{labels[i]} Error")
        ax2.set_xlabel("Time [s]")
        ax2.set_ylabel("Error [m]")
        ax2.legend()
        ax2.grid(True)
        ax2.set_title("Position Tracking Error")

        # 3. Control Torques
        fig3, ax3 = plt.subplots(figsize=(10, 6))
        for i in range(tau.shape[1]):
            ax3.plot(times, tau[:, i], label=f"Joint {i + 1}")
        ax3.set_xlabel("Time [s]")
        ax3.set_ylabel("Torque [Nm]")
        ax3.legend()
        ax3.grid(True)
        ax3.set_title("Control Torques")

        # 4. External Forces / Torques (if provided)
        fig4, fig5 = None, None
        if self.external_forces:
            ext_f = np.array(self.external_forces)
            fig4, ax4 = plt.subplots(figsize=(10, 6))
            for i in range(ext_f.shape[1]):
                ax4.plot(times[: ext_f.shape[0]], ext_f[:, i], label=f"Axis {i + 1}")
            ax4.set_xlabel("Time [s]")
            ax4.set_ylabel("Force [N]")
            ax4.legend()
            ax4.grid(True)
            ax4.set_title("External Forces")

        if self.external_torques:
            ext_t = np.array(self.external_torques)
            fig5, ax5 = plt.subplots(figsize=(10, 6))
            for i in range(ext_t.shape[1]):
                ax5.plot(times[: ext_t.shape[0]], ext_t[:, i], label=f"Axis {i + 1}")
            ax5.set_xlabel("Time [s]")
            ax5.set_ylabel("Torque [Nm]")
            ax5.legend()
            ax5.grid(True)
            ax5.set_title("External Torques")

        if save:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            fig1.savefig(os.path.join(self.log_dir, f"pos_tracking_{timestamp}.png"))
            fig2.savefig(os.path.join(self.log_dir, f"pos_error_{timestamp}.png"))
            fig3.savefig(os.path.join(self.log_dir, f"torques_{timestamp}.png"))
            if self.external_forces and fig4 is not None:
                fig4.savefig(os.path.join(self.log_dir, f"external_forces_{timestamp}.png"))
            if self.external_torques and fig5 is not None:
                fig5.savefig(os.path.join(self.log_dir, f"external_torques_{timestamp}.png"))

        if show:
            plt.show()
