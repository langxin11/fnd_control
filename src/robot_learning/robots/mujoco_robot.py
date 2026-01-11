"""MuJoCo 机器人仿真环境。"""

import os
import warnings
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import imageio
import mujoco
import mujoco.viewer
import numpy as np


class MujocoRobot:
    """MuJoCo 机器人仿真环境。

    功能特性:
        - 模型加载与仿真步进。
        - 运动学与动力学计算。
        - 传感器数据读取。
        - 交互式/离线渲染。
        - 碰撞检测与关节限位检查。
        - Mocap 目标控制。
        - 上下文管理器支持。
    """

    def __init__(
        self,  # noqa: W291
        model_path: str,  # noqa: W291
        render: bool = True,
        record: bool = False,  # noqa: W291
        video_path: str = "simulation.mp4",
        video_fps: int = 30,
        dt: Optional[float] = None,
        ee_name: str = "attachment_site",
        ee_type: str = "site",
        camera_name: str = "track_cam",
        init_keyframe: Optional[str] = None,
    ):
        """
        初始化 MuJoCo 机器人环境

        Args:
            model_path: MuJoCo XML 模型文件路径
            render: 是否启用交互式渲染
            record: 是否启用离线渲染录像
            video_path: 离线渲染视频保存路径
            video_fps: 离线渲染视频帧率
            dt: 仿真时间步长,若为 None 则使用模型默认值
            ee_name: 末端执行器名称
            ee_type: 末端执行器类型,"site" 或 "body"
            camera_name: 相机名称
            init_keyframe: 初始化关键帧名称
        """
        self.model_path = Path(model_path)
        self.ee_name = ee_name
        self.ee_type = ee_type
        self.camera_name = camera_name
        self.init_keyframe = init_keyframe
        # noqa: W293
        # ---- 加载模型 ----
        self._setup_model()

        # 设置时间步长
        if dt is not None:
            self.model.opt.timestep = dt
        self.dt = self.model.opt.timestep

        # 渲染配置
        """ self.headless = "DISPLAY" not in os.environ or not os.environ["DISPLAY"]
        self.render = render and not self.headless
        if render and self.headless:
            warnings.warn("DISPLAY environment variable not found. Disabling interactive rendering.") """
        self.render = render

        # 离线渲染相关
        self.record = record
        self.frames: List[np.ndarray] = []
        self.video_path = video_path
        self.video_fps = video_fps
        self.renderer: Optional[mujoco.Renderer] = None
        self.render_options: Optional[mujoco.MjvOption] = None

        if self.record:
            self._setup_rendering()

        # 交互式渲染相关
        self.viewer: Optional[mujoco.viewer.Handle] = None
        if self.render:
            self._setup_viewer()

        # 初始化到指定关键帧
        if self.init_keyframe is not None:
            self.reset(self.init_keyframe)

    def _setup_model(self):
        """加载 MuJoCo 模型与数据"""
        self.model = mujoco.MjModel.from_xml_path(str(self.model_path))
        self.data = mujoco.MjData(self.model)

        # 维度信息
        self.nu = self.model.nu  # 控制输入维度
        self.nq = self.model.nq  # 状态位置维度
        self.nv = self.model.nv  # 状态速度维度
        self.nsensor = self.model.nsensor  # 传感器数量

        # 获取末端执行器 ID（支持 site 和body）
        if self.ee_type == "site":
            obj_type = mujoco.mjtObj.mjOBJ_SITE
        elif self.ee_type == "body":
            obj_type = mujoco.mjtObj.mjOBJ_BODY
        else:
            raise ValueError(f"Invalid ee_type: {self.ee_type}. Must be 'site' or 'body'.")

        self.eef_id = mujoco.mj_name2id(self.model, obj_type, self.ee_name)
        if self.eef_id == -1:
            raise ValueError(f"End-effector '{self.ee_name}' of type '{self.ee_type}' not found in model.")

        # 获取相机 ID
        self.camera_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_CAMERA, self.camera_name)
        if self.camera_id == -1:
            warnings.warn(f"Camera '{self.camera_name}' not found in model. Using default camera.")
            self.camera_id = 0

    def _setup_viewer(self):
        """初始化交互式查看器"""
        try:
            self.viewer = mujoco.viewer.launch_passive(
                model=self.model, data=self.data, show_left_ui=False, show_right_ui=False
            )
            # 开启可视化标志
            mujoco.mjv_defaultFreeCamera(self.model, self.viewer.cam)
            self.viewer.opt.flags[mujoco.mjtVisFlag.mjVIS_PERTFORCE] = True
            self.viewer.opt.flags[mujoco.mjtVisFlag.mjVIS_TRANSPARENT] = True
            self.viewer.opt.flags[mujoco.mjtVisFlag.mjVIS_CONTACTPOINT] = True
            self.viewer.opt.flags[mujoco.mjtVisFlag.mjVIS_CONTACTFORCE] = True
        except Exception as e:
            warnings.warn(f"Viewer launch failed: {e}")
            self.render = False

    def _setup_rendering(self):
        """初始化离屏渲染器"""
        self.render_options = mujoco.MjvOption()
        mujoco.mjv_defaultOption(self.render_options)
        self.render_options.flags[mujoco.mjtVisFlag.mjVIS_PERTFORCE] = True
        self.render_options.flags[mujoco.mjtVisFlag.mjVIS_TRANSPARENT] = True
        self.render_options.flags[mujoco.mjtVisFlag.mjVIS_CONTACTPOINT] = True
        self.render_options.flags[mujoco.mjtVisFlag.mjVIS_CONTACTFORCE] = True

        self.renderer = mujoco.Renderer(self.model, height=720, width=1280)

    # ==================== 仿真步进 ====================

    def step(self, tau: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        执行单步仿真

        Args:
            tau: 控制输入向量,形状为 (nu,)

        Returns:
            qpos: 关节位置
            qvel: 关节速度
            eef_pos: 末端执行器位置
        """
        if tau.shape != (self.model.nu,):
            raise ValueError(f"Control input tau must have shape ({self.model.nu},), but got {tau.shape}")

        self.data.ctrl[:] = tau
        mujoco.mj_step(self.model, self.data)

        # 输出常用状态
        qpos = self.data.qpos.copy()
        qvel = self.data.qvel.copy()
        eef_pos = self.get_ee_pos()

        # 渲染逻辑
        if self.render and self.viewer is not None:
            self.viewer.sync()

        # 录制逻辑
        if self.record and self.renderer is not None:
            expected_frames = int(self.data.time * self.video_fps)
            if len(self.frames) < expected_frames:
                try:
                    self.renderer.update_scene(self.data, camera=self.camera_id)
                    pixels = self.renderer.render()
                    self.frames.append(pixels)
                except Exception as e:
                    warnings.warn(f"Rendering failed: {e}")

        return qpos, qvel, eef_pos

    def reset(self, keyframe: Optional[str] = None) -> None:
        """
        重置仿真环境到初始状态或指定关键帧

        Args:
            keyframe: 关键帧名称,若为 None 则使用默认初始状态
        """
        if keyframe is not None:
            key_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_KEY, keyframe)
            if key_id == -1:
                raise ValueError(f"Keyframe '{keyframe}' not found in model.")
            mujoco.mj_resetDataKeyframe(self.model, self.data, key_id)
        else:
            mujoco.mj_resetData(self.model, self.data)
        # 更新正运动学
        mujoco.mj_forward(self.model, self.data)
        # 清空录制帧
        if self.record:
            self.frames = []

    # ==================== 状态获取 ====================

    def get_joint_states(self) -> Tuple[np.ndarray, np.ndarray]:
        """获取当前关节位置和速度"""
        return self.data.qpos.copy(), self.data.qvel.copy()

    def get_ee_pos(self) -> np.ndarray:
        """获取末端执行器位置"""
        if self.ee_type == "site":
            return self.data.site(self.ee_name).xpos.copy()
        else:  # body
            return self.data.body(self.ee_name).xpos.copy()

    def get_ee_state(self) -> Tuple[np.ndarray, np.ndarray]:
        """获取末端执行器位置和线速度"""
        # 1. 获取位置 (两者的API稍有不同，分开写)
        if self.ee_type == "site":
            pos = self.data.site(self.ee_name).xpos.copy()
            obj_type = mujoco.mjtObj.mjOBJ_SITE
        else:  # body
            pos = self.data.body(self.ee_name).xpos.copy()
            obj_type = mujoco.mjtObj.mjOBJ_BODY

        # 2. 获取速度 (使用统一的接口计算 6D 速度)
        # 获取对象 ID
        obj_id = mujoco.mj_name2id(self.model, obj_type, self.ee_name)

        # 计算 6D 速度 (前3位角速度，后3位线速度)
        vel_6d = np.zeros(6)
        mujoco.mj_objectVelocity(self.model, self.data, obj_type, obj_id, vel_6d, 0)

        # 提取线速度 (Linear Velocity)
        vel = vel_6d[3:6].copy()

        return pos, vel

    def get_ee_pose(self) -> Tuple[np.ndarray, np.ndarray]:
        """获取末端执行器位置和姿态（旋转矩阵）"""
        if self.ee_type == "site":
            pos = self.data.site(self.ee_name).xpos.copy()
            mat = self.data.site(self.ee_name).xmat.copy().reshape(3, 3)
        else:  # body
            pos = self.data.body(self.ee_name).xpos.copy()
            mat = self.data.body(self.ee_name).xmat.copy().reshape(3, 3)
        return pos, mat

    # ==================== 传感器 ====================

    def get_sensor_data(self, sensor_name: str) -> np.ndarray:
        """获取指定传感器的数据"""
        sensor_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_SENSOR, sensor_name)
        if sensor_id == -1:
            raise ValueError(f"Sensor '{sensor_name}' not found in model.")

        sensor_adr = self.model.sensor_adr[sensor_id]
        sensor_dim = self.model.sensor_dim[sensor_id]
        return self.data.sensordata[sensor_adr : sensor_adr + sensor_dim].copy()

    def get_all_sensor_data(self) -> Dict[str, np.ndarray]:
        """获取所有传感器数据"""
        sensor_dict = {}
        for i in range(self.model.nsensor):
            sensor_name = mujoco.mj_id2name(self.model, mujoco.mjtObj.mjOBJ_SENSOR, i)
            if sensor_name:
                sensor_adr = self.model.sensor_adr[i]
                sensor_dim = self.model.sensor_dim[i]
                sensor_dict[sensor_name] = self.data.sensordata[sensor_adr : sensor_adr + sensor_dim].copy()
        return sensor_dict

    # ==================== 运动学与动力学 ====================

    def compute_jacobian(self) -> Tuple[np.ndarray, np.ndarray]:
        """
        计算末端执行器的雅可比矩阵（site 使用 mj_jacSite，body 使用质心雅可比 mj_jacBodyCom）

        Returns:
            jacp: 线速度雅可比 (3 x nv)
            jacr: 角速度雅可比 (3 x nv)
        """
        jacp = np.zeros((3, self.model.nv))
        jacr = np.zeros((3, self.model.nv))

        if self.ee_type == "site":
            mujoco.mj_jacSite(self.model, self.data, jacp, jacr, self.eef_id)
        else:  # body
            mujoco.mj_jacBodyCom(self.model, self.data, jacp, jacr, self.eef_id)

        return jacp, jacr

    def compute_forward_kinematics(self) -> None:
        """更新正运动学（位置和速度）"""
        mujoco.mj_kinematics(self.model, self.data)

    def compute_inverse_dynamics(self, qacc: np.ndarray) -> np.ndarray:
        """
        计算逆动力学：给定加速度计算所需力矩

        Args:
            qacc: 关节加速度

        Returns:
            qfrc: 所需的广义力
        """
        # 1. 将输入的加速度赋值给 data.qacc
        # 注意：这里使用 [:] 是为了确保数据被复制到底层 C 结构体中，而不是改变 Python 引用
        self.data.qacc[:] = qacc

        # 2. 调用逆动力学函数 (无需传入 qacc，它会自动读取 self.data.qacc)
        mujoco.mj_inverse(self.model, self.data)

        # 3. 返回计算出的广义力
        return self.data.qfrc_inverse.copy()

    def compute_mass_matrix(self) -> np.ndarray:
        """计算质量矩阵 M(q)"""
        M = np.zeros((self.model.nv, self.model.nv))
        mujoco.mj_fullM(self.model, M, self.data.qM)
        return M

    def compute_coriolis_gravity(self) -> np.ndarray:
        """计算科氏力和重力项 C(q,dq) + g(q)"""
        return self.data.qfrc_bias.copy()

    # ==================== 碰撞与限位 ====================

    def check_joint_limits(self) -> Dict[str, bool]:
        """检查关节是否在限位内"""
        limit_status = {}
        for i in range(self.model.njnt):
            jnt_name = mujoco.mj_id2name(self.model, mujoco.mjtObj.mjOBJ_JOINT, i)
            if jnt_name:
                jnt_range = self.model.jnt_range[i]
                if jnt_range[0] < jnt_range[1]:  # 有限位
                    qpos_adr = self.model.jnt_qposadr[i]
                    q = self.data.qpos[qpos_adr]
                    in_limit = jnt_range[0] <= q <= jnt_range[1]
                    limit_status[jnt_name] = in_limit
        return limit_status

    def check_collision(self) -> bool:
        """检查是否存在碰撞"""
        return self.data.ncon > 0

    def get_contact_info(self) -> List[Dict]:
        """获取接触信息"""
        contacts = []
        for i in range(self.data.ncon):
            contact = self.data.contact[i]
            geom1_name = mujoco.mj_id2name(self.model, mujoco.mjtObj.mjOBJ_GEOM, contact.geom1) or "unknown"
            geom2_name = mujoco.mj_id2name(self.model, mujoco.mjtObj.mjOBJ_GEOM, contact.geom2) or "unknown"
            contacts.append(
                {
                    "geom1": geom1_name,
                    "geom2": geom2_name,
                    "pos": contact.pos.copy(),
                    "dist": contact.dist,
                    "frame": contact.frame.copy(),
                }
            )
        return contacts

    # ==================== Mocap 控制 ====================

    def set_mocap_pos(self, mocap_name: str, pos: np.ndarray) -> None:
        """设置 mocap body 的位置"""
        mocap_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, mocap_name)
        if mocap_id == -1:
            raise ValueError(f"Mocap body '{mocap_name}' not found.")
        body_mocapid = self.model.body_mocapid[mocap_id]
        if body_mocapid == -1:
            raise ValueError(f"Body '{mocap_name}' is not a mocap body.")
        self.data.mocap_pos[body_mocapid] = pos

    def set_mocap_quat(self, mocap_name: str, quat: np.ndarray) -> None:
        """设置 mocap body 的姿态（四元数: w,x,y,z）"""
        mocap_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, mocap_name)
        if mocap_id == -1:
            raise ValueError(f"Mocap body '{mocap_name}' not found.")
        body_mocapid = self.model.body_mocapid[mocap_id]
        if body_mocapid == -1:
            raise ValueError(f"Body '{mocap_name}' is not a mocap body.")
        self.data.mocap_quat[body_mocapid] = quat

    # ==================== 视频保存 ====================

    def to_mp4(self) -> None:
        """保存录制的帧为 MP4 视频"""
        if not self.frames:
            warnings.warn("No frames to save. Video not created.")
            return

        # 确保输出目录存在
        directory = os.path.dirname(self.video_path)
        if directory and not os.path.exists(directory):
            try:
                os.makedirs(directory)
                print(f"Created directory: {directory}")
            except Exception as e:
                print(f"Failed to create directory {directory}: {e}")
                return

        # 保存视频
        try:
            imageio.mimsave(self.video_path, self.frames, fps=self.video_fps)
            print(f"Video saved to {self.video_path} ({len(self.frames)} frames)")
        except Exception as e:
            print(f"Failed to save video: {e}")
            # 尝试保存采样帧
            try:
                print("Attempting to save sample frames as images...")
                img_dir = os.path.join(directory or ".", "frames")
                os.makedirs(img_dir, exist_ok=True)
                for i in range(0, len(self.frames), 10):
                    imageio.imwrite(os.path.join(img_dir, f"frame_{i:04d}.png"), self.frames[i])
                print(f"Sample frames saved to {img_dir}")
            except Exception as e2:
                print(f"Failed to save individual frames: {e2}")

    # ==================== 上下文管理器 ====================

    def __enter__(self):
        """上下文管理器入口"""
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """上下文管理器退出，清理资源"""
        self.close()

    def close(self) -> None:
        """关闭环境并保存录制视频"""
        if self.record and self.frames:
            self.to_mp4()

        if self.viewer is not None:
            self.viewer.close()
            self.viewer = None

        if self.renderer is not None:
            self.renderer.close()
            self.renderer = None


# ==================== 测试代码 ====================


def test_mujoco_robot():
    """测试 MujocoRobot 环境类的各项功能"""
    model_path = "./assets/kuka_xml_urdf/iiwa14_dock.xml"

    print("=== Testing MujocoRobot Class ===\n")

    # 使用上下文管理器
    with MujocoRobot(
        model_path=model_path,
        render=True,
        record=False,
        video_path="outputs/test_simulation.mp4",
        ee_name="attachment_site",
        ee_type="site",
        dt=0.002,
    ) as env:
        print(f"Model loaded: {env.model_path.name}")
        print(f"DOF: nq={env.nq}, nv={env.nv}, nu={env.nu}")
        print(f"Sensors: {env.nsensor}")
        print(f"Timestep: {env.dt}s\n")

        # 重置到 home 关键帧（如果存在）
        try:
            env.reset("home")
            print("Reset to 'home' keyframe")
        except ValueError:
            env.reset()
            print("Reset to default state")

        # 仿真循环
        print("\nRunning simulation...")
        for step in range(500):
            q, dq = env.get_joint_states()

            # 简单的重力补偿控制
            tau = np.zeros(env.nu)

            qpos, qvel, eef_pos = env.step(tau)

            if step % 100 == 0:
                print(f"Step {step}: EE pos = {eef_pos}")

        # 测试传感器读取
        print("\n=== Testing Sensor Reading ===")
        if env.nsensor > 0:
            sensor_data = env.get_all_sensor_data()
            for name, data in sensor_data.items():
                print(f"{name}: {data}")
        else:
            print("No sensors in model")

        # 测试雅可比计算
        print("\n=== Testing Jacobian Computation ===")
        jacp, jacr = env.compute_jacobian()
        print(f"Linear Jacobian shape: {jacp.shape}")
        print(f"Angular Jacobian shape: {jacr.shape}")

        # 测试动力学计算
        print("\n=== Testing Dynamics ===")
        M = env.compute_mass_matrix()
        c_g = env.compute_coriolis_gravity()
        print(f"Mass matrix shape: {M.shape}")
        print(f"Coriolis + Gravity vector shape: {c_g.shape}")

        # 测试关节限位检查
        print("\n=== Testing Joint Limits ===")
        limits = env.check_joint_limits()
        for jnt, in_limit in limits.items():
            status = "OK" if in_limit else "VIOLATED"
            print(f"{jnt}: {status}")

        # 测试碰撞检测
        print("\n=== Testing Collision Detection ===")
        has_collision = env.check_collision()
        print(f"Collision detected: {has_collision}")
        if has_collision:
            contacts = env.get_contact_info()
            for i, contact in enumerate(contacts):
                print(f"Contact {i}: {contact['geom1']} <-> {contact['geom2']}")

        # 测试末端执行器状态
        print("\n=== Testing End-Effector State ===")
        ee_pos, ee_vel = env.get_ee_state()
        ee_pos_full, ee_rot = env.get_ee_pose()
        print(f"EE Position: {ee_pos}")
        print(f"EE Velocity: {ee_vel}")
        print(f"EE Rotation matrix shape: {ee_rot.shape}")

    print("\n=== Test Complete ===")


if __name__ == "__main__":
    test_mujoco_robot()
