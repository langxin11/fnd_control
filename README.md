<!--
 * @Author: langxin11
 * @Date: 2026-01-01 16:27:04
 * @LastEditors: Your Name
 * @LastEditTime: 2026-01-01 18:29:53
 * @Description: readme文件
 -->
# Robot Learning Framework

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)

本项目是一个机器人学习与任务空间控制框架，结合 MuJoCo 物理引擎和 Pinocchio 动力学库，提供机器人仿真、轨迹规划与控制的完整流程。

## 快速开始

- 运行统一入口（可选 Pinocchio 控制器）
  - `python main.py --use-pinocchio`
- 仅 MuJoCo 控制器
  - `python scripts/main_mujoco.py`
- 仅 Pinocchio 控制器
  - `python scripts/main_pinocchio.py`

常用参数：
- `--model-path` MuJoCo 模型 XML
- `--urdf-path` Pinocchio URDF
- `--pin-ee-frame` Pinocchio 末端 frame
- `--duration`/`--dt` 仿真时长与时间步
- `--render`/`--no-render`


## 目录结构

```plaintext
project/
├── assets/
│   └── kuka_xml_urdf/
│       ├── iiwa14_dock.xml
│       ├── iiwa14_dock.urdf
│       ├── urdf/
│       │   └── iiwa14_dock_relative.urdf
│       └── meshes/
├── main.py
├── scripts/
│   ├── main_mujoco.py
│   └── main_pinocchio.py
├── src/
│   └── robot_learning/
│       ├── controllers/
│       │   ├── MujocoTaskSpaceController.py
│       │   └── PinocchioTaskSpaceController.py
│       ├── robots/
│       │   ├── mujoco_robot.py
│       │   ├── pinocchio_robot.py
│       │   └── hybrid_robot.py
│       ├── traj/
│       │   └── Trajectory.py
│       └── utils/
│           ├── ik.py
│           └── logger.py
└── tests/
    ├── test.py
    ├── test_robot_interface.py
    └── viewer.py
```
