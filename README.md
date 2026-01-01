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

本项目是一个机器人学习框架，结合 MuJoCo 物理引擎和 Pinocchio 动力学库，为机器人仿真与控制提供完简单方案。

目录结构
```plaintext
project/
├── pyproject.toml
├── README.md
├── LICENSE
├── .gitignore
├── src/
│   └── robot_learning/                 # 顶层包名（命名空间）
│       ├── __init__.py
│       ├── _version.py                 # 可选：版本号集中管理
│       │
│       ├── robots/                     # 后端与机器人封装
│       │   ├── __init__.py
│       │   ├── mujoco_robot.py
│       │   ├── pinocchio_robot.py
│       │   └── hybrid_robot.py
│       │
│       ├── tasks/                      # 任务定义与拼接
│       │   ├── __init__.py
│       │   ├── task_base.py
│       │   └── stack.py
│       │
│       ├── controllers/                # 控制算法（可插拔）
│       │   ├── __init__.py
│       │   ├── controller_base.py
│       │   ├── joint_pd.py
│       │   └── fnd.py
│       │
│       ├── utils/                      # 数学与通用工具
│       │   ├── __init__.py
│       │   └── linalg.py
│       │
│       ├── configs/                    # 可选：默认配置（包内资源）
│       │   └── default.yaml
│       │
│       └── assets/                     # 可选：少量必要模型资源（谨慎放大文件）
│           └── robots/...
│
├── scripts/                             # 运行入口（不作为包导入）
│   └── run_tracking.py
└── tests/
    ├── test_imports.py
    └── test_task_stack.py


```