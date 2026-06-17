# Homogeneous Control Toolbox

齐次控制工具箱及四旋翼无人机控制仿真。

## 项目结构

```
├── hcs_toolbox_py/       # Python 齐次控制工具箱 (22 函数)
│   ├── lpc2hpc.py        #   线性 -> 齐次比例控制器升级
│   ├── lpic2hpic.py      #   线性 -> 齐次比例-积分控制器升级
│   ├── lo2ho.py          #   线性 -> 齐次观测器升级 (对偶原理)
│   ├── e_hpc.py          #   齐次控制器在线计算
│   ├── e_ho.py / si_ho.py #  齐次观测器在线更新 (显式/半隐式)
│   ├── hnorm.py          #   齐次范数 (二分法)
│   ├── hproj.py          #   齐次投影
│   └── tests/            #   验证脚本
├── hcs_toolbox_cpp/      # C++ 工具箱 (header-only)
├── demo_uav/             # 四旋翼三回路 HPC 仿真 (Wang 2020 复现)
│   ├── python/           #   Python 版本
│   └── cpp/              #   C++ 版本 + ROS2 对接
├── HCS_Toolbox_ver02/    # MATLAB 原始工具箱 (Polyakov)
├── docs/                 # 文档
├── figures/              # 仿真结果图
└── 理论笔记/             # 齐次控制理论基础
```

## 已验证可用的方案

**Wang Siyuan (2020) 三回路齐次控制器** (`demo_uav/`):

- Z 回路 (2D): HPIC 高度控制
- Yaw 回路 (2D): HPC 偏航控制
- XY 回路 (8D): HPC 水平位置+姿态控制 (LQR + lpc2hpc)

该方案已在 Quanser QDrone 真机上验证飞行。

## SE(3) 方案说明

曾尝试将 Lee et al. (2010) 的 SE(3) 几何控制器与齐次控制结合，经 PX4 真实化仿真验证后发现**不可行**：

- 根因: 齐次非线性的增益放大机制对电机延迟 (0.02s) 极其敏感
- 表现: 100Hz 控制频率 + 电机延迟 → 位置发散

SE(3) 相关代码已移除。相关理论推导保留在 git 历史 commit `4632d58` ~ `e38b5d7` 中。

## 参考文献

1. Polyakov, A. (2020). *Generalized Homogeneity in Systems and Control*. Springer.
2. Wang, S. (2020). *Homogeneous Quadrotor Control: Theory and Experiment*. PhD Thesis.
3. Lee, T. et al. (2010). Geometric tracking control of a quadrotor UAV on SE(3). *CDC*.
4. Zhou, Y. et al. (2023). Generalized Homogeneous Rigid-Body Attitude Control. *IEEE TAC*.
