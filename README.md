# Homogeneous Toolbox — SE(3) 几何齐次跟踪控制

齐次控制（Homogeneous Control）工具箱及四旋翼 SE(3) 几何齐次跟踪控制器实现。

## 项目概述

基于 Polyakov (2020) 齐次系统理论和 Zhou et al. (2023) SO(3) 姿态齐次控制方法，将 Lee et al. (2010) 的 SE(3) 几何跟踪控制器中的线性 PD 项**升级为齐次控制器**。

**两种运行模式**：
- **全状态反馈**（`se3_homogeneous_full.py`）：需要位置 + 速度 + 姿态 + 角速度
- **输出反馈**（`se3_homogeneous_outfb.py`）：仅需位置 + 姿态 + 角速度（速度由齐次观测器估计）

**齐次度选择**：
- μ=0：指数收敛（退化为几何 PD）
- μ<0：有限时间精确收敛（μ=-0.5 推荐）

## 目录结构

```
Homogeneous_control/
├── hcs_toolbox_py/             # Python 齐次控制工具箱 (22 函数)
│   ├── lpc2hpc.py / lpic2hpic.py  # 线性→齐次控制器升级
│   ├── lo2ho.py                   # 线性→齐次观测器升级 (对偶原理)
│   ├── e_hpc.py / e_hpic.py       # 控制量在线计算
│   ├── e_ho.py / si_ho.py         # 观测器在线更新 (显式/半隐式)
│   ├── hnorm.py                   # 齐次范数 (二分法)
│   └── tests/                     # 验证脚本
├── models/
│   └── quadrotor_se3.py           # 四旋翼 SE(3) 全动力学 (RK4 + SVD重投影)
├── design/
│   ├── design_position_hpc.py     # 位置回路 HPC 设计 (双积分器)
│   ├── design_position_ho.py      # 位置回路 HO 设计 (仅需位置测量)
│   ├── design_attitude_hpc.py     # 姿态回路 so(3) 齐次设计 (Zhou 2023)
│   ├── attitude_command.py        # 期望姿态解算 (Lee 2010)
│   └── torque_mapping.py          # 虚拟控制→力矩映射 (逆动力学)
├── controllers/
│   ├── se3_homogeneous_full.py    # 全状态反馈 SE(3) 齐次控制器
│   └── se3_homogeneous_outfb.py   # 输出反馈 SE(3) 齐次控制器 (Phase 3a)
├── simulation/
│   └── simulator.py               # 通用仿真循环 (支持噪声注入)
├── visualization/
│   └── plotter.py                 # 多控制器对比绘图
├── scripts/
│   ├── se3_controller_interface.py    # ROS2 SITL 对接接口 (独立模块)
│   ├── run_phase1_simulations.py      # Phase 1/2 仿真 (对比 Lee PD)
│   ├── run_phase3a_simulations.py     # Phase 3a 仿真 (输出反馈验证)
│   ├── compute_rho_tilde.py           # Lyapunov 衰减率估计
│   └── test_hover.py                  # 悬停调试
├── config/
│   └── se3_hpc_params.yaml           # ROS2 SITL 参数配置
├── docs/
│   ├── Phase0_实现报告.md             # Phase 0 实现报告
│   ├── Phase2_有限时间收敛理论推导.md  # μ<0 理论证明 + 数值验证
│   ├── Phase3a_输出反馈理论推导.md     # 输出反馈 ISS 分析
│   └── Bug修复记录.md                 # Bug 记录
├── hcs_toolbox_cpp/                   # C++ 工具箱 (header-only, 16 头文件)
├── demo_uav/                          # 线性化 8D HPC 仿真 (Wang 2020 复现)
└── 规划文档/                          # 课题规划与理论基础
```

## 快速开始

### 环境要求

```bash
pip install numpy scipy matplotlib
```

### 运行验证

```bash
# 工具箱验证
python3 hcs_toolbox_py/tests/validate_lo2ho.py
python3 hcs_toolbox_py/tests/validate_e_ho.py

# SE(3) 全状态反馈仿真 (对比 Lee PD)
python3 scripts/run_phase1_simulations.py

# Lyapunov 衰减率估计
python3 scripts/compute_rho_tilde.py

# 输出反馈仿真
python3 scripts/run_phase3a_simulations.py

# ROS2 对接接口独立测试
python3 scripts/se3_controller_interface.py
```

### 使用示例

```python
import numpy as np
from models.quadrotor_se3 import QuadrotorSE3
from controllers.se3_homogeneous_outfb import SE3HomogeneousOutFB

# 四旋翼参数
m = 1.4
J = np.diag([0.0211, 0.0219, 0.0366])

# 创建输出反馈控制器 (仅需位置测量, 速度由 HO 估计)
ctrl = SE3HomogeneousOutFB(m=m, J=J, mu_p=-0.5, mu_a=-0.5, dt=0.01)
ctrl.reset(pos_measured=[0,0,0], pos_d=[1,0.5,-2])

# 每步控制
for step in range(1000):
    u = ctrl.compute_control(state, pos_d=[1,0.5,-2], vel_d=[0,0,0], yaw_d=0)
    # u = [thrust, tau_x, tau_y, tau_z]
    state = model.step_rk4(state, u, dt=0.001)
```

## ROS2 + PX4 SITL 对接

### 接口文件

`scripts/se3_controller_interface.py` — 独立模块，可在 ROS2 节点中直接导入。

### 坐标系约定

| 框架 | 坐标系 | Z 轴方向 |
|------|--------|---------|
| 算法内部 | Z-Up | 垂直向上 |
| PX4 状态 | NED | 垂直向下（地） |
| PX4 力矩 | FRD | 机体前-右-下 |

**坐标转换在 ROS2 节点中完成**，接口模块只处理 Z-Up。

### ROS2 节点伪代码

```python
from se3_controller_interface import (
    SE3ControllerInterface,
    ned_to_zup, zup_to_ned, quat_to_rot_matrix
)

# 初始化
ctrl = SE3ControllerInterface(mode='outfb', mu_p=-0.5, mu_a=-0.5)
ctrl.set_target(pos_d=[0, 0, -2], yaw_d=0)

# PX4 状态回调
def odometry_callback(msg):
    # NED → Z-Up
    pos_zup, vel_zup = ned_to_zup(
        [msg.position[0], msg.position[1], msg.position[2]],
        [msg.velocity[0], msg.velocity[1], msg.velocity[2]]
    )
    # 四元数 → 旋转矩阵
    R = quat_to_rot_matrix(msg.q[0], msg.q[1], msg.q[2], msg.q[3])
    # FRD → Z-Up 角速度 (X不变, Y不变, Z反转)
    omega_zup = [msg.angular_velocity[0],
                 msg.angular_velocity[1],
                 -msg.angular_velocity[2]]

    # 控制计算
    u = ctrl.step(pos_zup, vel_zup, R, omega_zup)

    # Z-Up → NED/FRD
    thrust_ned, tau_frd = zup_to_ned(u[0], u[1:4])

    # 发布到 PX4
    publish_control(thrust_ned, tau_frd)
```

### 关键转换公式

| 转换 | 公式 |
|------|------|
| NED→Z-Up 位置 | `[N, E, -D]` |
| NED→Z-Up 速度 | `[vN, vE, -vD]` |
| Z-Up→NED 推力 | `-thrust_zup` |
| Z-Up→FRD 力矩 | `[τx, τy, -τz]` |

### 参数配置

编辑 `config/se3_hpc_params.yaml` 调整控制器参数。关键参数：

- `mu_p = -0.5`：位置回路齐次度（有限时间收敛）
- `mu_a = -0.5`：姿态回路齐次度
- `K1 = 200`：姿态比例增益 → 有效力矩 J·K1 ≈ 4 N·m/rad
- `max_tilt_deg = 60`：推力倾角限制

## 核心算法

### 齐次控制升级公式

```
u = K₀·x + ||x||_d^(1+μ) · (K-K₀) · d(-ln||x||_d) · x

||x||_d — 齐次范数（基于膨胀 d(s)=expm(s·Gd)）
μ       — 齐次度（μ<0:有限时间, μ=0:指数, μ>0:近固定时间）
```

### 姿态 so(3) 齐次控制器 (Zhou 2023)

```
ξ = [θ_e; ω_e]
Gd = [(1-μ)I₃   0 ]
     [   0      I₃]
u_hom = ||ξ||_d^(1+μ) · [-K₁  -k₂I₃] · d(-ln||ξ||_d) · ξ
M = J·(R_d'·u_hom - ω_d^×·ω + ω̇_d) + ω×J·ω
```

### 齐次观测器 (对偶原理)

```
lo2ho(A, C, L) ≡ lpc2hpc(A', C', L') 的转置
ż = A·z + B·u + [L₀ + |Cz-y|^ν·d(log|Cz-y|)·(L-L₀)]·(Cz-y)
```

## 当前进度

| Phase | 内容 | 状态 |
|:-----:|------|:----:|
| 0 | 基础夯实（工具箱+仿真框架） | ✅ |
| 1 | 全状态反馈 μ=0 版本 | ✅ |
| 2 | μ<0 有限时间收敛分析 | ✅ ρ̃_p=0.66, ρ̃_a=1.80 |
| 3a | 输出反馈（齐次观测器） | ✅ ISE退化<11% @ σ=0.1m |
| 3b | SITL 实验验证 | 🔲 对接文件已就绪 |
| 4 | 论文撰写 | 🔲 |

## 关键数值结果

| 指标 | 数值 | 说明 |
|------|------|------|
| ρ̃_p | 0.655 (R²=0.986) | 位置回路 Lyapunov 衰减率 |
| ρ̃_a | 1.802 (R²=0.987) | 姿态回路 Lyapunov 衰减率 |
| T_p 上界 | 4.40 s | 纯 Z 轴悬停收敛时间 |
| T_a 上界 | 0.90 s | 30° 姿态误差收敛时间 |
| T_total 上界 | 5.29 s | 级联阶跃响应 |
| 输出反馈 ISE | 3.23 vs 3.43 (全状态) | 观测器滤波效应改善性能 |
| 噪声退化 | +11% @ σ=0.1m | GPS 级噪声下几乎无退化 |

## 参考文献

1. Polyakov, A. (2020). *Generalized Homogeneity in Systems and Control*. Springer.
2. Zhou, Y., Polyakov, A. & Zheng, G. (2023). Generalized Homogeneous Rigid-Body Attitude Control. *IEEE TAC*.
3. Wang, S. (2020). *Homogeneous Quadrotor Control: Theory and Experiment*. PhD Thesis, Centrale Lille.
4. Lee, T., Leok, M. & McClamroch, N.H. (2010). Geometric tracking control of a quadrotor UAV on SE(3). *CDC*.

## License

MIT License — 仅供学术研究使用。引用请注明来源。
