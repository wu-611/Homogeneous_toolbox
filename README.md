# Homogeneous Toolbox — SE(3) 几何齐次跟踪控制

齐次控制（Homogeneous Control）工具箱及四旋翼 SE(3) 几何齐次跟踪控制器实现。

## 项目概述

本项目基于 Polyakov (2020) 齐次系统理论和 Zhou et al. (2023) SO(3) 姿态齐次控制方法，将 Lee et al. (2010) 的 SE(3) 几何跟踪控制器中的线性 PD 项**升级为齐次控制器**，实现：

- **μ=0**：指数收敛（与几何 PD 等价，但提供统一的齐次框架）
- **μ<0**：有限时间精确收敛（理论保证收敛时间上界）
- **位置回路 + 齐次观测器**：从全状态反馈升级为输出反馈（仅需位置测量，无需速度传感器）

## 目录结构

```
Homogeneous_control/
├── hcs_toolbox_py/           # Python 齐次控制工具箱
│   ├── lpc2hpc.py            # 线性比例控制 → 齐次控制升级
│   ├── lpic2hpic.py          # 线性比例-积分控制 → 齐次控制升级
│   ├── lo2ho.py              # 线性观测器 → 齐次观测器升级（对偶原理）
│   ├── e_hpc.py / e_hpic.py  # HPC/HPIC 控制量在线计算
│   ├── e_ho.py / si_ho.py    # 齐次观测器在线更新（显式/半隐式）
│   ├── hnorm.py              # 齐次范数（二分法）
│   └── tests/                # 验证脚本
├── models/
│   └── quadrotor_se3.py      # 四旋翼 SE(3) 全动力学模型 (RK4)
├── design/
│   ├── design_position_hpc.py    # 位置回路 HPC 设计
│   ├── design_attitude_hpc.py    # 姿态回路 so(3) 齐次设计 (Zhou 2023)
│   ├── attitude_command.py       # 期望姿态解算 (Lee 2010)
│   └── torque_mapping.py         # 虚拟控制 → 力矩映射
├── controllers/
│   └── se3_homogeneous_full.py   # SE(3) 齐次控制器 + Lee PD 对比基线
├── simulation/
│   └── simulator.py              # 通用仿真循环
├── visualization/
│   └── plotter.py                # 多控制器对比可视化
├── scripts/
│   ├── run_phase1_simulations.py # Phase 1 仿真（阶跃+螺旋跟踪）
│   └── test_hover.py             # 悬停调试脚本
├── demo_uav/                     # 已有：三回路 HPC 仿真复现
├── hcs_toolbox_cpp/              # 已有：C++ 工具箱实现
├── docs/                         # 文档
│   ├── Phase0_实现报告.md        # Phase 0 实现报告
│   └── Bug修复记录.md            # Bug 记录与修复
└── 规划文档/                     # 课题规划与理论基础
```

## 快速开始

### 环境要求

- Python 3.8+
- numpy, scipy, matplotlib

```bash
pip install numpy scipy matplotlib
```

### 运行验证

```bash
# 验证齐次观测器移植正确性
python3 hcs_toolbox_py/tests/validate_lo2ho.py
python3 hcs_toolbox_py/tests/validate_e_ho.py

# 验证已有的三回路 HPC 仿真
python3 demo_uav/python/uav_homogeneous_control.py

# 运行 Phase 1 仿真（SE(3) 齐次控制器对比）
python3 scripts/run_phase1_simulations.py
```

### 使用示例

```python
import numpy as np
from models.quadrotor_se3 import QuadrotorSE3
from controllers.se3_homogeneous_full import SE3HomogeneousController

# 四旋翼参数
m, J, g = 1.4, np.diag([0.0211, 0.0219, 0.0366]), 9.81

# 创建模型和控制器
model = QuadrotorSE3(m, J, g)
ctrl = SE3HomogeneousController(m, J, g, mu_p=-0.5, mu_a=-0.5)

# 悬停控制
state = model.make_state(pos=[0,0,0], vel=[0,0,0], R=np.eye(3), omega=[0,0,0])
u = ctrl.compute_control(state, pos_d=[1,0.5,-2], vel_d=[0,0,0], yaw_d=0)
# u = [thrust, tau_x, tau_y, tau_z]
```

## 核心算法

### 齐次控制升级公式

```
u = K₀·x + ||x||_d^(1+μ) · (K-K₀) · d(-ln||x||_d) · x

其中：
  ||x||_d  — 齐次范数（基于膨胀 d(s)=expm(s·Gd)）
  μ        — 齐次度（μ<0:有限时间, μ=0:指数, μ>0:近固定时间）
```

### 姿态 so(3) 齐次控制器 (Zhou 2023)

```
ξ = [θ_e; ω_e]     （指数坐标误差 + 角速度误差）
Gd = [(1-μ)I₃   0 ]
     [   0      I₃]
P  = [  I₃    εI₃ ]
     [ εI₃   K₁⁻¹ ]
u_hom = ||ξ||_d^(1+μ) · [-K₁  -k₂I₃] · d(-ln||ξ||_d) · ξ
M = J·(R_d'·u_hom - ω_d^×·ω + ω̇_d) + ω×J·ω
```

### 齐次观测器升级 (对偶原理)

```
lo2ho(A, C, L) ≡ lpc2hpc(A', C', L') 的转置
ż = A·z + B·u + [L₀ + |Cz-y|^ν·d(log|Cz-y|)·(L-L₀)]·(Cz-y)
```

## 当前进度

- [x] Phase 0: 基础夯实（工具箱补全、验证脚本、仿真框架）
- [x] Phase 1: 全状态反馈 μ=0 版本（位置 HPC + 姿态 so(3) 齐次设计）
- [ ] Phase 2: μ<0 有限时间收敛分析
- [ ] Phase 3a: 输出反馈扩展（齐次观测器）
- [ ] Phase 3b: SITL 实验验证
- [ ] Phase 4: 论文撰写

## 参考文献

1. Polyakov, A. (2020). *Generalized Homogeneity in Systems and Control*. Springer.
2. Zhou, Y., Polyakov, A. & Zheng, G. (2023). Generalized Homogeneous Rigid-Body Attitude Control. *IEEE TAC*.
3. Wang, S. (2020). *Homogeneous Quadrotor Control: Theory and Experiment*. PhD Thesis, Centrale Lille.
4. Lee, T., Leok, M. & McClamroch, N.H. (2010). Geometric tracking control of a quadrotor UAV on SE(3). *CDC*.

## License

MIT License — 仅供学术研究使用。引用请注明来源。
