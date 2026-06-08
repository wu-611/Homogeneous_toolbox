# Phase 0 实现报告 — SE(3) 几何齐次跟踪控制

> 日期：2026-06-08
> 对应计划文档：`硕士课题详细执行计划.md` Phase 0（月份 1-2）

---

## 目录

1. [项目结构总览](#1-项目结构总览)
2. [T0.2.1 现有工具箱审计](#2-t021-现有工具箱审计)
3. [T0.2.2 观测器函数移植](#3-t022-观测器函数移植)
4. [T0.2.3 验证脚本](#4-t023-验证脚本)
5. [T0.3 仿真框架搭建](#5-t03-仿真框架搭建)
6. [T0.4 关键仿真复现](#6-t04-关键仿真复现)
7. [当前参数配置](#7-当前参数配置)
8. [Phase 1 设计模块](#8-phase-1-设计模块)
9. [仿真结果](#9-仿真结果)
10. [文件清单](#10-文件清单)

---

## 1. 项目结构总览

```
Homogeneous_control/
├── hcs_toolbox_py/           # Python 齐次控制工具箱
│   ├── __init__.py           # 包入口（含 lo2ho, e_ho, si_ho 导出）
│   ├── hnorm.py              # [修改] 齐次范数计算（增加数值保护）
│   ├── lo2ho.py              # [新建] 线性→齐次观测器升级
│   ├── e_ho.py               # [新建] 齐次观测器显式 Euler 更新
│   ├── si_ho.py              # [新建] 齐次观测器半隐式更新
│   └── tests/
│       ├── validate_lo2ho.py # [新建] lo2ho 验证脚本
│       └── validate_e_ho.py  # [新建] e_ho/si_ho 验证脚本
├── models/
│   └── quadrotor_se3.py      # [新建] 四旋翼 SE(3) 全动力学模型
├── design/
│   ├── design_position_hpc.py    # [新建] 位置回路 HPC 设计
│   ├── design_attitude_hpc.py    # [新建] 姿态回路 so(3) 齐次设计
│   ├── attitude_command.py       # [新建] 期望姿态解算
│   └── torque_mapping.py         # [新建] 虚拟控制→力矩映射
├── controllers/
│   └── se3_homogeneous_full.py   # [新建] SE(3) 齐次控制器 + Lee PD 基线
├── simulation/
│   └── simulator.py              # [新建] 通用仿真循环
├── visualization/
│   └── plotter.py                # [新建] 多控制器对比绘图
├── scripts/
│   ├── run_phase1_simulations.py # [新建] Phase 1 仿真脚本
│   └── test_hover.py             # [新建] 悬停调试脚本
├── figures/                      # 仿真结果图片
└── docs/                         # 文档
```

---

## 2. T0.2.1 现有工具箱审计

### 2.1 审计范围

对 `hcs_toolbox_py/` 中所有函数进行了逐文件检查，确认与 MATLAB HCS_Toolbox_ver02 的对应关系：

| Python 文件 | MATLAB 对应 | 功能 | 状态 |
|------------|------------|------|:----:|
| `hnorm.py` | `hnorm.m` | 齐次范数（二分法） | ✓ 已有 |
| `hproj.py` | `hproj.m` | 齐次投影 | ✓ 已有 |
| `lpc2hpc.py` | `lpc2hpc.m` | 线性→齐次比例控制器升级 | ✓ 已有 |
| `lpic2hpic.py` | `lpic2hpic.m` | 线性→齐次比例-积分升级 | ✓ 已有 |
| `e_hpc.py` | `e_hpc.m` | HPC 控制量计算 | ✓ 已有 |
| `e_hpic.py` | `e_hpic.m` | HPIC 控制量计算 | ✓ 已有 |
| `e_fhpc.py` | `e_fhpc.m` | 固定时间 HPC | ✓ 已有 |
| `e_fhpic.py` | `e_fhpic.m` | 固定时间 HPIC | ✓ 已有 |
| `block_con.py` | `block_con.m` | 块可控分解 | ✓ 已有 |
| `trans_con.py` | `trans_con.m` | 正交变换 | ✓ 已有 |
| `ZOH.py` | `ZOH.m` | 零阶保持离散化 | ✓ 已有 |
| `hpc_design.py` | `hpc_design.m` | 直接 HPC 设计 | ✓ 已有 |
| `hpic_design.py` | `hpic_design.m` | 直接 HPIC 设计 | ✓ 已有 |
| `fhpc_design.py` | `fhpc_design.m` | 直接 FHPC 设计 | ✓ 已有 |
| `fhpic_design.py` | `fhpic_design.m` | 直接 FHPIC 设计 | ✓ 已有 |
| `hphi.py` | `hphi.m` | 齐次同胚映射 | ✓ 已有 |
| `hadd.py` | `hadd.m` | 齐次加法 | ✓ 已有 |
| `hdot.py` | `hdot.m` | 齐次标量乘法 | ✓ 已有 |
| `hinner.py` | `hinner.m` | 齐次内积 | ✓ 已有 |
| `lp_norm.py` | `lp_norm.m` | Lp 信号范数 | ✓ 已有 |
| `output_form.py` | `output_form.m` | 输出反馈化简 | ✓ 已有 |
| `lo2ho.py` | `lo2ho.m` | 线性→齐次观测器升级 | **新建** |
| `e_ho.py` | `e_ho.m` | HO 显式 Euler | **新建** |
| `si_ho.py` | `si_ho.m` | HO 半隐式 | **新建** |

### 2.2 审计结论

现有 Python 工具箱**完整覆盖**了 MATLAB 工具箱的核心功能（控制器部分）。缺失的 3 个观测器函数（`lo2ho`, `e_ho`, `si_ho`）已在 T0.2.2 中补全。

---

## 3. T0.2.2 观测器函数移植

### 3.1 lo2ho.py

**数学原理**：齐次观测器升级是对偶于 `lpc2hpc` 的操作。

```
lo2ho(A, C, L) ≡ lpc2hpc(A', C', L') 的转置
```

**核心实现**（仅一行有效计算）：

```python
def lo2ho(A, C, L):
    K0, G0, P, nu_max, nu_min = lpc2hpc(A.T, C.T, L.T)
    return K0.T, -G0.T, P, -nu_min, -nu_max
```

**关键细节**：
- `L0 = K0_ctl'` — 对偶系统的线性增益转置
- `G0_obs = -G0_ctl'` — 膨胀生成元取负（因为观测器处理对偶系统）
- `nu_obs = -mu_ctl` — 齐次度符号反转（观测误差"收缩"，控制器"推动"）

### 3.2 e_ho.py

**数学公式**：

```
z_{k+1} = (I + h*(A + S*C)) * z_k + h*f - h*S*y

其中 S = L0 + expm(ln(|Cz-y|) * (nu*I + Gd - I)) * L
```

**饱和处理**：
- `alpha`（默认 1e-6）：防止输出误差过小导致增益爆炸
- `beta`（默认 inf）：防止输出误差过大导致增益衰减过度

### 3.3 si_ho.py

**与 e_ho 的区别**：

```
e_ho:  z_{k+1} = (I + h*(A + S*C)) * z_k + h*f - h*S*y     （显式）
si_ho: z_{k+1} = (I - h*(A + S*C))^(-1) * (z_k + h*f - h*S*y)  （半隐式）
```

半隐式方法通过矩阵求逆获得更好的数值稳定性，适用于刚性系统（如谐波振荡器）。

---

## 4. T0.2.3 验证脚本

### 4.1 validate_lo2ho.py

**测试模型**：双积分器 `A=[[0,1],[0,0]]`, `C=[[1,0]]`（四旋翼位置通道的正则模型）

**测试项目**：

| # | 测试内容 | 验证方法 | 结果 |
|:-:|---------|---------|:----:|
| 1 | lo2ho 参数有效性 | L0, G0, P 维度正确、nu 范围合理 | ✓ |
| 2 | A+L0*C 幂零性 | 特征值全为零（双积分器，L0=[0,0]） | ✓ |
| 3 | 观测器收敛性 | 初始误差 2.24 → HO 终态 0.0041 | ✓ |
| 4 | HO 收敛速度优势 | HO 达 1% 误差需 0.50s，线性需 2.26s | **4.5x** |
| 5 | G0 结构性质 | trace(G0)=1（双积分器正则值） | ✓ |
| 6 | PGd+Gd'P > 0 | 特征值 [0.30, 2.45] 均为正 | ✓ |
| 7 | 对偶性验证 | L0==K0_ctl', G0==-G0_ctl' | ✓ |

### 4.2 validate_e_ho.py

**测试模型**：谐波振荡器 `A=[[0,1],[-1,0]]`, `C=[[1,0]]`

**测试项目**：

| # | 测试内容 | 验证方法 | 结果 |
|:-:|---------|---------|:----:|
| 1 | e_ho/si_ho 基本运算 | 100 步后无 NaN/Inf | ✓ |
| 2 | 观测误差收敛 | 长时间仿真：误差从 1.0 → 0.001 | **1045x** |
| 3 | 线性退化为 Luenberger | ν=0,α=β=1 时与线性观测器偏差 < 0.004 | ✓ |
| 4 | si_ho 稳定性优势 | 大步长 h=0.05 时 si_ho 仍稳定 | ✓ |

---

## 5. T0.3 仿真框架搭建

### 5.1 四旋翼 SE(3) 动力学模型 (`models/quadrotor_se3.py`)

**状态向量**（18 维）：`[pos(3), vel(3), R(9, flat), omega(3)]`

**动力学方程**：

```
dpos    = vel
dvel    = [0, 0, -g] + (thrust/m) * R * e3
dR      = R * omega^
domega  = J^{-1} * (tau - omega x J*omega)
```

**关键实现细节**：
- RK4 积分，每步用 SVD 将 R 重新投影到 SO(3)（防止数值漂移）
- `exp_so3(theta)` — Rodrigues 公式
- `log_so3(R)` — 含 180° 边界情况的解析对数映射
- `jacobian_r_inv(theta)` — 右 Jacobian 逆，关联角速度与指数坐标速率

### 5.2 仿真器 (`simulation/simulator.py`)

**功能**：
- 通用仿真循环（支持全状态反馈 / 输出反馈）
- 可选的测量噪声（高斯白噪声）
- 参考轨迹日志记录
- 离散时间模型支持

### 5.3 可视化 (`visualization/plotter.py`)

**ResultPlotter 类**：
- `plot_comparison()` — 多控制器对比（3×4 面板：位置误差、速度误差、姿态误差 + 范数对数图）
- `plot_trajectory_3d()` — 3D 轨迹图（含 matplotlib 3D 不可用时的 2D 回退方案）
- `plot_control_inputs()` — 控制输入对比

---

## 6. T0.4 关键仿真复现

### 6.1 demo_lpc2hpc（已有）

位置：`hcs_toolbox_py/demo_lpc2hpc.py`
状态：已有，与 MATLAB 输出一致。

### 6.2 demo_lo2ho（新建 → 整合为验证脚本）

由于 `demo_lo2ho.m` 使用旋转倒立摆模型（QUBE-Servo 2），包含非线性和执行器动力学修正项，直接 1:1 复现风险较高。采用替代方案：
- 在 `validate_lo2ho.py` 中使用双积分器（与四旋翼直接相关）进行全面验证
- 所有数学性质（对偶性、幂零性、收敛性）均得到验证

### 6.3 uav_homogeneous_control（已有）

位置：`demo_uav/python/uav_homogeneous_control.py`
状态：已有，三回路（Z/HPIC、Yaw/HPC、XY/HPC）仿真正常。

### 6.4 Zhou 2023 姿态仿真复现（新建）

位置：`scripts/run_phase1_simulations.py`
状态：已完成 SE(3) 全状态反馈的阶跃响应和轨迹跟踪仿真。

---

## 7. 当前参数配置

### 7.1 四旋翼物理参数

| 参数 | 值 | 单位 |
|------|-----|------|
| m | 1.4 | kg |
| J_xx | 0.0211 | kg·m² |
| J_yy | 0.0219 | kg·m² |
| J_zz | 0.0366 | kg·m² |
| g | 9.81 | m/s² |

### 7.2 位置回路 HPC 参数

| 参数 | 值 | 说明 |
|------|-----|------|
| 期望极点 | [-2, -3] | s² + 5s + 6 = 0 |
| K_linear | [-8.4, -7.0] | k_p=6m, k_d=5m |
| μ_p (指数收敛) | 0.0 | Gd=I（均匀膨胀） |
| μ_p (有限时间) | -0.5 | lpc2hpc 给出的最负容许值 |
| α (饱和下界) | 0.1 | 防近零点过增益 |
| β (饱和上界) | 10.0 | 防远点过衰减 |

### 7.3 姿态回路 HPC 参数

| 参数 | 值 | 说明 |
|------|-----|------|
| K1 | 200·I₃ | 角加速度比例增益 [rad/s² per rad] |
| k2 | 100 | 角加速度阻尼增益 [rad/s² per rad/s] |
| 有效力矩增益 ≈ J·K1 | [4.2, 4.4, 7.3] N·m/rad | 与 Lee PD (kR=4) 可比 |
| μ_a (指数收敛) | 0.0 | 退化为指数坐标几何 PD |
| μ_a (有限时间) | -0.5 | 有限时间收敛 |
| ε（P 矩阵耦合） | ~0.035 | 自动计算：0.5·min(ε_pd, ε_gd) |
| 力矩限幅 | ±20 N·m | 防止数值爆炸 |
| 推力限幅 | [0.1, 80] N | 防止负推力和过大推力 |

### 7.4 Lee PD 对比基线参数

| 参数 | 值 | 说明 |
|------|-----|------|
| kx | 8.4 | 位置比例增益（等效 HPC 的 K_linear[0]/m） |
| kv | 7.0 | 速度阻尼增益 |
| kR | 4.0 | 姿态比例增益 [N·m/rad] |
| komega | 2.0 | 角速度阻尼增益 [N·m/(rad/s)] |

---

## 8. Phase 1 设计模块

### 8.1 位置回路 HPC (`design/design_position_hpc.py`)

**架构**：3 个独立通道，每个是双积分器。

```
每个通道 i ∈ {x, y, z}:
  A = [[0, 1], [0, 0]],  B = [[0], [1/m]]
  线性增益 K → lpc2hpc(A, B, K) → K0, G0, P, μ_min, μ_max
  取 μ ∈ [μ_min, μ_max],  Gd = I + μ·G0
```

**μ=0 特殊性质**：
- `K0 = [0, 0]`（双积分器的 A 矩阵末行已幂零）
- `Gd = I`（均匀膨胀，齐次范数退化为加权欧氏范数）
- 控制律退化为 `u = K·x`（线性反馈）

### 8.2 姿态回路 so(3) 齐次设计 (`design/design_attitude_hpc.py`)

**状态**：`ξ = [θ_e; ω_e] ∈ ℝ⁶`

**关键公式**（Zhou 2023）：

```
膨胀生成元:  Gd = [(1-μ)I₃     0   ]
                   [   0        I₃  ]

形状矩阵:    P  = [  I₃     εI₃  ]
                   [ εI₃   K₁⁻¹  ]

控制律:      u_hom = ||ξ||_d^(1+μ) · K · expm(-ln||ξ||_d · Gd) · ξ
             K = [-K₁  -k₂I₃]
```

**ε 的自动计算**：
```
ε < ε_pd = √(λ_min(K₁⁻¹))              （P ≻ 0 条件）
ε < ε_gd = 2√(1-μ) / ((2-μ)√λ_max(K₁)) （PGd+Gd'P ≻ 0 条件）
ε = 0.5 · min(ε_pd, ε_gd)
```

**μ=0 特殊性质**：
- `Gd = I₆`（均匀膨胀）
- 控制律退化为 `u = -K₁·θ_e - k₂·ω_e`（指数坐标几何 PD）

### 8.3 期望姿态解算 (`design/attitude_command.py`)

```
输入: b3_des（期望推力方向）, ψ_ref（偏航参考）
输出: R_d = [b1_des, b2_des, b3_des]

b1_des = b2_des × b3_des
b2_des = b3_des × b1_ref / |b3_des × b1_ref|
b1_ref = [cos(ψ_ref), sin(ψ_ref), 0]
```

含奇点处理：当 `b3_des` 与 `b1_ref` 平行时，使用替代参考方向。

### 8.4 力矩映射 (`design/torque_mapping.py`)

```
M = J · (R_d' · u_hom - ω_d^× · ω + ω̇_d) + ω × J·ω
```

各项物理含义：
- `J·R_d'·u_hom`：齐次控制从 so(3) → 惯性力矩 → 体坐标系
- `-J·ω_d^×·ω`：期望角速度耦合补偿
- `J·ω̇_d`：前馈期望角加速度
- `ω×J·ω`：陀螺效应补偿

---

## 9. 仿真结果

### 9.1 仿真环境

- 积分方法：RK4, dt = 0.001s
- 仿真时长：阶跃 8s, 螺旋 10s
- 初始状态：原点悬停（pos=0, vel=0, R=I, ω=0）

### 9.2 阶跃悬停结果

目标位置：`[1.0, 0.5, -2.0]` m

| 控制器 | 终态 \|\|e_pos\|\| | ISE [m²·s] | 控制能量 |
|--------|:-----------------:|:----------:|:--------:|
| Lee PD (几何) | **0.049 m** | 3.00 | 39.78 |
| HPC μ=0 (指数收敛) | 0.178 m | 9.35 | 48.35 |
| **HPC μ<0 (有限时间)** | **0.056 m** | **3.43** | **40.02** |

**分析**：
- HPC μ=0 在阶跃响应中表现最差。小误差时与 Lee PD 等价，但大初始误差（1m+）时，姿态耦合导致 HPC 的齐次范数饱和会限制控制量
- HPC μ<0 接近 Lee PD 的性能（ISE 3.43 vs 3.00），且具备**有限时间精确收敛**的理论保证（Lee PD 仅指数收敛）
- Lee PD 最精确，但无齐次性质

### 9.3 Z 轴单独测试

目标位置：`[0, 0, -2]` m（纯 Z 轴，无姿态耦合）

| 控制器 | 5s 后 \|\|e_pos\|\| |
|--------|:-------------------:|
| Lee PD | 4e-7 m |
| HPC μ=0 | 5e-5 m |

Z 轴（无姿态耦合）两者均完美跟踪，验证了位置回路 HPC 本身的正确性。

### 9.4 X 偏移测试

初始：`pos=[0.5, 0, -2]`, 目标：`pos=[0, 0, -2]`

| 控制器 | 3s 后 \|\|e_pos\|\| | 初始 τ_y |
|--------|:-------------------:|:--------:|
| Lee PD | 0.223 m | -6.33 N·m |
| HPC μ=0 | **0.030 m** | -1.55 N·m |

HPC μ=0 的力矩更小（1.55 vs 6.33 N·m），但位置收敛更快（因为位置增益更高）。这说明姿态和位置增益之间存在复杂的耦合关系。

### 9.5 已知问题

1. **螺旋轨迹跟踪 RMS 偏大**（~25m）— 需要更好的轨迹初始化和增益调优
2. **姿态 HPC 增益 K1=200 的 P 矩阵不平衡** — K₁⁻¹=0.005 与 I 相差 200 倍，可能影响齐次范数的数值精度
3. **hnorm 数值溢出** — 大状态向量时 `expm(-Gd*s)` 可能溢出，已添加 try/except 保护

---

## 10. 文件清单

### 10.1 新建文件（17 个）

| 文件 | 行数 | 功能 |
|------|:---:|------|
| `hcs_toolbox_py/lo2ho.py` | 36 | 线性→齐次观测器升级 |
| `hcs_toolbox_py/e_ho.py` | 69 | HO 显式 Euler |
| `hcs_toolbox_py/si_ho.py` | 69 | HO 半隐式 |
| `hcs_toolbox_py/tests/validate_lo2ho.py` | 137 | lo2ho 验证 |
| `hcs_toolbox_py/tests/validate_e_ho.py` | 136 | e_ho/si_ho 验证 |
| `models/quadrotor_se3.py` | 186 | SE(3) 动力学 |
| `simulation/simulator.py` | 132 | 仿真循环 |
| `visualization/plotter.py` | 220 | 可视化 |
| `design/design_position_hpc.py` | 159 | 位置 HPC |
| `design/design_attitude_hpc.py` | 201 | 姿态 HPC |
| `design/attitude_command.py` | 125 | 期望姿态 |
| `design/torque_mapping.py` | 104 | 力矩映射 |
| `controllers/se3_homogeneous_full.py` | 235 | SE(3) 控制器 |
| `scripts/run_phase1_simulations.py` | 195 | Phase 1 仿真 |
| `scripts/test_hover.py` | 67 | 悬停调试 |
| `docs/Phase0_实现报告.md` | — | 本文档 |
| `docs/Bug修复记录.md` | — | Bug 记录 |

### 10.2 修改文件（2 个）

| 文件 | 修改内容 |
|------|---------|
| `hcs_toolbox_py/__init__.py` | 添加 `lo2ho`, `e_ho`, `si_ho` 的导出和文档字符串 |
| `hcs_toolbox_py/hnorm.py` | 添加数值溢出保护、NaN/Inf 检测 |
