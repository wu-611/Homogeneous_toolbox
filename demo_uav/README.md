# 四旋翼无人机三回路齐次控制器 — 复现与仿真

基于王思远 (Wang Siyuan) 2020 年博士论文（Centrale Lille），将线性 PID 控制器升级为齐次控制器 (HPC/HPIC)，
应用于四旋翼无人机的位置与姿态控制。

---

## 目录

1. [文件说明](#1-文件说明)
2. [理论背景](#2-理论背景)
3. [三回路架构](#3-三回路架构)
4. [数值仿真 (Python/C++)](#4-数值仿真-pythonc)
5. [ROS2 + PX4 SITL 部署](#5-ros2--px4-sitl-部署)
6. [仿真结果](#6-仿真结果)
7. [坐标系约定 (关键)](#7-坐标系约定-关键)
8. [参数对照表](#8-参数对照表)

---

## 1. 文件说明

```
Homogeneous_control/
├── hcs_toolbox_py/                     # Python 齐次控制工具箱 (22 个函数)
├── hcs_toolbox_cpp/                    # C++ 齐次控制工具箱 (header-only, 16 个头文件)
│   ├── include/hcs_toolbox/            #   核心函数
│   ├── src/demo_*.cpp                  #   数值仿真 Demo
│   └── scripts/plot_*.py               #   绘图脚本
├── demo_uav/
│   ├── README.md                       # 本文档
│   ├── python/
│   │   ├── uav_homogeneous_control.py  # Python 三回路仿真
│   │   └── uav_homogeneous_step_response.png
│   ├── cpp/
│   │   └── README.md                   # C++ 仿真说明
│   ├── Compute_All_HPC_Params.m        # MATLAB 参数计算脚本 (原始参考)
│   ├── Wang_Siyuan_DLE.pdf             # 王思远博士论文
│   ├── function.txt                    # Simulink 工程实现代码 (含坐标约定)
│   └── 四旋翼齐次控制器复现与开发规范...md # 开发规范文档
└── px4_nmpc_ws/uav_nmpc/               # ROS2 节点 (PX4 SITL 部署)
    ├── include/px4nmpc_learn/
    │   ├── hcs_coordinate_converter.hpp # NED↔Z-Up 坐标转换
    │   └── hcs_uav_controller.hpp       # 三回路 HPC 控制器类
    ├── src/
    │   ├── hcs_uav_controller.cpp       # 控制器实现
    │   └── hcs_node.cpp                 # ROS2 节点
    ├── launch/hcs_offboard.launch.py    # SITL 启动文件
    └── config/hcs_params.yaml           # 控制器参数
```

---

## 2. 理论背景

### 齐次控制核心思想

将已有的线性控制器 `u = K·x` 升级为非线性齐次控制器：

```
u = K0·x + ||x||_d^(1+μ) · (K-K0) · d(-ln||x||_d) · x
```

其中：
- `K0`：齐次化反馈增益（线性分量）
- `K-K0`：非线性增益
- `μ`：齐次度（μ < 0 → 有限时间收敛，μ > 0 → 近似固定时间收敛）
- `d(s) = expm(s·Gd)`：膨胀算子（dilation operator）
- `||x||_d`：典范齐次范数

### 升级流程

```
线性控制器 K       ──lpc2hpc()──→  齐次控制器 {K0, G0, P, μ}
线性PI控制器 K,Ki  ──lpic2hpic()─→  齐次PI控制器 {K0, G0, P, Ki_new, μ}

运行时: u = e_hpc(x, K0, K-K0, Gd, μ, hnorm, α, β)
       u = uh + ∫ui dt  (HPIC — 带积分)
```

### 为什么比线性 PID 更好

1. **无 peaking 效应**：线性高增益控制器在快速响应时会产生过冲，齐次控制器通过非线性缩放消除此问题
2. **有限时间收敛**（μ < 0）：状态在有限时间内精确到达原点，而非渐近逼近
3. **鲁棒性**：对匹配/非匹配扰动具有更好的抑制能力
4. **性能下限保证**：通过 α/β 饱和参数，确保在任何情况下性能不低于原线性控制器

---

## 3. 三回路架构

四旋翼线性化模型可解耦为三个独立子系统：

```
┌─────────────────────────────────────────────────────────┐
│                   Z 回路 (HPIC)                          │
│  状态: [z_error, vz]           输入: 净推力 Fz           │
│  模型: 双积分器 (m=1.07kg)                              │
│  μ = mu_min (最负容许值, 有限时间收敛)                    │
│  线性增益: K=[-5, -2], Ki=[-0.1, 0]                    │
├─────────────────────────────────────────────────────────┤
│                  Yaw 回路 (HPC)                          │
│  状态: [ψ_error, ωz]           输入: 偏航力矩 τz          │
│  模型: 双积分器 (Izz=0.0129)                            │
│  μ = -0.5 (有限时间收敛)                                │
│  线性增益: K=[-0.39, -0.21]                              │
├─────────────────────────────────────────────────────────┤
│                   XY 回路 (HPC)                          │
│  状态: [x,y, vx,vy, θ,-φ, q,-p]  输入: [τy, τx_virtual] │
│  模型: 8阶级联系统 (Ixx=6.85e-3, Iyy=6.62e-3)           │
│  μ = -1.0 (有限时间收敛, hn⁰=1)                          │
│  线性增益: LQR 设计 (Q位置=10, R力矩=1)                  │
│  关键: -φ 使A矩阵对称, 输出τ_roll需取反恢复              │
└─────────────────────────────────────────────────────────┘
```

### XY 回路 8 阶状态空间

```
状态向量: ξ = [x, y, ẋ, ẏ, θ, -φ, q, -p]ᵀ

A_xy = [0  0  I  0  0  0  0  0]      B_xy = [0      0  ]
       [0  0  0  I  0  0  0  0]             [0      0  ]
       [0  0  0  0  g  0  0  0]             [0      0  ]
       [0  0  0  0  0  g  0  0]             [0      0  ]
       [0  0  0  0  0  0  I  0]             [0      0  ]
       [0  0  0  0  0  0  0  I]             [0      0  ]
       [0  0  0  0  0  0  0  0]             [1/Iyy  0  ]  ← 俯仰力矩
       [0  0  0  0  0  0  0  0]             [0   1/Ixx]  ← 滚转力矩(虚拟)

ẋ=v, v̇=g·θ, θ̇=q, q̇=τ_pitch/Iyy   (俯仰通道)
ẏ=v, v̇=g·(-φ), φ̇=p, ṗ=τ_roll/Ixx  (滚转通道, -φ使A对称)
```

---

## 4. 数值仿真 (Python/C++)

### 4.1 Python 数值仿真

```bash
cd /home/wzy/Homogeneous_control
pip install numpy scipy matplotlib   # 依赖

# 运行仿真
python3 demo_uav/python/uav_homogeneous_control.py

# 查看图片 (WSL2)
explorer.exe demo_uav/python/uav_homogeneous_step_response.png
```

### 4.2 C++ 数值仿真

```bash
cd /home/wzy/Homogeneous_control/hcs_toolbox_cpp

# 构建
mkdir -p build && cd build
cmake .. && make -j$(nproc)

# 运行仿真 → 生成 CSV
./bin/demo_uav_homogeneous_control

# 绘图
python3 ../scripts/plot_uav_comparison.py

# 查看图片
explorer.exe ../scripts/uav_comparison_cpp.png
```

输出三个 CSV 文件 (`uav_{z,yaw,xy}_comparison_cpp.csv`) 和一个 3×3 对比图表。

---

## 5. ROS2 + PX4 SITL 部署

### 5.1 前置依赖

| 组件 | 版本要求 | 作用 |
|------|---------|------|
| Ubuntu | 22.04 (推荐) | 操作系统 |
| ROS 2 | Humble | 机器人中间件 |
| PX4-Autopilot | v1.14+ | 飞控固件 (SITL 模式) |
| px4_msgs | 匹配 PX4 版本 | PX4 ↔ ROS2 消息定义 |
| Micro XRCE-DDS Agent | 最新 | PX4 ↔ ROS2 通信桥 |
| Eigen3 | 3.4+ | 线性代数 (apt install libeigen3-dev) |

### 5.2 环境安装 (完整流程)

#### 步骤 1: 安装 ROS 2 Humble

```bash
# 添加 ROS 2 仓库
sudo apt update && sudo apt install curl gnupg lsb-release
sudo curl -sSL https://raw.githubusercontent.com/ros/rosdistro/master/ros.key \
  -o /usr/share/keyrings/ros-archive-keyring.gpg
echo "deb [arch=$(dpkg --print-architecture) signed-by=/usr/share/keyrings/ros-archive-keyring.gpg] \
  http://packages.ros.org/ros2/ubuntu $(lsb_release -cs) main" \
  | sudo tee /etc/apt/sources.list.d/ros2.list > /dev/null

sudo apt update
sudo apt install ros-humble-desktop python3-colcon-common-extensions
sudo apt install ros-humble-px4-msgs   # PX4 ROS2 消息包
```

#### 步骤 2: 安装 PX4 SITL

```bash
cd ~
git clone https://github.com/PX4/PX4-Autopilot.git --branch v1.14.3 --depth 1
cd PX4-Autopilot
git submodule update --init --recursive

# 安装 PX4 构建依赖
bash ./Tools/setup/ubuntu.sh

# 验证 SITL 构建
make px4_sitl gazebo-classic
# 看到 pxh> 提示符表示成功, Ctrl+C 退出
```

#### 步骤 3: 安装 Micro XRCE-DDS Agent (PX4-ROS2 桥)

```bash
# 方式1: 从源码构建
cd ~
git clone https://github.com/eProsima/Micro-XRCE-DDS-Agent.git --depth 1
cd Micro-XRCE-DDS-Agent
mkdir build && cd build
cmake .. && make -j$(nproc)
sudo make install
sudo ldconfig

# 方式2: 使用 snap (Ubuntu 22.04)
sudo snap install micro-xrce-dds-agent
```

#### 步骤 4: 配置 ROS2 工作空间

```bash
# 创建工作空间
mkdir -p ~/px4_nmpc_ws/src
cd ~/px4_nmpc_ws/src

# 克隆或复制本项目代码
# (假设代码在 /home/wzy/Homogeneous_control)
ln -s /home/wzy/px4_nmpc_ws/uav_nmpc .

cd ~/px4_nmpc_ws

# 安装 Eigen3
sudo apt install libeigen3-dev

# 构建
colcon build --packages-select px4nmpc_learn --symlink-install

# 加载环境
source install/setup.bash
```

### 5.3 启动顺序 (SITL 完整流程)

终端按顺序打开 **4 个窗口**：

```bash
# ====== 窗口 1: PX4 SITL ======
cd ~/PX4-Autopilot
make px4_sitl gazebo-classic
# 等待出现 pxh> 提示符

# ====== 窗口 2: Micro XRCE-DDS Agent ======
MicroXRCEAgent udp4 -p 8888
# 看到 "client connected" 表示 PX4 已连接

# ====== 窗口 3: ROS2 工作空间 ======
cd ~/px4_nmpc_ws
source install/setup.bash

# 验证 PX4 消息可达
ros2 topic list | grep fmu
# 应看到 /fmu/out/vehicle_odometry 等话题

# ====== 窗口 4: 齐次控制器 (在确认话题可用后) ======
cd ~/px4_nmpc_ws
source install/setup.bash

# 启动 HCS 节点
ros2 launch px4nmpc_learn hcs_offboard.launch.py

# 自定义目标位置 (例如移动到 (1, 0.5, -2) NED):
ros2 launch px4nmpc_learn hcs_offboard.launch.py \
  target_x:=1.0 target_y:=0.5 target_z:=-2.0 target_yaw:=0.0
```

### 5.4 解锁与飞行

在 QGroundControl (或通过 MAVLink 命令) 中:

```
1. 等待 GPS/定位就绪 (SITL 自动就绪)
2. 切换到 Offboard 模式
3. 解锁 (Arm)
4. 无人机自动飞向目标位置并悬停
```

### 5.5 监控与调试

```bash
# 观察 PX4 状态
ros2 topic echo /fmu/out/vehicle_odometry --field position

# 观察控制器输出
ros2 topic echo /fmu/in/vehicle_thrust_setpoint
ros2 topic echo /fmu/in/vehicle_torque_setpoint

# 观察 HCS 节点日志
ros2 topic echo /rosout | grep hcs
```

### 5.6 SITL 测试用例

| 测试 | 命令 | 验证标准 |
|------|------|---------|
| 悬停 | `target_x:=0 target_y:=0 target_z:=-2.0` | 高度稳定在 2m, 漂移 < 0.2m |
| X阶跃 | `target_x:=1.0 target_y:=0 target_z:=-2.0` | 无超调, 收敛 < 3s |
| Y阶跃 | `target_x:=0 target_y:=1.0 target_z:=-2.0` | 无超调, 收敛 < 3s |
| 偏航 | `target_yaw:=1.57` | 平滑旋转 90°, 无高度变化 |
| 抗扰动 | 悬停中 Gazebo 加风力 | 恢复时间 < 2s |

---

## 6. 仿真结果

### 数值仿真: 阶跃响应对比（线性 vs 齐次）

| 回路 | 初始误差 | 线性 ||x(T)|| | 齐次 ||x(T)|| | 提升 |
|------|---------|-------------------|-------------------|------|
| **Z** (HPIC) | e_z = 1m | 0.044694 | **0.004325** | **10.3×** |
| **Yaw** (HPC) | e_ψ = 0.5rad | 0.000002 | **0.000000** | — |
| **XY** (HPC) | e_x=1m, e_y=0.5m | 0.000032 | **0.000000** | **有限时间收敛** |

### 关键发现

1. **Z 轴**：HPIC（齐次比例-积分）比线性 PI 精度高 **10 倍**。齐次积分项 `nx^(1+2μ)·Ki_new·px` 提供了更强的扰动抑制。

2. **XY 轴**：HPC 在 μ=-1 时实现了**有限时间收敛**——状态在 ~4s 内精确到达机器零 (~1e-31)，而线性 LQR 仅渐近收敛至 3.2e-5。

3. **Yaw 轴**：两者接近。2 阶系统本身容易控制，增益设计得当的线性 PD 也能很好收敛。

### 生成的图表

- `python/uav_homogeneous_step_response.png` — 3×3 面板 (线性 vs HPC 对比, 对数坐标)
- `hcs_toolbox_cpp/scripts/uav_comparison_cpp.png` — C++ 版本 3×3 对比

---

## 7. 坐标系约定 (关键)

这是 ROS2 部署中最容易出错的环节。详细分析见 `hcs_coordinate_converter.hpp` 注释。

### 三个坐标系

| 坐标系 | X | Y | Z | 用途 |
|--------|---|---|---|------|
| **NED** (世界) | 北 | 东 | **地 ↓** | PX4 位置/速度 |
| **FRD** (机体) | 前 | 右 | **下** | PX4 角速度 |
| **Z-Up** (论文) | 北 | 东 | **天 ↑** | 论文状态方程 |

### 核心转换

```
NED → Z-Up (位置):     z_zup  = -z_ned,   vz_zup = -vz_ned
NED → Z-Up (水平):     x, y 不变 (北/东方向一致)
Z-Up → NED (推力):     thrust_ned_z = -Fz_zup  (向上推力→NED向下)
```

### XY 回路符号约定 (最关键)

NED/FRD 中姿态角的加速度符号不对称：
- `+θ` (机头下压) → `ax_NED < 0` (向南) — **负**
- `+φ` (向右滚转) → `ay_NED > 0` (向东) — **正**

论文解决方案：状态向量用 `[-φ, -p]` 使 A 矩阵对称 (+g, +g)。
代价：控制器输出的滚转力矩需取反恢复 → `τ_roll_physical = -τ_roll_virtual`

```
构建状态:  ξ_xy = [e_x, e_y, e_vx, e_vy, θ, -φ, q, -p]
控制输出:  τ_pitch = u(0)           ← 不取反
           τ_roll  = -u(1)          ← 必须取反! (来源: function.txt L72)
```

---

## 8. 参数对照表

### 无人机物理参数

| 参数 | 数值仿真值 | SITL 部署值 | 单位 |
|------|----------|-----------|------|
| 质量 m | 1.4 | **1.07** | kg |
| Ixx (滚转) | 0.0211 | **0.00685** | kg·m² |
| Iyy (俯仰) | 0.0219 | **0.00662** | kg·m² |
| Izz (偏航) | 0.0366 | **0.0129** | kg·m² |
| 重力 g | 9.8 | 9.81 | m/s² |

> **注意**: 数值仿真和 SITL 部署使用不同参数! `Compute_All_HPC_Params.m` 使用仿真值,
> 而 `hcs_params.yaml` 和 `四旋翼齐次控制器复现与开发规范...md` 使用 SITL 部署值。

### Z 回路 (HPIC)

| 参数 | 值 |
|------|-----|
| A_z | [0 1; 0 0] |
| B_z | [0; 1/m] |
| 线性比例 K | [-5, -2] |
| 线性积分 Ki | [-0.1, 0] |
| 齐次度 μ | mu_min (自动选择最负容许值) |

### Yaw 回路 (HPC)

| 参数 | 值 |
|------|-----|
| A_yaw | [0 1; 0 0] |
| B_yaw | [0; 1/Izz] |
| 线性 K | [-0.39, -0.21] |
| 齐次度 μ | -0.5 |

### XY 回路 (HPC, 8D)

| 参数 | 值 |
|------|-----|
| LQR Q | diag([10,10,5,5,2,2,0.1,0.1]) |
| LQR R | diag([1,1]) |
| 齐次度 μ | -1.0 |
| Gd 特征值 | [4,4,3,3,2,2,1,1] |

LQR 增益 (2×8, X/Y 通道解耦):

```
τ_pitch: [-3.16,   0,  -2.98,   0,  -5.98,   0,  -0.60,   0  ]
τ_roll:  [  0,  -3.16,   0,  -2.97,   0,  -5.93,   0,  -0.59 ]
```

---

## 参考文献

- Wang, S. (2020). *Homogeneous Control: from Theory to Applications*. PhD Thesis, Centrale Lille.
- Polyakov, A. (2019). *HCS Toolbox for MATLAB* (ver 0.2).
