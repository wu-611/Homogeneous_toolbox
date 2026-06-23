# CLAUDE.md — Homogeneous Control Toolbox

齐次控制工具箱及四旋翼三回路 HPC/HPIC 数值仿真。

## 项目定位

纯**工具箱 + 数值仿真验证**仓库。ROS2 SITL 部署版在 `/home/wzy/px4_nmpc_ws/hcs_3ring/`，
其 Gazebo x500 真值参数和推力链/混控/限幅等工程代码不放入本仓库。

## 目录结构

```
hcs_toolbox_py/           # Python 工具箱 (22 函数)
hcs_toolbox_cpp/          # C++ 工具箱 (header-only, 17 头文件)
demo_uav/                 # 四旋翼三回路仿真 (Wang 2020 复现)
  python/                 #   Python 版
  cpp/                    #   C++ 版 (独立 CMake)
HCS_Toolbox_ver02/        # MATLAB 原始工具箱 (Polyakov ver 0.2)
```

## 构建与运行

```bash
# Python 仿真
python3 demo_uav/python/uav_homogeneous_control.py

# C++ 仿真
cd demo_uav/cpp && mkdir -p build && cd build
cmake .. && make -j$(nproc)
./bin/demo_uav_homogeneous_control
python3 ../plot_uav_comparison.py
```

## 核心理论流程

```
线性 K  ──lpc2hpc(A,B,K)──→  {K0, G0, P, [mu_min, mu_max]}
线性 PI ──lpic2hpic(A,B,K,Ki)──→  {K0, G0, P, Ki_new, [mu_min, mu_max]}

Gd = I + μ·G0
u = K0·x + nx^(1+μ)·(K-K0)·expm(-ln(nx)·Gd)·x   (HPC)
u = uh + ∫ui dt                                   (HPIC)
nx = hnorm(x, Gd, P)
```

三回路 (μ 选择): Z HPIC=-0.5, Yaw HPC=-0.5, XY HPC=-1.0
仿真参数: m=1.4, Ixx=0.0211, Iyy=0.0219, Izz=0.0366, DT=0.001

## 代码约定

- 命名空间: `hcs_toolbox` (C++), `hcs_toolbox_py` (Python)
- C++ 依赖: Eigen 3.4+ (含 unsupported/MatrixFunctions 用于 expm)
- 工具箱为 header-only, 所有函数在 `hcs_toolbox` 命名空间
- e_hpc alpha=0.1 (标准参数)
- hnorm 内部迭代 50 次, 精度 1e-12
- **Commit 信息必须使用中文**

## hcs_toolbox_cpp 已知问题

`lqr.hpp` — 高维系统 Schur 排序有 bug，XY demo 用 Python scipy CARE 预计算的 LQR 增益硬编码。

## 禁止引入

- ADRC (自抗扰控制)
- SE(3) 几何控制器 (已证明不可行)
- Gazebo/SITL 工程代码 (属于 px4_nmpc_ws, 不污染工具库)
- lo2ho/e_ho/eso 的 C++ 版本 (Python 工具箱保留，C++ 不使用)
