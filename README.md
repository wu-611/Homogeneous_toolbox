# Homogeneous Control Toolbox

齐次控制工具箱及四旋翼无人机控制仿真。

## 项目结构

```
├── hcs_toolbox_py/       # Python 齐次控制工具箱 (22 函数)
├── hcs_toolbox_cpp/      # C++ 工具箱 (header-only, 17 头文件)
├── demo_uav/             # 四旋翼三回路 HPC 仿真
│   ├── python/           #   Python 数值仿真
│   ├── cpp/              #   C++ 数值仿真 (独立 CMake)
│   ├── Compute_All_HPC_Params.m  # MATLAB 参数计算脚本
│   └── function.txt      # Simulink 参考实现
├── HCS_Toolbox_ver02/    # MATLAB 原始工具箱 (Polyakov ver 0.2)
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

## 核心理论

将已有线性控制器升级为齐次控制器：

```
线性 K    ──lpc2hpc(A,B,K)──→  {K0, G0, P, [mu_min, mu_max]}
线性 PI   ──lpic2hpic(A,B,K,Ki)──→  {K0, G0, P, Ki_new, [mu_min, mu_max]}

Gd = I + μ·G0
u = K0·x + nx^(1+μ)·(K-K0)·d(-ln nx)·x   (HPC)
u = uh + ∫ui dt                            (HPIC)
```

## 已验证方案

**三回路齐次控制器** (Wang Siyuan 2020 博士论文):

| 回路 | 类型 | μ | 状态 | 输出 |
|------|------|-----|------|------|
| Z | HPIC | -0.5 | [ez, vz] | 推力 Fz |
| Yaw | HPC | -0.5 | [eψ, ωz] | 扭矩 τz |
| XY | HPC | -1.0 | 8D 级联 | 扭矩 τx, τy |

该方案已在 Quanser QDrone 真机上验证。Gazebo SITL 版本在 `/home/wzy/px4_nmpc_ws/hcs_3ring/`。

## 参考文献

1. Polyakov, A. (2020). *Generalized Homogeneity in Systems and Control*. Springer.
2. Wang, S. (2020). *Homogeneous Quadrotor Control: Theory and Experiment*. PhD Thesis.
3. Zhou, Y. et al. (2023). Generalized Homogeneous Rigid-Body Attitude Control. *IEEE TAC*.
