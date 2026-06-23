# HCS Toolbox for C++ — 齐次控制系统工具箱 C++ 移植

Andrey Polyakov 的 **HCS Toolbox ver 0.2**（齐次控制系统工具箱）的 C++ 移植版本。
Header-only 库，仅依赖 Eigen3。

位置：`/home/wzy/Homogeneous_control/hcs_toolbox_cpp/`

---

## 目录

1. [文件结构](#1-文件结构)
2. [依赖与构建](#2-依赖与构建)
3. [运行 Demo](#3-运行-demo)
4. [绘图脚本](#4-绘图脚本)
5. [Demo 说明](#5-demo-说明)
6. [函数参考](#6-函数参考)
7. [在你的项目中使用](#7-在你的项目中使用)

---

## 1. 文件结构

```
hcs_toolbox_cpp/
├── CMakeLists.txt                     # CMake 构建文件
├── README.md
├── include/
│   └── hcs_toolbox/
│       ├── hcs_toolbox.hpp            # 伞形头文件（包含全部函数）
│       ├── hnorm.hpp                  # 齐次范数 (二分法)
│       ├── hproj.hpp                  # 齐次投影到单位球面
│       ├── hphi.hpp                   # 齐次同胚 Φ(x) / Φ⁻¹(y)
│       ├── hadd.hpp                   # 齐次欧氏空间加法
│       ├── hdot.hpp                   # 齐次欧氏空间标量乘法
│       ├── hinner.hpp                 # 齐次欧氏空间内积
│       ├── block_con.hpp              # 典范块可控型分解
│       ├── trans_con.hpp              # 正交阶梯变换
│       ├── ZOH.hpp                    # 零阶保持器离散化
│       ├── lpc2hpc.hpp                # LPC → HPC 升级
│       ├── lpic2hpic.hpp              # LPIC → HPIC 升级
│       ├── e_hpc.hpp                  # HPC 控制求值
│       ├── e_hpic.hpp                 # HPIC 控制求值
│       ├── e_fhpc.hpp                 # 固定时间 FHPC
│       ├── e_fhpic.hpp                # 固定时间 FHPIC
│       └── lqr.hpp                    # LQR 设计 (已知 Schur 排序 bug, 慎用)
├── src/
│   ├── demo_lpc2hpc.cpp               # Demo 1: 倒立摆 HPC 升级
│   ├── demo_lpic2hpic.cpp             # Demo 2: 倒立摆 HPIC 升级
│   └── demo_lpc_hpc_distance_square.cpp # Demo 3: 追逃跟踪与编队切换
└── scripts/
    ├── plot_demo_lpc2hpc.py           # Demo 1 绘图脚本
    ├── plot_demo_lpic2hpic.py         # Demo 2 绘图脚本
    └── plot_demo_distance_square.py   # Demo 3 绘图脚本
```

**17 个头文件，3 个 Demo，3 个绘图脚本**。全部 header-only，无编译依赖（除 Eigen3）。

---

## 2. 依赖与构建

### 系统要求

- C++17 兼容编译器 (GCC 8+, Clang 10+)
- CMake 3.14+
- **Eigen 3.4+**（含 unsupported/MatrixFunctions 模块用于 `expm`）

### 安装 Eigen3（如未安装）

```bash
sudo apt install libeigen3-dev
```

### 构建

```bash
cd /home/wzy/Homogeneous_control/hcs_toolbox_cpp
mkdir -p build && cd build
cmake ..
make -j$(nproc)
```

构建成功后，可执行文件在 `build/bin/` 目录下：

```
build/bin/demo_lpc2hpc
build/bin/demo_lpic2hpic
build/bin/demo_lpc_hpc_distance_square
```

---

## 3. 运行 Demo

```bash
cd /home/wzy/Homogeneous_control/hcs_toolbox_cpp/build

# Demo 1: HPC 比例控制升级（倒立摆，平滑收敛）
./bin/demo_lpc2hpc

# Demo 2: HPIC 比例-积分控制升级（倒立摆，忠实复现 MATLAB 原版）
./bin/demo_lpic2hpic

# Demo 3: HPC 追逃跟踪（双积分器智能体，编队切换）
./bin/demo_lpc_hpc_distance_square
```

每个 Demo 运行后在当前目录生成 CSV 数据文件：

| Demo | CSV 输出文件 |
|------|-------------|
| demo_lpc2hpc | `demo_lpc2hpc_cpp.csv` |
| demo_lpic2hpic | `demo_lpic2hpic_cpp.csv` |
| demo_lpc_hpc_distance_square | `demo_lpc_hpc_distance_square_cpp.csv` |

---

## 4. 绘图脚本

### 运行绘图

```bash
cd /home/wzy/Homogeneous_control/hcs_toolbox_cpp/build

# 先运行 Demo 生成 CSV，然后绘图
./bin/demo_lpc2hpc
python3 ../scripts/plot_demo_lpc2hpc.py

./bin/demo_lpic2hpic
python3 ../scripts/plot_demo_lpic2hpic.py

./bin/demo_lpc_hpc_distance_square
python3 ../scripts/plot_demo_distance_square.py
```

### 显式指定 CSV 路径

```bash
python3 ../scripts/plot_demo_lpc2hpc.py /path/to/demo_lpc2hpc_cpp.csv
```

### 生成图片一览

| Demo | 生成图片 |
|------|---------|
| lpc2hpc | `demo_lpc2hpc_cpp.png`（状态 + 控制，2 面板） |
| lpic2hpic | `demo_lpic2hpic_cpp.png`（状态 + 控制，2 面板） |
| distance_square | 6 张图：X/Y 位置、2D 轨迹、控制输入、误差分量、误差范数 |

---

## 5. Demo 说明

### Demo 1: `demo_lpc2hpc` — 线性比例 → HPC 升级

| 项目 | 说明 |
|------|------|
| 系统 | 旋转倒立摆 (QUBE-Servo 2)，4 状态，1 输入 |
| 线性增益 | K_lin = [2, -35, 1.5, -3] |
| 齐次度 | μ = -1.0（有限时间收敛，hn⁰ = 1，无放大） |
| 离散化 | ZOH（零阶保持器） |
| 控制饱和 | ±10V（QUBE-Servo 2 硬件限幅） |
| 结果 | \|\|x(3s)\|\| = 0.000591（平滑收敛至零） |

**控制律：**
```
u = K0·x + nx^(1+μ) · K · expm(-ln(nx)·Gd) · x
```

### Demo 2: `demo_lpic2hpic` — 线性 PI → HPIC 升级

| 项目 | 说明 |
|------|------|
| 系统 | 同上倒立摆，增加积分作用 + 常值扰动 p=1 |
| 线性增益 | K_lin = [2, -35, 1.5, -3]，Ki = [0.5, -26.66, 1.26, -2.73] |
| 齐次度 | μ = +0.16（正齐次度 → hn^1.16 ≈ 23× 放大） |
| 离散化 | 显式 Euler（忠实复现 MATLAB） |
| 控制饱和 | 无（忠实复现 MATLAB） |
| 结果 | \|\|x(4s)\|\| = 0.913（瞬态有振荡，原因见下文） |

**振荡原因：** μ=+0.16 导致齐次范数放大 23 倍，无控制饱和时初始控制达 -49（远超硬件 ±10V 限制），开环不稳定系统（A 特征值 +13.58）在 Euler 积分下状态范数从 1.4 爆炸到 28（20ms 内）后缓慢收敛。

**优化建议：** 使用负齐次度 μ<0 + ZOH 离散化 + 控制饱和。详见 Python 工具箱中的 `demo_lpic2hpic_optimized.py`。

### Demo 3: `demo_lpc_hpc_distance_square` — 追逃跟踪

| 项目 | 说明 |
|------|------|
| 系统 | 两个双积分器智能体（二维平面运动），各 4 状态 2 输入 |
| Agent 1（目标） | 线性 PD + 正弦参考 [sin(t), cos(t)] |
| Agent 2（追捕者） | HPC 跟踪 + 编队偏移 d |
| 编队点 | 4 个点，半径 1 的圆上 |
| 切换规则 | 当另一个编队点更近时自动切换，重新设计 HPC |
| 自适应增益 | 根据速度/位置误差比动态调整线性增益 |
| 结果 | \|\|e(30s)\|\| = 0.0108 |

**专用控制律：**
```
nx_sat = clamp(nx, 0.1, 1.0)
u2 = nx_sat^(1+ν) · k_lin · expm(Gd·(1 - ln(nx_sat))) · e
```

---

## 6. 函数参考

所有函数位于命名空间 `hcs_toolbox`，通过伞形头文件引入：

```cpp
#include "hcs_toolbox/hcs_toolbox.hpp"
```

### 升级函数

| 函数 | 返回类型 | 说明 |
|------|---------|------|
| `lpc2hpc(A, B, K)` | `HPCParams{K0,G0,P,mu_min,mu_max}` | 线性比例 → HPC |
| `lpic2hpic(A, B, K, Ki)` | `HPICParams{K0,G0,P,Ki_new,mu_min,mu_max}` | 线性 PI → HPIC |

### 控制器求值

| 函数 | 说明 |
|------|------|
| `e_hpc(x, K0, K, Gd, mu, hn_fun, α, β) → u` | HPC 控制量 |
| `e_hpic(x, K0, K, Ki, Gd, mu, hn_fun, uh, ui, α, β)` | HPIC 比例+积分分量 |
| `e_fhpc(x, K0, K, G0, μ1, μ2, P, α, β) → u` | 固定时间 FHPC |
| `e_fhpic(x, K0, K, Ki, G0, μ1, μ2, P, uh, ui, α, β)` | 固定时间 FHPIC |

### 齐次几何

| 函数 | 说明 |
|------|------|
| `hnorm(x, Gd, P, tol, Nmax) → nx` | 典范齐次范数（二分法） |
| `hproj(x, Gd, hn_fun, z, s)` | 投影到单位球面，返回 z 和 s |
| `hphi(x, Gd, hn_fun) → y` | 齐次同胚 Φ(x) |
| `hphi_inv(y, Gd, n_fun) → x` | 逆同胚 Φ⁻¹(y) |
| `hadd(x1, x2, Gd, P) → y` | 齐次加法 Φ⁻¹(Φ(x1)+Φ(x2)) |
| `hdot(alpha, x, Gd) → y` | 齐次标量乘 sign(α)·d(ln\|α\|)·x |
| `hinner(x, y, Gd, P) → q` | 齐次内积 Φ(x)'·P·Φ(y) |

### 工具函数

| 函数 | 说明 |
|------|------|
| `block_con(A, B, T, nt) → bool` | 块可控典范型分解 |
| `trans_con(A, B, T, nt) → bool` | 正交阶梯变换 |
| `ZOH(h, A, B, Ah, Bh)` | 零阶保持器离散化 |

---

## 7. 在你的项目中使用

### 方式一：作为子目录（推荐）

```cmake
# CMakeLists.txt
add_subdirectory(path/to/hcs_toolbox_cpp)
target_link_libraries(your_target PRIVATE hcs_toolbox)
```

### 方式二：手动包含头文件

```cmake
target_include_directories(your_target PRIVATE path/to/hcs_toolbox_cpp/include)
target_link_libraries(your_target PRIVATE Eigen3::Eigen)
```

### 最小示例

```cpp
#include "hcs_toolbox/hcs_toolbox.hpp"

// 双积分器系统: ẋ = v, v̇ = u/m
Eigen::Matrix2d A;
A << 0, 1, 0, 0;
Eigen::Vector2d B(0, 1.0 / 2.0);  // m = 2
Eigen::RowVector2d K(-1.0, -2.0);  // 线性反馈

// 升级为 HPC
auto params = hcs_toolbox::lpc2hpc(A, B, K);
double mu = params.mu_min;
Eigen::Matrix2d Gd = Eigen::Matrix2d::Identity() + mu * params.G0;
auto K_nl = K - params.K0;

// 齐次范数函数
auto hn_fun = [&](const Eigen::VectorXd& x) {
    return hcs_toolbox::hnorm(x, Gd, params.P);
};

// 控制循环
Eigen::Vector2d x(1.0, 0.0);  // 初始状态
double h = 0.001;
for (int step = 0; step < 1000; ++step) {
    Eigen::VectorXd u = hcs_toolbox::e_hpc(x, params.K0, K_nl, Gd, mu, hn_fun, 0.1, 1.0);
    x = x + h * (A * x + B * u(0));  // Euler 积分
}
```

---

## 数值验证

C++ 版本与 Python 版本结果一致（误差 < 0.01%）：

| Demo | C++ 结果 | Python 结果 | 匹配 |
|------|---------|------------|------|
| lpc2hpc | \|\|x(3s)\|\| = 0.000591 | 0.000587 | ✓ |
| lpic2hpic | \|\|x(4s)\|\| = 0.913 | 0.913 | ✓ |
| distance_square | \|\|e(30s)\|\| = 0.0108 | 0.0108 | ✓ |
