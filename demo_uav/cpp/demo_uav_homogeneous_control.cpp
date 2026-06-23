/**
 * @file demo_uav_homogeneous_control.cpp
 * @brief 四旋翼无人机三回路齐次控制器 — C++ 复现 Compute_All_HPC_Params.m
 *
 * ======================== 论文背景 ========================
 * 基于王思远 (Wang Siyuan) 2020 年博士论文（Centrale Lille），将线性 PID
 * 控制器升级为齐次控制器 (HPC/HPIC)。核心思想：引入膨胀对称性，通过
 * 典范齐次范数 ||x||_d 对状态进行非线性缩放，消除线性高增益的 peaking 效应。
 *
 * ======================== 三回路解耦架构 ========================
 * 四旋翼线性化模型可解耦为三个独立子系统：
 *
 *   Z   回路 (2维, HPIC): 高度误差 + 垂直速度 → 净推力 Fz
 *        - 模型: 双积分器 (m = 1.4 kg)
 *        - 控制: u = uh + v, dv/dt = ui (齐次比例 + 齐次积分)
 *        - 齐次度 μ = -0.5 (有限时间收敛)
 *
 *   Yaw 回路 (2维, HPC):  偏航角误差 + 偏航角速率 → 偏航力矩 τz
 *        - 模型: 双积分器 (Izz = 0.0366 kg·m²)
 *        - 控制: u = K0·x + nx^(1+μ)·(K-K0)·d(-ln(nx))·x
 *        - 齐次度 μ = -0.5 (有限时间收敛)
 *
 *   XY  回路 (8维, HPC):  水平位置+速度+姿态角+角速率 → 俯仰/滚转力矩
 *        - 模型: 8阶级联系统 (Ixx=0.0211, Iyy=0.0219)
 *        - 线性增益: LQR 设计 (Q位置权重=10, R力矩权重=1)
 *        - 齐次度 μ = -1.0 (有限时间收敛, 滑模类行为)
 *
 * ======================== 仿真内容 ========================
 * 每个回路同时仿真线性控制器与齐次控制器，对比阶跃响应。
 * 输出 CSV 文件，可用 ../scripts/plot_uav_comparison.py 绘图。
 *
 * 构建: cd hcs_toolbox_cpp/build && cmake .. && make
 * 运行: ./bin/demo_uav_homogeneous_control
 * 绘图: python3 ../scripts/plot_uav_comparison.py
 */

#include "hcs_toolbox/hcs_toolbox.hpp"   // 齐次控制工具箱 (header-only)
#include <Eigen/Dense>                    // 线性代数
#include <unsupported/Eigen/MatrixFunctions>  // expm (矩阵指数)
#include <iostream>                       // cout, endl
#include <fstream>                        // ofstream (CSV输出)
#include <vector>                         // vector
#include <cmath>                          // sqrt, pow, log
#include <iomanip>                        // setprecision

// ===========================================================================
// 全局物理常数
// ===========================================================================
constexpr double G  = 9.8;    // 重力加速度 [m/s²]
constexpr double DT = 0.001;  // 仿真步长 [s] (= 1ms, 匹配 1000Hz PX4 控制频率)

// ===========================================================================
// 仿真辅助函数: Z 轴 (高度) HPIC — 齐次比例-积分控制器
// ===========================================================================

/**
 * @brief Z 轴 HPIC (齐次比例-积分) 仿真
 *
 * 控制律:
 *   uh = K0·x + nx^(1+μ)·(K-K0)·d(-ln(nx))·x   (齐次比例分量)
 *   ui = nx^(1+2μ)·Ki_new·d(-ln(nx))·x          (齐次积分分量)
 *   u  = uh + v,   dv/dt = ui
 *
 * @param x0    初始状态 [z_error, vz] (2维)
 * @param steps 仿真步数
 * @param Ah    ZOH 离散化后的系统矩阵 (2×2)
 * @param Bh    ZOH 离散化后的控制向量 (2×1)
 * @param K0    齐次化反馈增益 (1×2) — 线性分量
 * @param K_nl  非线性比例增益 K - K0 (1×2)
 * @param Ki_new 重构后的积分增益 (1×2)
 * @param Gd    膨胀算子生成元 Gd = I + μ·G0 (2×2)
 * @param mu    齐次度 (此处 μ = -0.5)
 * @param P     形状矩阵 (2×2), 定义典范齐次范数
 * @return      状态轨迹矩阵 X (2 × steps+1)
 */
Eigen::MatrixXd sim_z_hpic(const Eigen::Vector2d& x0, int steps,
                           const Eigen::Matrix2d& Ah, const Eigen::Vector2d& Bh,
                           const Eigen::RowVector2d& K0, const Eigen::RowVector2d& K_nl,
                           const Eigen::RowVector2d& Ki_new,
                           const Eigen::Matrix2d& Gd, double mu,
                           const Eigen::Matrix2d& P) {
    // 预分配状态轨迹矩阵 (2行 × steps+1列, 列主序)
    Eigen::MatrixXd X(2, steps + 1);
    Eigen::Vector2d x = x0;
    double v_int = 0.0;  // 积分累加器 v(t) = ∫ ui dt
    X.col(0) = x;        // 记录初始状态

    // 构造齐次范数函数: 闭包捕获 Gd 和 P
    auto hn_fun = [&](const Eigen::VectorXd& v) { return hcs_toolbox::hnorm(v, Gd, P); };

    // ---- 仿真主循环 ----
    for (int k = 0; k < steps; ++k) {
        // 步骤1: 调用 e_hpic 计算齐次比例分量 uh 和积分速率 ui
        //         alpha=0.1 防止 nx→0 时控制奇异 (工程饱和处理)
        Eigen::VectorXd uh, ui;
        hcs_toolbox::e_hpic(x, K0, K_nl, Ki_new, Gd, mu, hn_fun, uh, ui, 0.1);

        // 步骤2: 总控制量 = 比例分量 + 积分累加值
        double u = uh(0) + v_int;

        // 步骤3: 积分累加 (显式 Euler 积分 dv/dt = ui)
        v_int += DT * ui(0);

        // 步骤4: 状态更新 (ZOH 离散化: x_{k+1} = Ah·x_k + Bh·u_k)
        x = Ah * x + Bh * u;
        X.col(k + 1) = x;
    }
    return X;
}

/**
 * @brief Z 轴线性 PI 仿真 (用于对比)
 *
 * 控制律: u = K·x + v,  dv/dt = Ki·x
 */
Eigen::MatrixXd sim_z_linear(const Eigen::Vector2d& x0, int steps,
                             const Eigen::Matrix2d& Ah, const Eigen::Vector2d& Bh,
                             const Eigen::RowVector2d& K, const Eigen::RowVector2d& Ki) {
    Eigen::MatrixXd X(2, steps + 1);
    Eigen::Vector2d x = x0;
    double v_int = 0.0;
    X.col(0) = x;
    for (int k = 0; k < steps; ++k) {
        double u = (K * x)(0) + v_int;    // 线性比例 + 积分
        v_int += DT * (Ki * x)(0);        // 积分累加
        x = Ah * x + Bh * u;
        X.col(k + 1) = x;
    }
    return X;
}

// ===========================================================================
// 仿真辅助函数: Yaw 轴 (偏航) HPC — 齐次比例控制器 (无积分)
// ===========================================================================

/**
 * @brief Yaw 轴 HPC (齐次比例) 仿真
 *
 * 控制律:
 *   u = K0·x + nx^(1+μ)·(K-K0)·d(-ln(nx))·x
 *
 * 与 Z 轴不同: Yaw 回路是纯比例控制 (HPC), 无积分项。
 * 双积分器系统本身可被 PD 镇定, 不需要积分来消除静差。
 *
 * @param x0   初始状态 [psi_error, omega_z] (2维)
 * @param K0   齐次化反馈增益 (1×2)
 * @param K_nl 非线性增益 (1×2)
 * @param Gd   膨胀算子生成元 (2×2)
 * @param mu   齐次度 (此处 μ = -0.5)
 */
Eigen::MatrixXd sim_yaw_hpc(const Eigen::Vector2d& x0, int steps,
                            const Eigen::Matrix2d& Ah, const Eigen::Vector2d& Bh,
                            const Eigen::RowVector2d& K0, const Eigen::RowVector2d& K_nl,
                            const Eigen::Matrix2d& Gd, double mu,
                            const Eigen::Matrix2d& P) {
    Eigen::MatrixXd X(2, steps + 1);
    Eigen::Vector2d x = x0;
    X.col(0) = x;
    auto hn_fun = [&](const Eigen::VectorXd& v) { return hcs_toolbox::hnorm(v, Gd, P); };
    for (int k = 0; k < steps; ++k) {
        // e_hpc: 齐次比例控制求值
        // alpha=0.1, beta=1.0 钳制齐次范数 nx ∈ [0.1, 1.0]
        // 当 μ=-0.5 时, nx^(1+μ) = nx^0.5, 对 nx<1 的衰减起缓解作用
        Eigen::VectorXd u_vec = hcs_toolbox::e_hpc(x, K0, K_nl, Gd, mu, hn_fun, 0.1, 1.0);
        x = Ah * x + Bh * u_vec(0);
        X.col(k + 1) = x;
    }
    return X;
}

/**
 * @brief Yaw 轴线性 PD 仿真 (用于对比)
 *
 * 控制律: u = K·x  (纯比例, 无积分)
 */
Eigen::MatrixXd sim_yaw_linear(const Eigen::Vector2d& x0, int steps,
                               const Eigen::Matrix2d& Ah, const Eigen::Vector2d& Bh,
                               const Eigen::RowVector2d& K) {
    Eigen::MatrixXd X(2, steps + 1);
    Eigen::Vector2d x = x0;
    X.col(0) = x;
    for (int k = 0; k < steps; ++k) {
        x = Ah * x + Bh * (K * x)(0);
        X.col(k + 1) = x;
    }
    return X;
}

// ===========================================================================
// 仿真辅助函数: XY 轴 (水平位置+姿态) HPC — 8维齐次比例控制器
// ===========================================================================

/**
 * @brief XY 轴 HPC (8维齐次比例) 仿真
 *
 * 状态向量: ξ = [x, y, vx, vy, θ, φ, q, p]^T  (8维)
 *   其中: x,y = 水平位置误差 [m]
 *         vx,vy = 水平速度误差 [m/s]
 *         θ = 俯仰角 (pitch) [rad]
 *         φ = 滚转角 (roll)  [rad]
 *         q = 俯仰角速率 [rad/s]
 *         p = 滚转角速率 [rad/s]
 *
 * 系统动力学 (级联结构):
 *   ẋ = vx,   v̇x = g·θ        (俯仰产生纵向加速度)
 *   ẏ = vy,   v̇y = g·φ        (滚转产生横向加速度)
 *   θ̇ = q,   q̇ = τ_pitch/Iyy  (俯仰力矩 → 俯仰角加速度)
 *   φ̇ = p,   ṗ = τ_roll /Ixx  (滚转力矩 → 滚转角加速度)
 *
 * 线性增益来源: LQR 求解 CARE (代数 Riccati 方程)
 *   Q = diag([10,10,5,5,2,2,0.1,0.1])
 *   R = eye(2)
 *   K = -inv(R)·B'·P  (P 为 Riccati 解)
 *
 * @param x0   初始状态 (8维)
 * @param K0   齐次化反馈增益 (2×8)
 * @param K_nl 非线性增益 (2×8)
 * @param Gd   膨胀算子生成元 (8×8), μ=-1 时 Gd = I - G0
 * @param mu   齐次度 (此处 μ = -1.0, 滑模类行为)
 */
Eigen::MatrixXd sim_xy_hpc(const Eigen::Matrix<double, 8, 1>& x0, int steps,
                           const Eigen::Matrix<double, 8, 8>& Ah,
                           const Eigen::Matrix<double, 8, 2>& Bh,
                           const Eigen::Matrix<double, 2, 8>& K0,
                           const Eigen::Matrix<double, 2, 8>& K_nl,
                           const Eigen::Matrix<double, 8, 8>& Gd, double mu,
                           const Eigen::Matrix<double, 8, 8>& P) {
    Eigen::MatrixXd X(8, steps + 1);
    Eigen::Matrix<double, 8, 1> x = x0;
    X.col(0) = x;
    auto hn_fun = [&](const Eigen::VectorXd& v) { return hcs_toolbox::hnorm(v, Gd, P); };
    for (int k = 0; k < steps; ++k) {
        // HPC 控制: u = K0·x + nx^(1+μ)·(K-K0)·expm(-ln(nx)·Gd)·x
        // μ=-1 时, nx^(1+μ) = nx^0 = 1 → 无幅值放大, 仅方向投影
        Eigen::VectorXd u = hcs_toolbox::e_hpc(x, K0, K_nl, Gd, mu, hn_fun, 0.1, 1.0);
        x = Ah * x + Bh * u;
        X.col(k + 1) = x;
    }
    return X;
}

/**
 * @brief XY 轴线性 LQR 仿真 (用于对比)
 *
 * 控制律: u = K·x  (纯线性状态反馈)
 */
Eigen::MatrixXd sim_xy_linear(const Eigen::Matrix<double, 8, 1>& x0, int steps,
                              const Eigen::Matrix<double, 8, 8>& Ah,
                              const Eigen::Matrix<double, 8, 2>& Bh,
                              const Eigen::Matrix<double, 2, 8>& K) {
    Eigen::MatrixXd X(8, steps + 1);
    Eigen::Matrix<double, 8, 1> x = x0;
    X.col(0) = x;
    for (int k = 0; k < steps; ++k) {
        x = Ah * x + Bh * (K * x);
        X.col(k + 1) = x;
    }
    return X;
}

// ===========================================================================
// 工具函数: 将齐次+线性对比数据写入 CSV 文件
// ===========================================================================

/**
 * @brief 输出 CSV 文件 (齐次 vs 线性对比)
 *
 * CSV 格式:
 *   t, state1_hpc, state2_hpc, ..., state1_lin, state2_lin, ...
 *
 * @param fname     输出文件名
 * @param t         时间向量
 * @param X_hpc     齐次控制器状态轨迹 (n × N)
 * @param X_lin     线性控制器状态轨迹 (n × N)
 * @param col_names 列名列表 (对齐次+线性各状态)
 */
void write_csv(const std::string& fname,
               const Eigen::VectorXd& t,
               const Eigen::MatrixXd& X_hpc,
               const Eigen::MatrixXd& X_lin,
               const std::vector<std::string>& col_names) {
    std::ofstream f(fname);
    // 写入表头
    f << "t";
    for (auto& c : col_names) f << "," << c;
    f << "\n";
    // 逐行写入数据
    for (int k = 0; k < t.size(); ++k) {
        f << t(k);
        for (int i = 0; i < X_hpc.rows(); ++i) f << "," << X_hpc(i, k);
        for (int i = 0; i < X_lin.rows(); ++i)  f << "," << X_lin(i, k);
        f << "\n";
    }
}

// ===========================================================================
// 主函数
// ===========================================================================
int main() {
    // 设置浮点数输出格式: 固定小数点, 6 位精度
    std::cout << std::fixed << std::setprecision(6);
    std::cout << "============================================================\n";
    std::cout << "四旋翼三回路齐次控制器 — C++ 仿真\n";
    std::cout << "复现 Compute_All_HPC_Params.m (王思远 2020 博士论文)\n";
    std::cout << "============================================================\n\n";

    // ========================================================================
    // 第1回路: Z 轴 (高度) HPIC — 齐次比例-积分控制器
    // ========================================================================
    // 模型: 双积分器  ẑ = u/m
    // 状态: x = [z_error, vz]^T
    // 控制: u = uh + ∫ui dt  (HPIC = HPC + 齐次积分)
    // ========================================================================
    std::cout << ">>> 第1回路: Z 轴 (高度) HPIC 齐次比例-积分控制器\n";

    // --- 物理参数 ---
    double m = 1.4;   // 无人机质量 [kg]

    // --- 状态空间模型 ---
    // A_z = [0 1; 0 0], B_z = [0; 1/m]
    Eigen::Matrix2d A_z;
    A_z << 0.0, 1.0,
           0.0, 0.0;
    Eigen::Vector2d B_z(0.0, 1.0 / m);

    // --- 线性控制器增益 (来自原厂 PID) ---
    // K_z_lin = [-5, -2]: 高度误差增益 -5, 速度误差增益 -2
    // Ki_z_lin = [-0.1, 0]: 仅对高度误差积分, 速度通道无积分
    Eigen::RowVector2d Kz_lin(-5.0, -2.0);
    Eigen::RowVector2d Kiz_lin(-0.1, 0.0);

    // --- 步骤1: 调用 lpic2hpic 升级线性 PI → 齐次 HPIC ---
    // 返回:
    //   K0_z:   齐次化反馈增益 (线性分量)
    //   G0_z:   膨胀生成元基础矩阵
    //   P_z:    形状矩阵 (定义典范齐次范数)
    //   Ki_new: 重构后的齐次积分增益
    //   mu_min, mu_max: 齐次度 μ 容许范围
    auto pz = hcs_toolbox::lpic2hpic(A_z, B_z, Kz_lin, Kiz_lin);

    // --- 步骤2: 选择齐次度 μ ---
    // μ = -0.5: 负齐次度 → 有限时间收敛
    // 必须在 [mu_min, mu_max] 范围内
    double mu_z = -0.5;

    // --- 步骤3: 构造膨胀算子生成元 Gd ---
    // Gd = I + μ·G0,  膨胀算子 d(s) = expm(s·Gd)
    // 特征值 λ(Gd) > 0 确保膨胀的单调性
    Eigen::Matrix2d Gd_z = Eigen::Matrix2d::Identity() + mu_z * pz.G0;

    // --- 步骤4: 计算非线性增益 ---
    // K_nl = K_lin - K0  (齐次控制器的非线性分量)
    Eigen::RowVector2d Kz_nl = Kz_lin - pz.K0;

    std::cout << "  mu = " << mu_z
              << ", 容许范围 [" << pz.mu_min << ", " << pz.mu_max << "]\n";
    std::cout << "  K0_z   = [" << pz.K0(0) << ", " << pz.K0(1) << "]\n";
    std::cout << "  Ki_new = [" << pz.Ki_new(0) << ", " << pz.Ki_new(1) << "]\n";
    std::cout << "  Gd_z 特征值: [" << Gd_z.eigenvalues().real().transpose() << "]\n";

    // --- 步骤5: ZOH 离散化 ---
    // 将连续系统 ẋ = A·x + B·u 离散化为 x_{k+1} = Ah·x_k + Bh·u_k
    Eigen::Matrix2d Ah_z;
    Eigen::Vector2d Bh_z;
    {
        Eigen::MatrixXd tmpA, tmpB;
        hcs_toolbox::ZOH(DT, A_z, B_z, tmpA, tmpB);
        Ah_z = tmpA;
        Bh_z = tmpB.col(0);
    }

    // --- 步骤6: 阶跃响应仿真 ---
    int Nz = static_cast<int>(5.0 / DT);  // 仿真 5 秒, 共 5000 步
    Eigen::Vector2d x0_z(1.0, 0.0);       // 初始: 高度误差 1m, 速度 0

    // 齐次 HPIC 仿真
    auto Xz_hpic = sim_z_hpic(x0_z, Nz, Ah_z, Bh_z,
                              pz.K0, Kz_nl, pz.Ki_new, Gd_z, mu_z, pz.P);
    // 线性 PI 仿真 (对比)
    auto Xz_lin  = sim_z_linear(x0_z, Nz, Ah_z, Bh_z, Kz_lin, Kiz_lin);

    std::cout << "  ||x(T)||  HPIC = " << Xz_hpic.col(Nz).norm()
              << "  (线性 PI: " << Xz_lin.col(Nz).norm() << ")\n\n";

    // ========================================================================
    // 第2回路: Yaw 轴 (偏航) HPC — 齐次比例控制器 (无积分)
    // ========================================================================
    // 模型: 双积分器  ψ̈ = τz / Izz
    // 状态: x = [psi_error, omega_z]^T
    // 控制: u = K0·x + nx^(1+μ)·(K-K0)·d(-ln(nx))·x
    // ========================================================================
    std::cout << ">>> 第2回路: Yaw 轴 (偏航) HPC 齐次比例控制器\n";

    // --- 物理参数 ---
    double Izz = 0.0366;  // Z 轴转动惯量 [kg·m²]

    // --- 状态空间模型 ---
    Eigen::Matrix2d A_yaw;
    A_yaw << 0.0, 1.0,
             0.0, 0.0;
    Eigen::Vector2d B_yaw(0.0, 1.0 / Izz);

    // --- 线性 PD 增益 ---
    // K_yaw_lin = [-0.39, -0.21]: 角度误差增益 -0.39, 角速度误差增益 -0.21
    Eigen::RowVector2d Ky_lin(-0.39, -0.21);

    // --- 升级线性 PD → 齐次 HPC ---
    // Yaw 回路无积分, 使用 lpc2hpc (不涉及 Ki)
    auto py = hcs_toolbox::lpc2hpc(A_yaw, B_yaw, Ky_lin);

    double mu_yaw = -0.5;
    Eigen::Matrix2d Gd_yaw = Eigen::Matrix2d::Identity() + mu_yaw * py.G0;
    Eigen::RowVector2d Ky_nl = Ky_lin - py.K0;

    std::cout << "  mu = " << mu_yaw
              << ", 容许范围 [" << py.mu_min << ", " << py.mu_max << "]\n";
    std::cout << "  K0_yaw = [" << py.K0(0) << ", " << py.K0(1) << "]\n";
    std::cout << "  Gd_yaw 特征值: [" << Gd_yaw.eigenvalues().real().transpose() << "]\n";

    // --- ZOH 离散化 ---
    Eigen::Matrix2d Ah_y;
    Eigen::Vector2d Bh_y;
    {
        Eigen::MatrixXd tmpA, tmpB;
        hcs_toolbox::ZOH(DT, A_yaw, B_yaw, tmpA, tmpB);
        Ah_y = tmpA;
        Bh_y = tmpB.col(0);
    }

    // --- 阶跃响应仿真 ---
    int Ny = static_cast<int>(5.0 / DT);
    Eigen::Vector2d x0_y(0.5, 0.0);       // 初始偏航角误差 0.5 rad (~28.6°)

    auto Xy_hpc = sim_yaw_hpc(x0_y, Ny, Ah_y, Bh_y,
                              py.K0, Ky_nl, Gd_yaw, mu_yaw, py.P);
    auto Xy_lin = sim_yaw_linear(x0_y, Ny, Ah_y, Bh_y, Ky_lin);

    std::cout << "  ||x(T)||  HPC = " << Xy_hpc.col(Ny).norm()
              << "  (线性 PD: " << Xy_lin.col(Ny).norm() << ")\n\n";

    // ========================================================================
    // 第3回路: XY 轴 (水平位置+姿态) HPC — 8维齐次比例控制器
    // ========================================================================
    // 模型: 8阶级联系统
    //   ẋ = v, v̇ = g·θ, θ̇ = q, q̇ = τ/Iyy  (俯仰通道)
    //   ẏ = vy, v̇y = g·φ, φ̇ = p, ṗ = τ/Ixx (滚转通道)
    // 控制: u = K0·x + nx^(1+μ)·(K-K0)·d(-ln(nx))·x
    // ========================================================================
    std::cout << ">>> 第3回路: XY 轴 (水平位置+姿态) HPC 控制器 (8维, LQR设计)\n";

    // --- 物理参数 ---
    double Ixx = 0.0211;  // X 轴转动惯量 [kg·m²] (滚转)
    double Iyy = 0.0219;  // Y 轴转动惯量 [kg·m²] (俯仰)

    // --- 构造 8×8 系统矩阵 A_xy ---
    // 结构:
    //   [0  0  I  0  0  0  0  0]   位置行 → 速度
    //   [0  0  0  I  0  0  0  0]
    //   [0  0  0  0  g  0  0  0]   速度行 → 加速度 (g·θ 产生水平加速度)
    //   [0  0  0  0  0  g  0  0]
    //   [0  0  0  0  0  0  I  0]   姿态行 → 角速度
    //   [0  0  0  0  0  0  0  I]
    //   [0  0  0  0  0  0  0  0]   角速度行 (由力矩驱动)
    //   [0  0  0  0  0  0  0  0]
    Eigen::Matrix<double, 8, 8> A_xy = Eigen::Matrix<double, 8, 8>::Zero();
    A_xy.block<2, 2>(0, 2) = Eigen::Matrix2d::Identity();      // 位置 → 速度
    A_xy.block<2, 2>(2, 4) = G * Eigen::Matrix2d::Identity();  // 姿态 → 加速度 (g·θ)
    A_xy.block<2, 2>(4, 6) = Eigen::Matrix2d::Identity();      // 角速度 → 角加速度

    // --- 构造 8×2 控制矩阵 B_xy ---
    // 输入: u = [τ_pitch, τ_roll]^T (俯仰力矩, 滚转力矩)
    // B_xy(6,0) = 1/Iyy: 俯仰力矩 → 俯仰角加速度 (第6行 = q̇)
    // B_xy(7,1) = 1/Ixx: 滚转力矩 → 滚转角加速度 (第7行 = ṗ)
    Eigen::Matrix<double, 8, 2> B_xy = Eigen::Matrix<double, 8, 2>::Zero();
    B_xy(6, 0) = 1.0 / Iyy;  // 俯仰通道: τ_pitch → q̇
    B_xy(7, 1) = 1.0 / Ixx;  // 滚转通道: τ_roll  → ṗ

    // --- LQR 增益矩阵 (预计算自 Python scipy.linalg.solve_continuous_are) ---
    // 权重设计:
    //   Q = diag([10, 10,   ← 重点关注位置精度
    //              5,  5,   ← 速度权重
    //              2,  2,   ← 姿态角权重
    //            0.1,0.1])  ← 角速度权重 (防止过激)
    //   R = eye(2)          ← 限制力矩幅值
    //
    // 结果 K 矩阵 (2×8) — X/Y 通道完美解耦 (交替零模式):
    //   行1 (俯仰): [-3.16, 0, -2.98, 0, -5.98, 0, -0.60, 0]
    //   行2 (滚转): [0, -3.16, 0, -2.97, 0, -5.93, 0, -0.59]
    Eigen::Matrix<double, 2, 8> Kxy_lin;
    Kxy_lin << -3.16227766,  0.,         -2.97618279,  0.,
               -5.97751232,  0.,         -0.60151063,  0.,
                0.,         -3.16227766,  0.,         -2.97089961,
                0.,         -5.9288273,   0.,         -0.59177404;

    std::cout << "  LQR 增益 K_xy (2×8):\n" << Kxy_lin << "\n";

    // --- 升级线性 LQR → 齐次 HPC (8维) ---
    // lpc2hpc 对 8 维系统进行块分解, 计算 K0, G0, P
    auto pxy = hcs_toolbox::lpc2hpc(A_xy, B_xy, Kxy_lin);

    double mu_xy = -1.0;  // μ = -1: 滑模类行为, hn^(1+μ) = hn^0 = 1
    Eigen::Matrix<double, 8, 8> Gd_xy =
        Eigen::Matrix<double, 8, 8>::Identity() + mu_xy * pxy.G0;
    Eigen::Matrix<double, 2, 8> Kxy_nl = Kxy_lin - pxy.K0;

    std::cout << "  mu = " << mu_xy
              << ", 容许范围 [" << pxy.mu_min << ", " << pxy.mu_max << "]\n";
    std::cout << "  Gd_xy 特征值: [" << Gd_xy.eigenvalues().real().transpose() << "]\n";

    // --- ZOH 离散化 ---
    Eigen::Matrix<double, 8, 8> Ah_xy;
    Eigen::Matrix<double, 8, 2> Bh_xy;
    {
        Eigen::MatrixXd tmpA, tmpB;
        hcs_toolbox::ZOH(DT, A_xy, B_xy, tmpA, tmpB);
        Ah_xy = tmpA;
        Bh_xy = tmpB;
    }

    // --- 阶跃响应仿真 ---
    int Nxy = static_cast<int>(8.0 / DT);  // 仿真 8 秒
    Eigen::Matrix<double, 8, 1> x0_xy;
    // 初始: X位置误差 1m, Y位置误差 0.5m, 其余为零
    x0_xy << 1.0, 0.5, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0;

    auto Xxy_hpc = sim_xy_hpc(x0_xy, Nxy, Ah_xy, Bh_xy,
                              pxy.K0, Kxy_nl, Gd_xy, mu_xy, pxy.P);
    auto Xxy_lin = sim_xy_linear(x0_xy, Nxy, Ah_xy, Bh_xy, Kxy_lin);

    std::cout << "  ||x(T)||  HPC = " << Xxy_hpc.col(Nxy).norm()
              << "  (线性 LQR: " << Xxy_lin.col(Nxy).norm() << ")\n\n";

    // ========================================================================
    // 输出 CSV 数据文件 (降采样 10 倍以减小文件体积)
    // ========================================================================
    std::cout << ">>> 写入 CSV 数据文件...\n";
    int ds = 10;  // 降采样因子: 每 10 步保存 1 个点 (100Hz)

    // --- Z 轴 CSV ---
    {
        int N = Nz / ds;
        Eigen::VectorXd t(N);
        Eigen::MatrixXd Xh(2, N), Xl(2, N);
        for (int i = 0; i < N; ++i) {
            t(i) = i * ds * DT;
            Xh.col(i) = Xz_hpic.col(i * ds);
            Xl.col(i) = Xz_lin.col(i * ds);
        }
        write_csv("uav_z_comparison_cpp.csv", t, Xh, Xl,
                  {"z_hpic", "vz_hpic", "z_lin", "vz_lin"});
    }

    // --- Yaw 轴 CSV ---
    {
        int N = Ny / ds;
        Eigen::VectorXd t(N);
        Eigen::MatrixXd Xh(2, N), Xl(2, N);
        for (int i = 0; i < N; ++i) {
            t(i) = i * ds * DT;
            Xh.col(i) = Xy_hpc.col(i * ds);
            Xl.col(i) = Xy_lin.col(i * ds);
        }
        write_csv("uav_yaw_comparison_cpp.csv", t, Xh, Xl,
                  {"yaw_hpc", "wz_hpc", "yaw_lin", "wz_lin"});
    }

    // --- XY 轴 CSV ---
    {
        int N = Nxy / ds;
        Eigen::VectorXd t(N);
        Eigen::MatrixXd Xh(8, N), Xl(8, N);
        for (int i = 0; i < N; ++i) {
            t(i) = i * ds * DT;
            Xh.col(i) = Xxy_hpc.col(i * ds);
            Xl.col(i) = Xxy_lin.col(i * ds);
        }
        write_csv("uav_xy_comparison_cpp.csv", t, Xh, Xl,
                  {"ex_hpc", "ey_hpc", "vx_hpc", "vy_hpc",
                   "th_hpc", "ph_hpc", "q_hpc",  "p_hpc",
                   "ex_lin", "ey_lin", "vx_lin", "vy_lin",
                   "th_lin", "ph_lin", "q_lin",  "p_lin"});
    }

    std::cout << "  完成。输出文件: uav_{z,yaw,xy}_comparison_cpp.csv\n";
    std::cout << "  绘图命令: python3 ../scripts/plot_uav_comparison.py\n\n";

    // ========================================================================
    // 结果汇总
    // ========================================================================
    std::cout << "============================================================\n";
    std::cout << "仿真结果汇总 (线性 vs 齐次 对比)\n";
    std::cout << "============================================================\n";

    double ez_hpc = Xz_hpic.col(Nz).norm();
    double ez_lin = Xz_lin.col(Nz).norm();
    std::cout << "  Z   回路 (HPIC, μ=" << mu_z << "):  "
              << "||x(T)|| = " << ez_hpc
              << "  (线性 PI: " << ez_lin
              << ", 提升 " << ez_lin / ez_hpc << "×)\n";

    double ey_hpc = Xy_hpc.col(Ny).norm();
    double ey_lin = Xy_lin.col(Ny).norm();
    std::cout << "  Yaw 回路 (HPC,  μ=" << mu_yaw << "): "
              << "||x(T)|| = " << ey_hpc
              << "  (线性 PD: " << ey_lin << ")\n";

    double exy_hpc = Xxy_hpc.col(Nxy).norm();
    double exy_lin = Xxy_lin.col(Nxy).norm();
    std::cout << "  XY  回路 (HPC,  μ=" << mu_xy << "): "
              << "||x(T)|| = " << exy_hpc
              << "  (线性 LQR: " << exy_lin;
    if (exy_hpc < 1e-10)
        std::cout << ", 有限时间收敛至机器零!";
    std::cout << ")\n\n";

    // --- 关键参数汇总 ---
    std::cout << "关键齐次控制器参数:\n";
    std::cout << "  Z:   Ki_new  = [" << pz.Ki_new(0) << ", " << pz.Ki_new(1) << "]\n";
    std::cout << "       Gd 特征值 = [" << Gd_z.eigenvalues().real().transpose() << "]\n";
    std::cout << "  Yaw: Gd 特征值 = [" << Gd_yaw.eigenvalues().real().transpose() << "]\n";
    std::cout << "  XY:  Gd 特征值 = [" << Gd_xy.eigenvalues().real().transpose() << "]\n";

    return 0;
}
