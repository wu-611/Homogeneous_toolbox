"""
位置回路齐次观测器 (Position HO) 设计模块
==========================================

功能：为四旋翼位置通道（双积分器）设计齐次观测器，
     仅使用位置测量估计速度。

数学模型：
    每个通道 (x, y, z) 是独立的双积分器：
        A = [[0, 1], [0, 0]]      # 位置→速度→加速度
        C = [[1, 0]]               # 只测位置

    线性 Luenberger 观测器：
        dz/dt = A z + B u + L (C z - y)

    齐次观测器（对偶 lpc2hpc）：
        lo2ho(A, C, L_lin) → L0, G0, P, nu_min, nu_max

关键概念：
    - ν < 0: 观测误差有限时间收敛到零
    - ν = 0: 退化为线性 Luenberger 观测器
    - 观测器极点需远快于控制器（|ν| > |μ_p| 推荐）

使用示例：
    ho = PositionHO(m=1.4, nu=-0.5)
    z = np.array([0.0, 0.0])  # [e_pos_hat, e_vel_hat]
    for each timestep:
        y = measured_position_error
        u_ctrl = last_position_hpc_output  # 或加速度前馈
        z = ho.update(z, y, u_ctrl, dt=0.001)
        e_vel_hat = z[1]
"""

import numpy as np
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
from hcs_toolbox_py import lo2ho, e_ho


class PositionHO:
    """
    位置回路齐次观测器（每个通道独立，共 3 个实例）。

    处理流程：
        1. lo2ho(A, C, L_lin) → L0, G0, P, nu_min, nu_max
        2. Gd = I + nu * G0
        3. L_nl = L_lin - L0
        4. 在线更新: z ← e_ho(dt, z, y, A, C, B*u_ctrl, L0, L_nl, Gd, nu)
    """

    def __init__(self, m, L_linear=None, nu=None):
        """
        参数:
            m: 飞行器质量 [kg]
            L_linear: Luenberger 观测器增益 (2×1)
                      默认：极点配置在 s=-8, -10
                      → 特征方程 (s+8)(s+10) = s² + 18s + 80 = 0
                      → L = [18, 80]'（双积分器，C=[1,0]）
            nu: 齐次度
                nu < 0: 观测误差有限时间收敛
                nu = 0: 退化为线性 Luenberger（推荐用于基准对比）
                None:  取 nu_min（最负容许值）
        """
        self.m = m

        # 双积分器模型
        self.A = np.array([[0., 1.], [0., 0.]])
        self.B = np.array([[0.], [1. / m]])   # 控制→加速度的映射
        self.C = np.array([[1., 0.]])          # 只测位置

        # 线性观测器增益
        if L_linear is None:
            # 极点 s=-8, -10 → 远快于控制器极点 s=-2, -3
            # 保证观测器先收敛（时间尺度分离）
            L_linear = np.array([[-18.0], [-80.0]])
        self.L_linear = np.asarray(L_linear).reshape(2, 1)

        # 升级为齐次观测器
        self._design(nu)

    def _design(self, nu):
        """
        核心升级步骤：
        1. lo2ho → L0, G0, P, nu_min, nu_max
        2. 检查 nu 容许性
        3. 构造 Gd = I + nu * G0
        4. 计算非线性增益 L_nl = L_linear - L0
        """
        # 步骤1：对偶升级
        L0, G0, P, nu_min, nu_max = lo2ho(self.A, self.C, self.L_linear)

        self.L0 = L0
        self.G0 = G0
        self.P = P
        self.nu_min = nu_min
        self.nu_max = nu_max

        # 步骤2：检查 nu 在容许范围内
        if nu is None:
            self.nu = nu_min  # 最负 → 最快有限时间估计
        else:
            self.nu = float(nu)
            if self.nu < nu_min - 1e-6 or self.nu > nu_max + 1e-6:
                print(f"Warning: nu={self.nu} 超出容许范围 "
                      f"[{nu_min:.4f}, {nu_max:.4f}]，已裁剪")
            self.nu = np.clip(self.nu, nu_min + 1e-6, nu_max - 1e-6)

        # 步骤3：膨胀生成元
        if abs(self.nu) < 1e-10:
            self.Gd = np.eye(2)
        else:
            self.Gd = np.eye(2) + self.nu * self.G0

        # 步骤4：非线性增益
        self.L_nl = self.L_linear - self.L0

        # 记录参数
        self._valid = True

    def update(self, z, y, u_ctrl, dt=0.001):
        """
        更新观测器状态一步。

        参数:
            z: 当前状态估计 [e_pos_hat, e_vel_hat] (2,)
            y: 测量输出（位置误差，标量）
            u_ctrl: 已知控制输入（加速度指令，标量）[m/s²]
            dt: 采样周期 [s]

        返回:
            z_new: 更新后的状态估计 (2,)

        齐次观测器动力学：
            dz/dt = A z + B u_ctrl + [L0 + |Cz-y|^ν·d(log|Cz-y|)·L_nl]·(Cz-y)

            e_ho 离散化：
            z_new = (I + dt*(A + S*C))z + dt*B*u_ctrl - dt*S*y
            S = L0 + expm(ln|Cz-y| * (ν*I + Gd - I)) * L_nl
        """
        # 计算前馈项: f = B * u_ctrl（已知控制输入的影响）
        f = (self.B * u_ctrl).flatten()

        # 调用 e_ho（工具箱函数）
        z_new = e_ho(
            h=dt, z=z, y=np.atleast_1d(y),
            A=self.A, C=self.C, f=f,
            L0=self.L0, L=self.L_nl,
            Gd=self.Gd, nu=self.nu,
            alpha=0.1,    # 防近零点颤振（与 HPC 一致）
            beta=100.0    # 防远点衰减过度
        )
        return z_new

    def reset(self):
        """重置观测器状态为零初始猜测。"""
        return np.zeros(2)

    def reset_with_guess(self, e_pos, e_vel_guess=0.0):
        """用位置测量和速度猜测初始化观测器。"""
        return np.array([e_pos, e_vel_guess])
