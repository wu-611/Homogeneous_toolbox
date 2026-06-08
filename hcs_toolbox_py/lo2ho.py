"""
线性观测器 → 齐次观测器 升级函数
==================================

原始 MATLAB: lo2ho.m (Polyakov, HCS Toolbox ver 0.2)

原理 —— 对偶性 (Duality):
    在线性系统理论中，控制器设计和观测器设计是对偶问题：
        控制器: 设计 K 使 A+BK 稳定
        观测器: 设计 L 使 A+LC 稳定
        对偶关系: (A, B) 的控制器 ↔ (A', B') 的观测器

    Polyakov 将此对偶性扩展到齐次框架:
        lo2ho(A, C, L) ≡ lpc2hpc(A', C', L') 的转置

数学公式:
    线性观测器:
        dž/dt = A·ž + f + L·(C·ž - y)
        误差 ė = (A+LC)·e  → 指数收敛

    齐次观测器:
        dž/dt = A·ž + f + [L₀ + |Cž-y|^(ν-1)·d(log|Cž-y|)·(L-L₀)]·(Cž-y)
        误差 ė = [A+L₀C + |Ce|^ν·d(log|Ce|)·(L-L₀)C]·e
        齐次度 = ν

    其中:
        ν < 0: 观测误差有限时间收敛到零
        ν = 0: 退化为线性 Luenberger 观测器
        ν > 0: 近固定时间收敛

关键转换:
    L0_obs = K0_ctl'           （线性分量增益的转置）
    G0_obs = -G0_ctl'          （膨胀生成元取负号）
    ν_obs  = -μ_ctl            （齐次度取反号）

为什么 G0 取负号:
    控制器膨胀: Gd = I + μ·G0_ctl
    观测器膨胀: Gd = I + ν·G0_obs
    对偶关系要求 G0_obs = -G0_ctl'
    因为 lpc2hpc(A', C', L') 处理的是对偶系统 (A', C')，
    其膨胀生成元与原始系统差一个负号。

为什么 ν = -μ:
    控制器的齐次度 μ 定义: f(d(s)x) = e^{μs}·d(s)·f(x)
    观测误差动力学是"收缩"的（衰减到零），而控制动力学是"推动"的。
    对偶系统中的齐次度符号相反。
"""

import numpy as np
from .lpc2hpc import lpc2hpc


def lo2ho(A, C, L):
    """
    将线性 Luenberger 观测器升级为齐次观测器 (HO)。

    参数:
        A: 系统矩阵 (n×n)
        C: 输出矩阵 (k×n)，满行秩
        L: Luenberger 观测器增益 (n×k)，需满足 A+LC 为 Hurwitz

    返回:
        L0:     齐次化观测器增益 (n×k) — 线性分量
                满足 A+L0·C 在"最后一块行"幂零
        G0:     膨胀生成元基础矩阵 (n×n)
                Gd = I + ν·G0
        P:      形状矩阵 (n×n)，定义齐次范数的加权欧氏范数
        nu_min: 最小容许齐次度（最负值 → 最快有限时间收敛）
        nu_max: 最大容许齐次度

    使用示例:
        # 双积分器 + 只测位置
        A = [[0, 1], [0, 0]]
        C = [[1, 0]]
        L = [[-7], [-12]]      # 极点配置在 -3, -4

        L0, G0, P, nu_min, nu_max = lo2ho(A, C, L)
        nu = nu_min             # 取最负值 → 有限时间收敛
        Gd = np.eye(2) + nu * G0
        L_nl = L - L0

        # 在线运行
        z = e_ho(dt, z, y, A, C, B*u, L0, L_nl, Gd, nu, alpha=0.1)
    """
    # 核心：在对偶系统上调用 lpc2hpc，然后转置回来
    # lpc2hpc(A', C', L') 将 (A', C') 视为控制系统的 (A, B)
    K0, G0, P, nu_max, nu_min = lpc2hpc(A.T, C.T, L.T)

    # 转置 + 符号调整（对偶性要求）
    return K0.T, -G0.T, P, -nu_min, -nu_max
