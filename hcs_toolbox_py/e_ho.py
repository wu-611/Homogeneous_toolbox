"""
齐次观测器 (HO) 显式 Euler 离散化
==================================

原始 MATLAB: e_ho.m (Polyakov, HCS Toolbox ver 0.2)

数学模型:
    连续时间齐次观测器:
        dž/dt = A·ž + f + (L₀ + |Cž-y|^(ν-1)·d(log|Cž-y|)·(L-L₀))·(Cž-y)

    显式 Euler 离散化:
        z_{k+1} = (I + h·(A + S·C))·z_k + h·f - h·S·y

    其中非线性增益 S:
        nCe = |C·z_k - y|          （输出误差的欧氏模）
        S = L₀ + expm(ln(nCe)·(ν·I + Gd - I))·L

关键区别 —— 为什么观测器用 |Ce| 而非 ||x||_d:
    控制器用 ||x||_d（齐次范数）因为全状态 x 已知。
    观测器只能用 |Ce|（输出误差的标量模）因为真实状态 x 未知，
    只能看到测量误差 Cz - y = Ce。

    这意味着观测器不需要二分法求齐次范数！计算量比控制器小。

饱和处理 (alpha, beta):
    nCe = max(alpha, min(beta, |Cz-y|))
    - alpha ∈ (0, 1]: 下界，防止近零点增益过大 → 高频颤振
    - beta ∈ [1, ∞):  上界，防止远点增益过小 → 收敛缓慢

    α=β=1 时退化为线性 Luenberger 观测器（性能不差于线性）。
"""

import numpy as np
from scipy.linalg import expm


def e_ho(h, z, y, A, C, f, L0, L, Gd, nu, alpha=1e-6, beta=np.inf):
    """
    齐次观测器显式 Euler 一步更新。

    参数:
        h:     采样周期 [s]
        z:     当前状态估计 (n,)
        y:     测量输出 (k,)
        A:     系统矩阵 (n×n)
        C:     输出矩阵 (k×n)
        f:     已知外源输入信号 (n,)，例如 f = B·u
        L0:    齐次化观测器增益 (n×k) — 线性分量
        L:     非线性观测器增益 (n×k) = L_lin - L0
        Gd:    膨胀生成元 (n×n) = I + ν·G0
        nu:    齐次度（负值=有限时间，零=线性，正值=近固定时间）
        alpha: 输出误差下界（防颤振），默认 1e-6
        beta:  输出误差上界（防发散），默认 ∞

    返回:
        znew: 更新后的状态估计 (n,)

    计算步骤:
        1. 计算输出误差模: nCe = |C·z - y|
        2. 施加饱和:        nCe = sat_{alpha, beta}(nCe)
        3. 计算非线性增益:  S = L0 + expm(ln(nCe)·(ν·I + Gd - I))·L
        4. 显式 Euler 步:   znew = (I + h·(A + S·C))·z + h·f - h·S·y
    """
    z = np.asarray(z).flatten()
    y = np.asarray(y).flatten()
    f = np.asarray(f).flatten()

    # 步骤1: 输出误差的欧氏模
    nCe = np.linalg.norm(C @ z - y)

    # 步骤2: 饱和处理
    # 先上界再下界（匹配 MATLAB 逻辑）
    nCe = min(nCe, beta)    # 防过大 → 增益衰减
    nCe = max(nCe, alpha)   # 防过小 → 增益爆炸
    nCe = max(nCe, 1e-20)   # 防零 → ln(0) = -∞

    n = A.shape[0]
    I_n = np.eye(n)

    # 步骤3: 构造非线性增益 S
    # S = L0 + expm(ln(nCe) * (ν·I + Gd - I)) * L
    # 注意: 当 ν=0, Gd=I 时，expm(0) = I，S = L0+L = L_lin（退化为线性）
    S = L0 + expm(np.log(nCe) * (nu * I_n + Gd - I_n)) @ L

    # 步骤4: 显式 Euler 一步
    # (I + h·(A + S·C))·z + h·f - h·S·y
    znew = (I_n + h * (A + S @ C)) @ z + h * f - h * S @ y
    return znew
