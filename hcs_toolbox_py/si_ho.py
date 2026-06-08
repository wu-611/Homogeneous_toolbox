"""
齐次观测器 (HO) 半隐式离散化
============================

原始 MATLAB: si_ho.m (Polyakov, HCS Toolbox ver 0.2)

与 e_ho 的区别:
    e_ho (显式 Euler):
        z_{k+1} = (I + h·(A + S·C))·z_k + h·f - h·S·y

    si_ho (半隐式):
        z_{k+1} = (I - h·(A + S·C))⁻¹ · (z_k + h·f - h·S·y)

为什么需要半隐式:
    显式 Euler 对刚性问题（大特征值、振荡系统）不稳定。
    例如谐波振荡器 A=[[0,1],[-1,0]] 的特征值是 ±j，
    在显式 Euler 下步长稍大就会发散。

    半隐式通过矩阵求逆获得更好的数值稳定性:
        - 稳定域覆盖左半平面（A-stable）
        - 适用于振荡或不稳定系统

计算量:
    si_ho 需要每步求解 (I - h·(A+S·C))⁻¹（n×n 矩阵求逆）。
    对于低维系统（n=2~4）开销可接受。
    对于高维系统，可用迭代求解器替代直接求逆。

选择建议:
    - 稳定/弱不稳定系统: e_ho（速度快，无求逆）
    - 强不稳定/振荡系统: si_ho（更稳定，需要求逆）
    - 实际飞控: ZOH 离散化 + e_ho 或 si_ho
"""

import numpy as np
from scipy.linalg import expm, inv


def si_ho(h, z, y, A, C, f, L0, L, Gd, nu, alpha=1e-6, beta=np.inf):
    """
    齐次观测器半隐式一步更新。

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
        nu:    齐次度
        alpha: 输出误差下界（防颤振），默认 1e-6
        beta:  输出误差上界（防发散），默认 ∞

    返回:
        znew: 更新后的状态估计 (n,)

    计算步骤:
        1. 计算输出误差模: nCe = |C·z - y|
        2. 施加饱和:        nCe = sat_{alpha, beta}(nCe)
        3. 计算非线性增益:  S = L0 + expm(ln(nCe)·(ν·I + Gd - I))·L
        4. 半隐式步:
           znew = (I - h·(A+S·C))⁻¹ · (z + h·f - h·S·y)

    与 e_ho 的差异仅在步骤4:
        e_ho:  z_{k+1} = (I + h·M)·z_k + h·f - h·S·y
        si_ho: z_{k+1} = (I - h·M)⁻¹·(z_k + h·f - h·S·y)
        其中 M = A + S·C
    """
    z = np.asarray(z).flatten()
    y = np.asarray(y).flatten()
    f = np.asarray(f).flatten()

    # 步骤1: 输出误差的欧氏模
    nCe = np.linalg.norm(C @ z - y)

    # 步骤2: 饱和处理
    nCe = min(nCe, beta)
    nCe = max(nCe, alpha)
    nCe = max(nCe, 1e-20)

    n = A.shape[0]
    I_n = np.eye(n)

    # 步骤3: 构造非线性增益 S
    S = L0 + expm(np.log(nCe) * (nu * I_n + Gd - I_n)) @ L

    # 步骤4: 半隐式一步
    # 求解 (I - h·(A+S·C)) · z_{k+1} = z_k + h·f - h·S·y
    M = A + S @ C
    znew = inv(I_n - h * M) @ (z + h * f - h * S @ y)
    return znew
