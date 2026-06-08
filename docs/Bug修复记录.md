# Bug 修复记录 — SE(3) 几何齐次跟踪控制

> 日期：2026-06-08
> 记录从 Phase 0 实现过程中遇到的所有 Bug、根因分析和修复方案。

---

## Bug 目录

| # | Bug 名称 | 严重度 | 状态 |
|:-:|---------|:-----:|:----:|
| 1 | [推力单位不匹配导致 HPC 发散](#bug-1-推力单位不匹配导致-hpc-发散) | 🔴 致命 | ✅ 已修复 |
| 2 | [姿态增益与力矩映射的缩放失配](#bug-2-姿态增益与力矩映射的缩放失配) | 🔴 致命 | ✅ 已修复 |
| 3 | [hnorm 数值溢出导致 SVD 不收敛](#bug-3-hnorm-数值溢出导致-svd-不收敛) | 🔴 致命 | ✅ 已修复 |
| 4 | [hnorm 重写时循环条件反转](#bug-4-hnorm-重写时循环条件反转) | 🟡 中等 | ✅ 已修复 |
| 5 | [Simulator 回调参数不匹配](#bug-5-simulator-回调参数不匹配) | 🟡 中等 | ✅ 已修复 |
| 6 | [matplotlib 版本冲突导致 3D 图不可用](#bug-6-matplotlib-版本冲突导致-3d-图不可用) | 🟢 轻微 | ✅ 已绕过 |
| 7 | [旋转倒立摆验证脚本数值发散](#bug-7-旋转倒立摆验证脚本数值发散) | 🟡 中等 | ✅ 已修复 |
| 8 | [HPC μ=0 阶跃响应初始力矩过小](#bug-8-hpc-μ0-阶跃响应初始力矩过小) | 🟡 中等 | ✅ 已调优 |

---

## Bug 1: 推力单位不匹配导致 HPC 发散

**发现时间**：2026-06-08，首次运行 `run_phase1_simulations.py`

**现象**：
```
HPC mu=0:  ||e_pos|| = 560149200262325921890337328827639151698193557928046559593652084015104.0000 m
```
位置误差爆炸至 ~10^87 米，随后 SVD 不收敛导致程序崩溃。

**根因分析**：

位置 HPC 的 `compute_control_vector()` 输出的是**加速度**（单位 m/s²），而非力（单位 N）。

原代码在推力计算处：
```python
# 错误代码
F_des = u_pos + np.array([0., 0., self.g])  # 这是比力 [N/kg], 不是力!
thrust = np.dot(F_des, R @ np.array([0., 0., 1.]))  # 推力值是比力而非牛顿
```

动力学方程中：
```python
dvel = np.array([0., 0., -self.g]) + (thrust / self.m) * (R @ e3)
```

这里 `thrust / self.m` 期望 `thrust` 是力 [N]。如果 `thrust` 实际上是比力 [N/kg]，则 `thrust/m` 的单位是 m/s²/kg，比正确值小 m 倍。对于 m=1.4：
- 悬停需推力 ≈ 13.7 N
- 错误代码给出 thrust ≈ 9.81（比力）
- 实际加速度贡献 = 9.81/1.4 = 7.0 m/s²，不足以抵消重力 (9.81 m/s²)
- 旋翼缓慢下降 → 位置误差累积 → 更大的加速度命令 → 姿态误差 → 发散

**修复方案**：

```python
# 正确代码
u_pos = self.pos_hpc.compute_control_vector(e_pos, e_vel)
F_des_force = self.m * (u_pos + np.array([0., 0., self.g]))  # 乘以质量转为力
thrust = np.dot(F_des_force, R @ np.array([0., 0., 1.]))  # 力投影到体轴
```

同时在 `se3_homogeneous_full.py` 第 140 行添加了详细的量纲注释。

**验证**：修复后 HPC μ=0 悬停推力 ≈ 13.73 N（= m·g），与 Lee PD 一致。

---

## Bug 2: 姿态增益与力矩映射的缩放失配

**发现时间**：2026-06-08，修复 Bug 1 后 HPC 仍发散

**现象**：

Z 轴测试正常（无姿态耦合），但 X/Y 偏移测试中 HPC 的初始力矩仅为 Lee PD 的 **1/50**：
```
Lee PD  tau_y = -6.33 N·m  (kR=12 N·m/rad)
HPC     tau_y = -0.026 N·m (K1=12 rad/s² per rad)
```

**根因分析**：

Zhou 2023 的姿态 HPC 将控制量 `u_hom` 定义为**角加速度**（rad/s²），而非力矩（N·m）。力矩通过 `M = J * u_hom` 映射。

- Lee PD：`M = -kR * θ_e = -12 * θ_e`（kR 直接是力矩增益）
- HPC：`M = J * (-K1 * θ_e) = -0.02 * 12 * θ_e = -0.24 * θ_e`（K1 是角加速度增益）

同样的数值 12，Lee PD 产生 12 N·m/rad 的力矩，HPC 仅产生 ~0.24 N·m/rad。差了约 50 倍（J ≈ 0.02）。

**设计层面的根本原因**：

Zhou 2023 论文的四旋翼参数为 `J = 0.0115·I₃`，而我们的模型 `J = diag([0.0211, 0.0219, 0.0366])`。即使使用相同的 K1=12，由于惯量差异和任务需求不同，有效力矩也不同。Zhou 论文设定了 ±0.5 N·m 的力矩限制，在 π 弧度姿态误差时恰好饱和——这是一个精心设计的"软"控制器。我们的应用场景需要更强的控制力矩。

**修复方案**：

将 HPC 姿态增益从 K1=12 增加到 K1=200，使有效力矩增益 (J·K1 ≈ [4.2, 4.4, 7.3] N·m/rad) 与 Lee PD (kR=4 N·m/rad) 可比：

```python
# 修复前
if K1 is None:
    self.K1 = 12.0 * np.eye(3)  # Zhou 2023 原始值，对 1.4kg 四旋翼过弱
if k2 is None:
    self.k2 = 6.0

# 修复后
if K1 is None:
    self.K1 = 200.0 * np.eye(3)  # 有效力矩增益 ≈ 4 N·m/rad
if k2 is None:
    self.k2 = 100.0
```

同时同步降低了 Lee PD 的增益（kx: 16→8.4, kR: 12→4）以保持公平对比。

**调优过程中的参数演变**：

| 迭代 | K1 | k2 | Lee PD kR | 结果 |
|:----:|:--:|:--:|:---------:|------|
| 1 | 12 | 6 | 12 | HPC 发散 |
| 2 | ~569 (自动缩放) | ~241 | 12 | hnorm 数值问题 |
| 3 | 100 | 50 | 12 | HPC 力矩为 Lee PD 的 25% |
| 4 | 500 | 250 | 12 | hnorm P 矩阵过于不平衡 |
| 5 | **200** | **100** | **4** | ✅ 正常 |

**教训**：
- 齐次控制器的增益不能从线性 PD 直接照搬——两者有不同的物理单位（角加速度 vs 力矩）
- G₀ 和 Gd 的结构（加权膨胀 vs 规范膨胀）决定了增益到力矩的映射关系
- 数值稳定性（hnorm 中的 P 矩阵条件数）对增益选择施加额外约束

---

## Bug 3: hnorm 数值溢出导致 SVD 不收敛

**发现时间**：2026-06-08，大初始状态误差仿真时

**现象**：
```
RuntimeWarning: overflow encountered in matmul
numpy.linalg.LinAlgError: SVD did not converge
```

**根因分析**：

轨迹链：
1. 大位置误差（1m）→ 大 u_pos（~17 m/s²）→ 大 F_des
2. → 大姿态误差（~0.86 rad）→ 状态 ξ = [θ_e; ω_e] 数值大
3. → `hnorm(ξ, Gd, P)` 中 `expm(-Gd * a)` 对大 s 值溢出
4. → P-norm 计算出现 Inf/NaN → 控制量 NaN → 状态 NaN
5. → RK4 下一步 SVD 旋转矩阵遇到 NaN → 崩溃

具体触发条件：当 `a`（二分法下界）非常负时（如 a < -100），`expm(-Gd * a)` 需要计算大矩阵指数，导致 double 精度溢出。

**修复方案**：

在 `hnorm.py` 中增加多层防护：

```python
# 1. 入口保护：极大/极小输入直接返回
if nrm > 1e100:
    return float('inf')
if nrm < 1e-100:
    return 0.0

# 2. 循环保护：expm 包裹 try/except
def safe_eval(s):
    try:
        y = expm(-Gd * s) @ x
        v = y @ P @ y
        if np.isnan(v) or np.isinf(v):
            return None, None
        return y, v
    except Exception:
        return None, None

# 3. 边界保护：限制二分法搜索范围
if a < -746:  # exp(746) ≈ double max
    a = -746
    break
if b > 710:
    return 0.0
```

此外，在 `se3_homogeneous_full.py` 中增加控制量饱和：
```python
thrust = np.clip(thrust, 0.1, 80.0)
M = np.clip(M, -20.0, 20.0)
```

**验证**：修复后 1m 初始位置误差的阶跃仿真可正常运行至结束。

---

## Bug 4: hnorm 重写时循环条件反转

**发现时间**：2026-06-08，对 hnorm.py 增加数值保护后

**现象**：

修复 Bug 3 后，`hnorm` 对某些输入返回错误结果：
```
x=[1.0, 0]: nx=1.56,  sqrt(x'Px)=1.56  ← μ=0 时两者应相等，这个看起来对
x=[0.1, 0]: nx=0.016, sqrt(x'Px)=0.156  ← 不正确！相差 10x
```

**根因分析**：

在修复 Bug 3 时，不慎反转了二分法的循环条件。

原始 MATLAB 逻辑（正确）：
```matlab
while (y'*P*y < 1.0) && (a > -746)
    a = a * 2.0;   % 当 y'Py < 1 时继续（向量太小，需更多膨胀）
end
```

第一次重写（错误）：
```python
for i in range(max_iter):
    if val < 1.0:
        break       # ❌ 当 val < 1 时退出（与原始逻辑相反！）
    a = a * 2.0
```

这导致当 `val < 1` 时立即退出下界搜索，而非继续寻找更负的 a。对于小范数输入，二分法的区间 [a, b] 完全不正确。

**修复方案**：

回退到正确的原始逻辑结构，仅添加数值保护：

```python
# 正确逻辑
while val < 1.0 and a > -746:
    a = a * 2.0
    _, val2 = safe_eval(a)
    if val2 is None:
        a = a / 2.0
        break
    val = val2
```

**教训**：
- 重写已有算法的循环条件时，必须逐行对比原始代码
- MATLAB `while (condition)` 和 Python `while condition:` 语义相同，但如果在 Python 中使用 `for` + `break` 模拟 `while`，条件需要取反，容易出错
- **直接使用 `while` 循环而非 `for` + `break` 更安全**（最终版本采用此方式）

---

## Bug 5: Simulator 回调参数不匹配

**发现时间**：2026-06-08，首次运行螺旋轨迹跟踪

**现象**：
```
TypeError: run_trajectory_tracking.<locals>.ctrl() takes 2 positional arguments
but 6 were given
```

**根因分析**：

`Simulator.run()` 在提供 `ref_traj` 时会传递额外参数：
```python
if ref_traj is not None:
    pos_d, vel_d, R_d, omega_d = ref_traj(t)
    u = controller(t, measured_state, pos_d, vel_d, R_d, omega_d)
```

但轨迹跟踪的回调函数 `ctrl(t, state)` 只接受 2 个参数（时间和状态），因为轨迹参考在函数内部计算。

**修复方案**：

在回调函数中添加 `*args` 吸收额外参数：
```python
# 修复前
def ctrl(t, state):
    ...

# 修复后
def ctrl(t, state, *args):
    ...
```

**教训**：
- 仿真器回调接口的设计需要统一。当前设计混合了"内置轨迹"（ctrl 内部计算 ref）和"外部轨迹"（simulator 传递 ref）两种模式
- 长期应统一为一种模式，或使用 `**kwargs` 传递可选参数

---

## Bug 6: matplotlib 版本冲突导致 3D 图不可用

**发现时间**：2026-06-08，首次导入 visualization.plotter

**现象**：
```
UserWarning: Unable to import Axes3D. This may be due to multiple versions of
Matplotlib being installed
ImportError: cannot import name 'docstring' from 'matplotlib'
```

**根因分析**：

系统安装了两个版本的 matplotlib：
- 系统包：`/usr/lib/python3/dist-packages/matplotlib/`（旧版）
- pip 安装：`/home/wzy/.local/lib/python3.10/site-packages/matplotlib/`（新版）

`mpl_toolkits.mplot3d` 从系统路径加载了旧版的 `axes3d.py`，该文件尝试导入新版 matplotlib 中已重命名/删除的 `docstring` 符号。

**绕过方案**：

1. 移除 `from mpl_toolkits.mplot3d import Axes3D` 导入（该导入仅用于类型提示，非必需）
2. 在 `plot_trajectory_3d()` 中增加 `try/except`：如果 3D 投影不可用，回退到 2D 投影（XY + XZ 平面图）

**彻底修复**（未执行，需要用户确认）：
```bash
pip uninstall matplotlib && apt install python3-matplotlib
# 或
apt remove python3-matplotlib && pip install matplotlib==3.7.1
```

---

## Bug 7: 旋转倒立摆验证脚本数值发散

**发现时间**：2026-06-08，首次运行 `validate_lo2ho.py`

**现象**：
```
Final ||e||: HO=543035626149418666819584.000000
AssertionError: HO error too large
```

**根因分析**：

最初版本的 `validate_lo2ho.py` 使用旋转倒立摆模型（QUBE-Servo 2，4 阶不稳定系统）直接复现 MATLAB `demo_lo2ho.m`。开环仿真（无反馈控制）下，倒立摆是不稳定的，观测器误差会指数增长。

MATLAB demo 中使用 `u = Klin*z`（线性反馈）来稳定系统，但我们简化为 `u = 0`（开环）。

**修复方案**：

1. 改用双积分器模型作为主验证模型（与四旋翼位置通道直接对应，且是临界稳定而非指数不稳定）
2. 添加反馈控制（使用线性增益 Klin）稳定倒立摆测试
3. 保持所有数学性质验证（对偶性、幂零性、收敛速度）

**最终结果**：双积分器验证全部通过，HO 收敛速度 4.5 倍于线性观测器。

---

## Bug 8: HPC μ=0 阶跃响应初始力矩过小

**发现时间**：2026-06-08，参数调优过程中

**现象**：

X 偏移测试中，HPC μ=0 初始 τ_y = -1.55 N·m，而 Lee PD 为 -6.33 N·m。尽管 HPC 位置收敛更快（0.03m vs 0.22m @3s），但在更大的初始误差（1m）下，力矩不足会导致姿态收敛过慢，位置误差持续累积。

**根因分析**：

姿态 HPC 的增益 K1=100 时，有效力矩增益 J·K1 ≈ 2 N·m/rad。对于 1m X 位置误差产生的 ~0.7 rad 姿态误差，力矩仅 ~1.4 N·m。而 Lee PD 的 kR=4 直接产生 2.8 N·m。

另外，HPC 的齐次范数饱和（α=0.1, β=10）在状态较大时限制了增益放大效果。

**修复方案**：

增加 K1 至 200（有效力矩 ~4 N·m/rad），并将力矩限幅从 ±5 提高到 ±20 N·m。

---

## 总结

### Bug 分类统计

| 类别 | 数量 | 典型 Bug |
|------|:---:|---------|
| 单位/量纲错误 | 1 | 推力单位不匹配 |
| 参数缩放错误 | 2 | 姿态增益、初始力矩 |
| 数值稳定性 | 2 | hnorm 溢出、SVD 不收敛 |
| 逻辑错误 | 1 | hnorm 循环条件反转 |
| 接口不匹配 | 1 | 回调参数 |
| 环境问题 | 1 | matplotlib 版本冲突 |

### 重要教训

1. **物理单位必须逐行验证**：齐次控制框架中，控制器增益、虚拟控制、实际力/力矩之间的物理单位转换链条长且容易出错（加速度 → 比力 → 力 → 推力投影 → 牛顿第二定律）

2. **Zhou 2023 的增益不能直接套用**：论文的参数基于特定四旋翼（J≈0.01, 力矩限幅 ±0.5 N·m），换用不同参数的模型时必须重新缩放

3. **二分法算法重写陷阱**：`while (cond)` 在 Python 和 MATLAB 中语义相同，但改用 `for` + `break` 模拟时条件需要取反，极易出错

4. **先简后繁的测试策略**：先测试 Z 轴（无姿态耦合），再测试 X/Y（有姿态耦合）；先用小初始误差，再用大初始误差。这比直接跑全场景仿真高效得多
