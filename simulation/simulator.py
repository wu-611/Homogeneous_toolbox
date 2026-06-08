"""
通用仿真循环框架
================

为四旋翼控制器提供统一的仿真运行环境。

支持模式:
    1. 全状态反馈: controller(t, state) → u
    2. 输出反馈:   controller(t, measured_state) → u
    3. 轨迹跟踪:   配合 ref_traj(t) → (pos_d, vel_d, R_d, ω_d) 使用

测量噪声:
    通过 noise_std 参数在位置测量上叠加高斯白噪声，
    用于测试输出反馈方案对传感器噪声的鲁棒性。
"""

import numpy as np
import time


class Simulator:
    """
    通用仿真循环

    使用示例:
        sim = Simulator(model, dt=0.001, t_max=10.0)
        result = sim.run(controller, state0, noise_std=0.01)
        # result['t'] — 时间网格
        # result['state'] — 状态轨迹 (N+1, 18)
        # result['input'] — 控制输入轨迹 (N+1, 4)
    """

    def __init__(self, model, dt=0.001, t_max=10.0):
        """
        参数:
            model: QuadrotorSE3 四旋翼动力学模型
            dt:    仿真步长 [s]（默认 0.001s = 1kHz）
            t_max: 仿真总时长 [s]
        """
        self.model = model
        self.dt = dt
        self.t_max = t_max
        self.N = int(t_max / dt)       # 总步数
        self.reset()

    def reset(self):
        """清空日志数据（每次 run() 前自动调用）。"""
        self.state_log = None
        self.input_log = None
        self.ref_log = None
        self.time_grid = None

    def run(self, controller, state0, ref_traj=None, noise_std=0.0):
        """
        运行仿真。

        仿真循环（每步）:
            1. 测量（可选噪声）: 对位置叠加 N(0, noise_std²) 噪声
            2. 控制计算:         调用 controller 获取 [thrust, τx, τy, τz]
            3. 状态积分:         RK4 一步
            4. 日志记录:         保存状态、控制量、参考轨迹

        参数:
            controller: callable
                全状态模式:  controller(t, state) → u
                轨迹模式:    controller(t, state, pos_d, vel_d, R_d, ω_d) → u
            state0: (18,) ndarray
                初始 SE(3) 状态
            ref_traj: callable or None
                参考轨迹生成器 ref_traj(t) → (pos_d, vel_d, R_d, omega_d)
                None 表示悬停/阶跃场景
            noise_std: float
                位置测量噪声标准差 [m]（0 = 无噪声）

        返回:
            result: dict, 包含:
                't':     时间网格 (N+1,)
                'state': 状态轨迹 (N+1, 18)
                'input': 控制输入轨迹 (N+1, 4)
                'ref':   参考轨迹 (N+1, 18)，仅当 ref_traj 不为 None
        """
        self.reset()
        state = state0.copy()

        n_state = len(state0)
        n_input = 4

        # 预分配日志数组（避免动态扩展开销）
        state_log = np.zeros((self.N + 1, n_state))
        input_log = np.zeros((self.N + 1, n_input))
        ref_log = np.zeros((self.N + 1, 18)) if ref_traj is not None else None
        time_grid = np.linspace(0, self.t_max, self.N + 1)

        state_log[0] = state

        for k in range(self.N):
            t = time_grid[k]

            # ---- 测量（可选噪声）----
            # 只对位置添加噪声（模拟 GPS/动作捕捉的测量误差）
            if noise_std > 0:
                measured_state = state.copy()
                measured_state[0:3] += noise_std * np.random.randn(3)
            else:
                measured_state = state

            # ---- 控制计算 ----
            if ref_traj is not None:
                # 轨迹跟踪模式：将参考状态传递给控制器
                pos_d, vel_d, R_d, omega_d = ref_traj(t)
                u = controller(t, measured_state, pos_d, vel_d, R_d, omega_d)
                ref_log[k] = self.model.make_state(pos_d, vel_d, R_d, omega_d)
            else:
                # 悬停/阶跃模式：控制器自行处理
                u = controller(t, measured_state)

            input_log[k] = u

            # ---- 状态积分 (RK4) ----
            state = self.model.step_rk4(state, u, self.dt)
            state_log[k + 1] = state

        # 记录末尾参考轨迹（用于误差计算）
        if ref_traj is not None:
            pos_d, vel_d, R_d, omega_d = ref_traj(self.t_max)
            ref_log[self.N] = self.model.make_state(pos_d, vel_d, R_d, omega_d)

        # 最后的控制输入（仅用于日志，未实际施加）
        input_log[self.N] = input_log[self.N - 1]

        self.state_log = state_log
        self.input_log = input_log
        self.ref_log = ref_log
        self.time_grid = time_grid

        return {
            't': time_grid,
            'state': state_log,
            'input': input_log,
            'ref': ref_log
        }

    def run_discrete_time(self, model_discrete, controller, state0, n_steps=None):
        """
        使用离散时间动力学模型运行仿真。

        与 run() 的区别:
            - 不使用 RK4，而是使用预离散化的模型 model_discrete(state, u)
            - 适用于 ZOH 离散化后的线性系统仿真

        参数:
            model_discrete: callable, 离散动力学 model_discrete(state, u) → next_state
            controller:      callable, 控制器 controller(k, state) → u
            state0:          初始状态
            n_steps:         仿真步数（默认 N）

        返回:
            result: dict, 包含 't' 和 'state'
        """
        self.reset()
        if n_steps is None:
            n_steps = self.N

        state = state0.copy()
        n_state = len(state0)

        state_log = np.zeros((n_steps + 1, n_state))
        state_log[0] = state

        for k in range(n_steps):
            u = controller(k, state)
            state = model_discrete(state, u)
            state_log[k + 1] = state

        self.state_log = state_log
        self.time_grid = np.linspace(0, n_steps * self.dt, n_steps + 1)

        return {
            't': self.time_grid,
            'state': state_log,
        }
