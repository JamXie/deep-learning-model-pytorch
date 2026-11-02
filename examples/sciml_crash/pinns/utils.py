"""碰撞安全学习路径的共享工具函数。

对应文献框架：
  AI for PDEs in solid mechanics: A review (Wang et al., 2025)
  Sections 2.1, 3.2 — 配点采样与基准解
"""

from __future__ import annotations

import numpy as np
from scipy.integrate import solve_ivp


# ==============================================================================
# 数值 ODE 求解器 — 为弹簧-质量系统生成基准解
# ==============================================================================

def spring_mass_ode(t, y, m, c, k_func, F_func):
    """ODE 系统：m*x'' + c*x' + F_spring(x) = F_impact(t)。

    转换为状态空间形式：
        d/dt [x] = [x']
        d/dt [v] = (F_impact(t) - c*v - F_spring(x)) / m

    参数：
        t: 时间标量
        y: 状态向量 [x, v]
        m: 质量
        c: 阻尼系数
        k_func: 弹簧力函数 F_spring(x)，可为非线性
        F_func: 外力函数 F_impact(t)

    返回：
        [dx/dt, dv/dt]
    """
    x, v = y
    dxdt = v
    dvdt = (F_func(t) - c * v - k_func(x)) / m
    return [dxdt, dvdt]


def solve_spring_mass_rk45(m, c, k_func, F_func, x0, v0, t_span, n_points=500):
    """使用 RK45（Dormand-Prince）求解弹簧-质量 ODE。

    参数：
        m, c: 质量和阻尼
        k_func: 弹簧力函数 F_spring(x)，可为非线性
        F_func: 外力函数 F_impact(t)
        x0, v0: 初始位移和速度
        t_span: (t_start, t_end)
        n_points: 输出时间点数量

    返回：
        t: 时间数组
        x: 位移数组
        v: 速度数组
        a: 加速度数组 (= (F - c*v - F_spring(x)) / m)
    """
    t_eval = np.linspace(t_span[0], t_span[1], n_points)
    sol = solve_ivp(
        lambda t, y: spring_mass_ode(t, y, m, c, k_func, F_func),
        t_span=t_span,
        y0=[x0, v0],
        t_eval=t_eval,
        method="RK45",
        rtol=1e-8,
        atol=1e-10,
    )
    t = sol.t
    x = sol.y[0]
    v = sol.y[1]
    a = (F_func(t) - c * v - k_func(x)) / m
    return t, x, v, a


# ==============================================================================
# 弹簧力模型 — 从线性到碰撞相关的非线性
# ==============================================================================

def linear_spring(x, k=10.0):
    """线性弹簧：F = k * x。"""
    return k * x


def bilinear_spring(x, k1=10.0, k2=2.0, x_yield=0.05):
    """双线性弹塑性弹簧 — 模拟碰撞能量吸收。

    弹性区：F = k1 * x                                   当 |x| <= x_yield
    塑性区：F = k1*x_yield + k2*(|x|-x_yield) * sign(x)  当 |x| > x_yield

    这近似了典型的车辆碰撞力-位移关系：
      弹性斜坡 -> 塑性平台 -> 密实化
    """
    if np.isscalar(x):
        if abs(x) <= x_yield:
            return k1 * x
        return (k1 * x_yield + k2 * (abs(x) - x_yield)) * np.sign(x)

    x = np.asarray(x)
    elastic = np.abs(x) <= x_yield
    result = np.zeros_like(x)
    result[elastic] = k1 * x[elastic]
    plastic = ~elastic
    result[plastic] = (k1 * x_yield + k2 * (np.abs(x[plastic]) - x_yield)) * np.sign(x[plastic])
    return result


def multistage_spring(x, k1=15.0, k2=3.0, k3=30.0, x1=0.03, x2=0.12):
    """三阶段弹簧 — 更真实的碰撞力-位移关系。

    阶段1（弹性）：    F = k1 * x                 |x| <= x1
    阶段2（塑性平台）：F = F1 + k2 * (|x|-x1)      x1 < |x| <= x2
    阶段3（密实化）：  F = F2 + k3 * (|x|-x2)      |x| > x2

    典型车辆前部结构：保险杠 -> 碰撞盒 -> 纵梁。
    """
    if np.isscalar(x):
        abs_x = abs(x)
        if abs_x <= x1:
            f = k1 * abs_x
        elif abs_x <= x2:
            f = k1 * x1 + k2 * (abs_x - x1)
        else:
            f = k1 * x1 + k2 * (x2 - x1) + k3 * (abs_x - x2)
        return f * np.sign(x)

    x = np.asarray(x)
    abs_x = np.abs(x)
    result = np.zeros_like(x)
    s1 = abs_x <= x1
    s2 = (abs_x > x1) & (abs_x <= x2)
    s3 = abs_x > x2
    result[s1] = k1 * abs_x[s1]
    result[s2] = k1 * x1 + k2 * (abs_x[s2] - x1)
    result[s3] = k1 * x1 + k2 * (x2 - x1) + k3 * (abs_x[s3] - x2)
    return result * np.sign(x)


def rate_dependent_spring(x, v, k0=10.0, beta=0.1, n=1.0):
    """率相关弹簧 — 应变率敏感材料。

    F = k0 * x * (1 + beta * |v|^n)

    碰撞中常见：钢在高应变率下表现出强度增加。
    """
    return k0 * x * (1.0 + beta * np.abs(v) ** n)


# ==============================================================================
# 配点采样 — 综述论文第 3.2 节
# ==============================================================================

def sample_collocation_points(t_span, n_domain=200, n_ic=2, scheme="uniform"):
    """为 PINN 训练采样配点。

    参数：
        t_span: (t_start, t_end)
        n_domain: 域内点数
        n_ic: 初始条件点数
        scheme: 'uniform' / 'random' / 'latin_hypercube' / 'chebyshev'

    返回：
        t_domain: 域内配点 (n_domain, 1)
        t_ic: 初始条件点 (n_ic, 1)
    """
    if scheme == "uniform":
        t_domain = np.linspace(t_span[0], t_span[1], n_domain)
    elif scheme == "random":
        t_domain = np.random.uniform(t_span[0], t_span[1], n_domain)
        t_domain.sort()
    elif scheme == "chebyshev":
        # 切比雪夫节点 — 更适合多项式逼近
        k = np.arange(n_domain)
        t_domain = (t_span[0] + t_span[1]) / 2 + (t_span[1] - t_span[0]) / 2 * np.cos(
            (2 * k + 1) / (2 * n_domain) * np.pi
        )
        t_domain = np.sort(t_domain)
    elif scheme == "latin_hypercube":
        # 一维的简单拉丁超立方采样
        segments = np.linspace(t_span[0], t_span[1], n_domain + 1)
        t_domain = np.array([np.random.uniform(segments[i], segments[i + 1]) for i in range(n_domain)])
        t_domain.sort()
    else:
        raise ValueError(f"Unknown sampling scheme: {scheme}")

    t_ic = np.array([t_span[0]] * n_ic)

    return (
        t_domain.reshape(-1, 1).astype(np.float32),
        t_ic.reshape(-1, 1).astype(np.float32),
    )


# ==============================================================================
# 精度指标
# ==============================================================================

def compute_metrics(t_ref, x_ref, t_pred, x_pred):
    """在预测时间网格上对比参考解，返回 RMSE、相对误差与 R² 等指标。

    参数：
        t_ref, x_ref: 参考解（RK45）
        t_pred, x_pred: 模型预测

    返回：
        字典，包含插值后的参考解及各项误差指标
    """
    x_ref_interp = np.interp(t_pred, t_ref, x_ref)
    eps = 1e-8

    rmse = float(np.sqrt(np.mean((x_pred - x_ref_interp) ** 2)))
    max_err = float(np.max(np.abs(x_pred - x_ref_interp)))
    nrmse = rmse / (np.max(np.abs(x_ref_interp)) + eps)
    sst = float(np.sum((x_ref_interp - np.mean(x_ref_interp)) ** 2)) + eps
    r2 = float(1.0 - np.sum((x_pred - x_ref_interp) ** 2) / sst)

    peak_true = float(np.max(np.abs(x_ref_interp)))
    peak_pred = float(np.max(np.abs(x_pred)))
    peak_re = abs(peak_pred - peak_true) / (peak_true + eps)

    return {
        "x_ref_interp": x_ref_interp,
        "rmse": rmse,
        "max_err": max_err,
        "nrmse": nrmse,
        "accuracy": max(0.0, 1.0 - nrmse),
        "r2": r2,
        "peak_re": peak_re,
    }
