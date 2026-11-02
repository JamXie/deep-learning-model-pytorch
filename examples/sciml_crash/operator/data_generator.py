"""碰撞算子学习的训练数据生成器。

生成不同参数组合的弹簧-质量 ODE 解，为训练 DeepONet 建立数据集：
    G: (m, c, k1, k2, x_yield, v0) -> x(t)

每个样本 = 一个具有不同参数的碰撞场景；
目标输出 = 在固定时间点上的位移 x(t) 时间历程。

运行（在仓库根目录）：
    python -m examples.sciml_crash.operator.data_generator --num-samples 500
"""

from __future__ import annotations

import argparse
import pickle

import numpy as np
from scipy.integrate import solve_ivp

from common.cli import add_common_args
from common.paths import resolve_output_dir
from examples.sciml_crash.pinns.utils import bilinear_spring


def solve_crash_scenario(m, c, k1, k2, x_yield, v0, t_span, n_time_points=200):
    """求解一个碰撞场景的弹簧-质量 ODE。

    参数：
        m: 质量 [kg]
        c: 阻尼 [N*s/m]
        k1: 弹性刚度 [N/m]
        k2: 塑性刚度 [N/m]
        x_yield: 屈服位移 [m]
        v0: 初始速度 [m/s]（负值表示碰撞）
        t_span: (起始时间, 结束时间)
        n_time_points: 输出时间点数量

    返回：
        t: 时间数组 (n_time_points,)
        x: 位移数组 (n_time_points,)
        v: 速度数组 (n_time_points,)
        a: 加速度数组 (n_time_points,)
    """

    def F_impact(t):
        return 0.0

    def k_func(x):
        return bilinear_spring(x, k1=k1, k2=k2, x_yield=x_yield)

    def ode_rhs(t, y):
        x, v = y
        return [v, (F_impact(t) - c * v - k_func(x)) / m]

    t_eval = np.linspace(t_span[0], t_span[1], n_time_points)
    sol = solve_ivp(ode_rhs, t_span, [0.0, v0], t_eval=t_eval, method="RK45", rtol=1e-8, atol=1e-10)

    t = sol.t
    x = sol.y[0]
    v = sol.y[1]
    a = np.array([(F_impact(ti) - c * v[i] - k_func(x[i])) / m for i, ti in enumerate(t)])
    return t, x, v, a


def generate_crash_dataset(n_samples=500, n_time_points=200, t_span=(0.0, 0.12), seed=42):
    """生成用于算子学习的碰撞场景数据集。

    参数范围（典型紧凑型轿车碰撞）：
        m:    800 – 2000 kg
        k1:   150 – 500 kN/m  （弹性刚度）
        k2:   10 – 50 kN/m    （塑性刚度，约为 k1 的 10%）
        x_yield: 20 – 60 mm
        v0:   -5 – -20 m/s    （18-72 km/h）
        c:    300 – 1000 N*s/m

    参数：
        n_samples: 样本数量
        n_time_points: 时间离散点数量
        t_span: 时间范围
        seed: 随机种子，用于可复现性

    返回：
        字典，键值包括：
            'params': (n_samples, 6) — [m, c, k1, k2, x_yield, v0]
            't_grid': (n_time_points,) — 共享时间网格
            'x_data': (n_samples, n_time_points) — 位移
            'v_data': (n_samples, n_time_points) — 速度
            'a_data': (n_samples, n_time_points) — 加速度
    """
    rng = np.random.RandomState(seed)

    print(f"Generating {n_samples} crash scenarios...")

    # 参数采样
    m_arr = rng.uniform(800, 2000, n_samples)            # kg
    c_arr = rng.uniform(300, 1000, n_samples)            # N*s/m
    k1_arr = rng.uniform(150e3, 500e3, n_samples)        # N/m
    k2_ratio = rng.uniform(0.05, 0.20, n_samples)        # k2/k1 比值
    k2_arr = k1_arr * k2_ratio
    xy_arr = rng.uniform(0.02, 0.06, n_samples)          # m
    v0_arr = rng.uniform(-20, -5, n_samples)             # m/s

    params = np.column_stack([m_arr, c_arr, k1_arr, k2_arr, xy_arr, v0_arr])

    t_grid = np.linspace(t_span[0], t_span[1], n_time_points)
    x_data = np.zeros((n_samples, n_time_points))
    v_data = np.zeros((n_samples, n_time_points))
    a_data = np.zeros((n_samples, n_time_points))

    for i in range(n_samples):
        m, c, k1, k2, xy, v0 = params[i]
        _, x, v, a = solve_crash_scenario(m, c, k1, k2, xy, v0, t_span, n_time_points)
        x_data[i] = x
        v_data[i] = v
        a_data[i] = a
        if (i + 1) % 100 == 0:
            print(f"  {i + 1}/{n_samples} done")

    print(f"Dataset generated: {n_samples} samples, {n_time_points} time points")
    print(f"  Parameter ranges:")
    print(f"    m:   {m_arr.min():.0f} - {m_arr.max():.0f} kg")
    print(f"    k1:  {k1_arr.min()/1000:.0f} - {k1_arr.max()/1000:.0f} kN/m")
    print(f"    v0:  {abs(v0_arr.max())*3.6:.0f} - {abs(v0_arr.min())*3.6:.0f} km/h")

    return {
        "params": params,
        "param_names": ["m", "c", "k1", "k2", "x_yield", "v0"],
        "t_grid": t_grid,
        "t_span": t_span,
        "x_data": x_data,
        "v_data": v_data,
        "a_data": a_data,
    }


def split_dataset(data, train_ratio=0.8, val_ratio=0.1, seed=42):
    """将数据集划分为训练集/验证集/测试集。

    返回：
        train_data, val_data, test_data — 具有相同结构的字典
    """
    n = len(data["params"])
    rng = np.random.RandomState(seed)
    indices = rng.permutation(n)

    n_train = int(n * train_ratio)
    n_val = int(n * val_ratio)
    # 小样本时保证训练/验证集非空（比率显式为 0 时保持为空）
    if train_ratio > 0:
        n_train = max(1, n_train)
    if val_ratio > 0:
        n_val = max(1, n_val)
    n_val = min(n_val, n - n_train)

    train_idx = indices[:n_train]
    val_idx = indices[n_train : n_train + n_val]
    test_idx = indices[n_train + n_val :]

    def subset(idx):
        return {
            "params": data["params"][idx],
            "param_names": data["param_names"],
            "t_grid": data["t_grid"],
            "t_span": data["t_span"],
            "x_data": data["x_data"][idx],
            "v_data": data["v_data"][idx],
            "a_data": data["a_data"][idx],
        }

    print(f"Split: train={n_train}, val={n_val}, test={len(test_idx)}")
    return subset(train_idx), subset(val_idx), subset(test_idx)


def save_dataset(path, payload):
    """把数据集字典保存为 pickle 文件。"""
    with open(path, "wb") as f:
        pickle.dump(payload, f)
    print(f"Dataset saved to {path}")


def load_dataset(path):
    """从 pickle 文件加载数据集字典。"""
    with open(path, "rb") as f:
        return pickle.load(f)


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="生成碰撞算子学习数据集")
    add_common_args(parser, epochs=False)
    parser.add_argument("--num-samples", type=int, default=500)
    parser.add_argument("--num-time", type=int, default=200, help="时间离散点数")
    parser.add_argument("--t-end", type=float, default=0.12, help="仿真终止时间 [s]")
    args = parser.parse_args(argv)

    output_dir = resolve_output_dir(args.output_dir, "sciml_crash")
    data = generate_crash_dataset(
        n_samples=args.num_samples,
        n_time_points=args.num_time,
        t_span=(0.0, args.t_end),
        seed=args.seed,
    )
    train, val, test = split_dataset(data, seed=args.seed)
    save_dataset(
        output_dir / "crash_dataset.pkl",
        {"full": data, "train": train, "val": val, "test": test},
    )
    print(f"  Train: {len(train['params'])} | Val: {len(val['params'])} | Test: {len(test['params'])}")


if __name__ == "__main__":
    main()
