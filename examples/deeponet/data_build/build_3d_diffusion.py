"""3D 扩散问题数据：一维空间 + 时间，学习 (f(x), (x,t)) -> s(x,t)。

方程（隐式欧拉）：
    s_t = D * s_xx + f(x),  s(0)=s(1)=0
训练查询点在 t ∈ [0,T]，外推查询点在 t ∈ (T, T+1]。

运行方式（在仓库根目录）：
    python -m examples.deeponet.data_build.build_3d_diffusion
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

from common.cli import add_common_args
from common.paths import resolve_output_dir
from examples.deeponet.data_build.common import generate_grf, save_to_excel


def build_laplacian(m: int, dx: float) -> np.ndarray:
    """一维二阶差分矩阵（Dirichlet 边界）。"""
    matrix = np.zeros((m, m))
    for i in range(1, m - 1):
        matrix[i, i - 1] = 1
        matrix[i, i] = -2
        matrix[i, i + 1] = 1
    matrix[0, :] = 0
    matrix[-1, :] = 0
    return matrix / dx ** 2


def solve_pde(f: np.ndarray, x_grid: np.ndarray, nt: int = 200, total_time: float = 5.0, diffusion: float = 0.1) -> np.ndarray:
    """隐式欧拉求解，返回 (nt, m) 的时空解。"""
    m = len(x_grid)
    dx = x_grid[1] - x_grid[0]
    dt = total_time / nt
    laplacian = build_laplacian(m, dx)
    implicit = np.eye(m) - dt * diffusion * laplacian

    s = np.zeros(m)
    history = []
    for _ in range(nt):
        s = np.linalg.solve(implicit, s + dt * f)
        history.append(s.copy())
    return np.array(history)


def interpolate_2d(field: np.ndarray, x_grid: np.ndarray, t_grid: np.ndarray, x: float, t: float) -> float:
    """双线性插值取 (x, t) 处的解。"""
    i = int(np.clip(np.searchsorted(x_grid, x) - 1, 0, len(x_grid) - 2))
    j = int(np.clip(np.searchsorted(t_grid, t) - 1, 0, len(t_grid) - 2))

    x1, x2 = x_grid[i], x_grid[i + 1]
    t1, t2 = t_grid[j], t_grid[j + 1]
    q11, q12 = field[j, i], field[j + 1, i]
    q21, q22 = field[j, i + 1], field[j + 1, i + 1]

    return float(
        q11 * (x2 - x) * (t2 - t)
        + q21 * (x - x1) * (t2 - t)
        + q12 * (x2 - x) * (t - t1)
        + q22 * (x - x1) * (t - t1)
    ) / ((x2 - x1) * (t2 - t1))


def generate_dataset(
    n: int = 1000,
    m: int = 50,
    nt: int = 200,
    total_time: float = 5.0,
    seed: int = 42,
    t_range: tuple[float, float] | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """生成 (f, (x,t), s)；t_range 为空时使用 [0, total_time]。"""
    rng = np.random.RandomState(seed)
    x_grid = np.linspace(0, 1, m)
    t_grid = np.linspace(0, total_time, nt)

    # 源项放大 5 倍，让不同样本解的量级更明显
    f_all = 5 * generate_grf(n, x_grid, length_scale=0.2, rng=rng)

    low, high = t_range if t_range is not None else (0.0, total_time)
    y = np.zeros((n, 2))
    s = np.zeros((n, 1))

    for i in range(n):
        field = solve_pde(f_all[i], x_grid, nt=nt, total_time=total_time)
        xi = rng.rand()
        ti = rng.rand() * (high - low) + low
        y[i] = [xi, ti]
        s[i, 0] = interpolate_2d(field, x_grid, t_grid, xi, ti)

    return f_all.astype(np.float32), y.astype(np.float32), s.astype(np.float32)


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="生成 3D 扩散（1D 空间 + 时间）数据")
    add_common_args(parser, epochs=False)
    parser.add_argument("--m", type=int, default=50, help="空间离散点数")
    parser.add_argument("--nt", type=int, default=200, help="时间步数")
    parser.add_argument("--num-train", type=int, default=1000)
    parser.add_argument("--num-val", type=int, default=200)
    parser.add_argument("--num-extra", type=int, default=200)
    parser.add_argument("--T", type=float, default=5.0, help="训练时间上限")
    args = parser.parse_args(argv)

    out_dir = resolve_output_dir(args.output_dir, "deeponet", "3d")

    f, y, s = generate_dataset(args.num_train + args.num_val, args.m, args.nt, args.T, args.seed)
    split = args.num_train
    save_to_excel(f[:split], y[:split], s[:split], out_dir / "train.xlsx", branch_sheet="f_function")
    save_to_excel(f[split:], y[split:], s[split:], out_dir / "val.xlsx", branch_sheet="f_function")

    # 外推：把仿真延长 1 个时间单位，查询 t ∈ (T, T+1]
    f_extra, y_extra, s_extra = generate_dataset(
        args.num_extra,
        args.m,
        args.nt,
        args.T + 1.0,
        args.seed + 1,
        t_range=(args.T, args.T + 1.0),
    )
    save_to_excel(f_extra, y_extra, s_extra, out_dir / "extra.xlsx", branch_sheet="f_function")
    print(f"Generated 3D diffusion data in {out_dir}")


if __name__ == "__main__":
    main()
