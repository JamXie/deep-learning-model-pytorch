"""2D 受迫单摆数据：s'' + sin(s) = u(t)，学习 (u(t), t) -> s(t)。

运行方式（在仓库根目录）：
    python -m examples.deeponet.data_build.build_2d_pendulum
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
from scipy.integrate import solve_ivp

from common.cli import add_common_args
from common.paths import resolve_output_dir
from examples.deeponet.data_build.common import generate_grf, save_to_excel


def solve_pendulum(u_func: np.ndarray, t_grid: np.ndarray) -> np.ndarray:
    """用 RK45 求解受迫单摆，返回角度序列 s(t)。"""

    def ode(t, state):
        angle, velocity = state
        force = np.interp(t, t_grid, u_func)
        return [velocity, force - np.sin(angle)]

    solution = solve_ivp(
        ode,
        t_span=(t_grid[0], t_grid[-1]),
        y0=[0.0, 0.0],
        t_eval=t_grid,
        method="RK45",
    )
    if not solution.success:
        raise RuntimeError(f"ODE 求解失败：{solution.message}")
    return solution.y[0]


def generate_dataset(
    n: int = 1000,
    m: int = 50,
    t_max: float = 2.0,
    seed: int = 42,
    y_range: tuple[float, float] = (0.0, 1.0),
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    rng = np.random.RandomState(seed)
    t_grid = np.linspace(0, t_max, m)
    u = generate_grf(n, t_grid, length_scale=0.3, rng=rng)

    solutions = np.stack([solve_pendulum(u[i], t_grid) for i in range(n)], axis=0)  # (N, m)
    y = rng.rand(n, 1) * (y_range[1] - y_range[0]) + y_range[0]

    s = np.zeros((n, 1))
    for i in range(n):
        s[i, 0] = np.interp(y[i, 0], t_grid, solutions[i])
    return u, y, s


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="生成 2D 受迫单摆数据")
    add_common_args(parser, epochs=False)
    parser.add_argument("--m", type=int, default=50)
    parser.add_argument("--num-train", type=int, default=1000)
    parser.add_argument("--num-val", type=int, default=200)
    parser.add_argument("--num-extra", type=int, default=200)
    parser.add_argument("--t-max", type=float, default=2.0)
    args = parser.parse_args(argv)

    out_dir = resolve_output_dir(args.output_dir, "deeponet", "2d")

    u, y, s = generate_dataset(args.num_train + args.num_val, args.m, args.t_max, args.seed, (0, 1))
    split = args.num_train
    save_to_excel(u[:split], y[:split], s[:split], out_dir / "train.xlsx")
    save_to_excel(u[split:], y[split:], s[split:], out_dir / "val.xlsx")

    u_extra, y_extra, s_extra = generate_dataset(
        args.num_extra, args.m, args.t_max, args.seed + 1, (1, 2)
    )
    save_to_excel(u_extra, y_extra, s_extra, out_dir / "extra.xlsx")
    print(f"Generated 2D pendulum data in {out_dir}")


if __name__ == "__main__":
    main()
