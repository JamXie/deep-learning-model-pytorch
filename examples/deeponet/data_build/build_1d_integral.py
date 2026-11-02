"""1D 积分算子数据：s(y) = ∫_0^y u(x) dx，u 为 [0,2] 上的高斯随机场。

训练/验证查询点 y ∈ [0,1]；外推测试 y ∈ [1,2]。

运行方式（在仓库根目录）：
    python -m examples.deeponet.data_build.build_1d_integral
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
from scipy.integrate import trapezoid

from common.cli import add_common_args
from common.paths import resolve_output_dir
from examples.deeponet.data_build.common import generate_grf, save_split, save_to_excel


def compute_integral(u: np.ndarray, y: np.ndarray, x_grid: np.ndarray) -> np.ndarray:
    """对每条曲线在 [0, y_i] 上做梯形积分。"""
    s = np.zeros((u.shape[0], 1))
    for i in range(u.shape[0]):
        mask = x_grid <= y[i, 0]
        if mask.sum() > 1:
            s[i, 0] = trapezoid(u[i, mask], x_grid[mask])
        else:
            s[i, 0] = 0.0
    return s


def generate_dataset(
    n: int,
    m: int,
    length_scale: float = 0.2,
    seed: int = 42,
    y_range: tuple[float, float] = (0.0, 1.0),
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """生成 (u, y, s)；u 定义在 [0,2]，y 可在训练区间或外推区间。"""
    rng = np.random.RandomState(seed)
    x_grid = np.linspace(0, 2, m)
    u = generate_grf(n, x_grid, length_scale=length_scale, rng=rng)
    y = rng.rand(n, 1) * (y_range[1] - y_range[0]) + y_range[0]
    s = compute_integral(u, y, x_grid)
    return u, y, s


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="生成 1D 积分算子数据")
    add_common_args(parser, epochs=False)
    parser.add_argument("--m", type=int, default=50, help="离散点数")
    parser.add_argument("--num-train", type=int, default=12000)
    parser.add_argument("--num-val", type=int, default=1000)
    parser.add_argument("--num-extra", type=int, default=1000)
    parser.add_argument("--length-scale", type=float, default=0.2)
    args = parser.parse_args(argv)

    out_dir = resolve_output_dir(args.output_dir, "deeponet", "1d")

    u, y, s = generate_dataset(args.num_train + args.num_val, args.m, args.length_scale, args.seed, (0, 1))
    split = args.num_train
    save_to_excel(u[:split], y[:split], s[:split], out_dir / "train.xlsx")
    save_to_excel(u[split:], y[split:], s[split:], out_dir / "val.xlsx")

    # 外推：y ∈ [1,2]
    u_extra, y_extra, s_extra = generate_dataset(
        args.num_extra, args.m, args.length_scale, args.seed + 1, (1, 2)
    )
    save_to_excel(u_extra, y_extra, s_extra, out_dir / "extra.xlsx")
    print(f"Generated 1D integral data in {out_dir}")


if __name__ == "__main__":
    main()
