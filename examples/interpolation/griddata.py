"""scipy griddata 最近邻插值预测压溃性能。

多维散点数据无法构成规则网格，这里用 nearest 方法作为基线；
若改用 cubic/linear，查询点落在凸包外会产生 NaN，需要先清理再评估。

运行方式（在仓库根目录）：
    python -m examples.interpolation.griddata
"""

from __future__ import annotations

import argparse

import matplotlib.pyplot as plt
import numpy as np
from scipy.interpolate import griddata
from sklearn.metrics import r2_score

from common.cli import add_common_args
from common.paths import resolve_output_dir
from common.plotting import save_figure, use_headless_backend
from common.seed import set_seed
from examples._shared.crash_data import (
    load_crash_frame,
    normalize_frame,
    split_dataset,
)


def plot_regression(y_true: np.ndarray, y_pred: np.ndarray, title: str, path) -> None:
    fig, axes = plt.subplots(1, y_true.shape[1], figsize=(5 * y_true.shape[1], 4.5))
    axes = np.atleast_1d(axes)
    for idx, ax in enumerate(axes):
        ax.scatter(y_true[:, idx], y_pred[:, idx], s=10, alpha=0.6)
        ax.plot([0, 1], [0, 1], "r:", label="x = y")
        ax.set_xlabel("Normalized true value")
        ax.set_ylabel("Normalized prediction")
        ax.set_title(f"{title} - output {idx}")
        ax.legend()
        ax.grid(alpha=0.3)
    save_figure(fig, path)


def main(argv: list[str] | None = None) -> None:
    use_headless_backend()
    parser = argparse.ArgumentParser(description="griddata 插值压溃性能预测示例")
    add_common_args(parser, epochs=False)
    parser.add_argument(
        "--method",
        default="nearest",
        choices=["nearest", "linear", "cubic"],
        help="griddata 插值方法；linear/cubic 在凸包外会产生 NaN",
    )
    args = parser.parse_args(argv)

    set_seed(args.seed)
    output_dir = resolve_output_dir(args.output_dir, "interpolation")

    frame = load_crash_frame(data_root=args.data_root)
    x, y = normalize_frame(frame)
    x_train, x_val, y_train, y_val = split_dataset(x, y, seed=args.seed)

    y_train_pred = griddata(x_train, y_train, x_train, method=args.method)
    y_val_pred = griddata(x_train, y_train, x_val, method=args.method)

    # 先清理 NaN 再计算 R2，否则 sklearn 会直接报错
    nan_count = int(np.isnan(y_train_pred).sum() + np.isnan(y_val_pred).sum())
    y_train_pred = np.nan_to_num(y_train_pred)
    y_val_pred = np.nan_to_num(y_val_pred)
    if nan_count:
        print(f"警告：{args.method} 插值产生 {nan_count} 个 NaN，已替换为 0。")

    print(f"Train R2: {r2_score(y_train, y_train_pred):.4f}")
    print(f"Val R2: {r2_score(y_val, y_val_pred):.4f}")
    plot_regression(y_val, y_val_pred, "griddata", output_dir / "interpolation_griddata.png")
    print(f"Saved figures to {output_dir}")


if __name__ == "__main__":
    main()
