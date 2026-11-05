"""RBF 插值（scipy RBFInterpolator，multiquadric 核）预测压溃性能。

运行方式（在仓库根目录）：
    python -m examples.interpolation.rbf
"""

from __future__ import annotations

import argparse

import matplotlib.pyplot as plt
import numpy as np
from scipy.interpolate import RBFInterpolator
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
    parser = argparse.ArgumentParser(description="RBF 插值压溃性能预测示例")
    add_common_args(parser, epochs=False)
    args = parser.parse_args(argv)

    set_seed(args.seed)
    output_dir = resolve_output_dir(args.output_dir, "interpolation")

    frame = load_crash_frame(data_root=args.data_root)
    x, y = normalize_frame(frame)
    x_train, x_val, y_train, y_val = split_dataset(x, y, seed=args.seed)

    # 与原始工程一致的核参数
    rbf = RBFInterpolator(x_train, y_train, kernel="multiquadric", epsilon=1.8)
    y_train_pred = np.nan_to_num(rbf(x_train))
    y_val_pred = np.nan_to_num(rbf(x_val))

    print(f"Train R2: {r2_score(y_train, y_train_pred):.4f}")
    print(f"Val R2: {r2_score(y_val, y_val_pred):.4f}")
    plot_regression(y_val, y_val_pred, "RBF Interpolation", output_dir / "interpolation_rbf.png")
    print(f"Saved figures to {output_dir}")


if __name__ == "__main__":
    main()
