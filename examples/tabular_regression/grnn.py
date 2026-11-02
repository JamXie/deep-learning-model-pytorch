"""广义回归神经网络（GRNN）：无需训练，直接以训练样本为核中心。

运行方式（在仓库根目录）：
    python -m examples.tabular_regression.grnn
"""

from __future__ import annotations

import argparse

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import r2_score

from common.cli import add_common_args
from common.device import get_device
from common.paths import resolve_output_dir
from common.plotting import save_figure, use_headless_backend
from common.seed import set_seed
from examples.tabular_regression.load_crash_data import (
    load_crash_frame,
    normalize_frame,
    split_dataset,
)


class GRNN(nn.Module):
    """高斯核回归：每个训练样本是一个核中心，输出为核加权平均。"""

    def __init__(self, train_inputs: torch.Tensor, train_targets: torch.Tensor, sigma: float = 0.1):
        super().__init__()
        self.register_buffer("train_inputs", train_inputs)
        self.register_buffer("train_targets", train_targets)
        self.sigma_squared = sigma ** 2

    def gaussian_rbf(self, x: torch.Tensor, centers: torch.Tensor) -> torch.Tensor:
        diff = x.unsqueeze(1) - centers.unsqueeze(0)  # (batch, centers, dim)
        dist = torch.sum(diff ** 2, dim=2)
        return torch.exp(-dist / (2 * self.sigma_squared))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        weights = self.gaussian_rbf(x, self.train_inputs)
        return torch.mm(weights, self.train_targets) / torch.sum(weights, dim=1, keepdim=True)


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
    parser = argparse.ArgumentParser(description="GRNN 压溃性能回归示例")
    add_common_args(parser, epochs=False)
    parser.add_argument("--sigma", type=float, default=0.16, help="高斯核宽度")
    args = parser.parse_args(argv)

    set_seed(args.seed)
    device = get_device(args.device)
    output_dir = resolve_output_dir(args.output_dir, "tabular_regression")

    frame = load_crash_frame(data_root=args.data_root)
    x, y = normalize_frame(frame)
    x_train, x_val, y_train, y_val = split_dataset(x, y, seed=args.seed)

    x_train_tensor = torch.tensor(x_train, dtype=torch.float32, device=device)
    y_train_tensor = torch.tensor(y_train, dtype=torch.float32, device=device)
    x_val_tensor = torch.tensor(x_val, dtype=torch.float32, device=device)

    model = GRNN(x_train_tensor, y_train_tensor, sigma=args.sigma)
    model.eval()
    with torch.no_grad():
        y_train_pred = model(x_train_tensor).cpu().numpy()
        y_val_pred = model(x_val_tensor).cpu().numpy()

    print(f"Train R2: {r2_score(y_train, y_train_pred):.4f}")
    print(f"Val R2: {r2_score(y_val, y_val_pred):.4f}")
    plot_regression(y_val, y_val_pred, "GRNN", output_dir / "grnn_regression.png")
    print(f"Saved figures to {output_dir}")


if __name__ == "__main__":
    main()
