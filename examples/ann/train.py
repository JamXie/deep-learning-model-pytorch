"""ANN 回归：由结构参数与加载角度预测压溃性能。

运行方式（在仓库根目录）：
    python -m examples.ann.train --epochs 300
"""

from __future__ import annotations

import argparse

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.metrics import r2_score
from torch.utils.data import DataLoader, TensorDataset

from common.cli import add_common_args
from common.device import get_device
from common.paths import resolve_output_dir
from common.plotting import save_figure, use_headless_backend
from common.seed import set_seed
from examples._shared.crash_data import (
    load_crash_frame,
    normalize_frame,
    split_dataset,
)


class ANN(nn.Module):
    """三层全连接网络，带 BatchNorm 与 Dropout。"""

    def __init__(self, input_dim: int, output_dim: int) -> None:
        super().__init__()
        self.ann = nn.Sequential(
            nn.Linear(input_dim, 128),
            nn.BatchNorm1d(128),
            nn.LeakyReLU(0.2),
            nn.Dropout(p=0.5),
            nn.Linear(128, 64),
            nn.BatchNorm1d(64),
            nn.LeakyReLU(0.2),
            nn.Dropout(p=0.5),
            nn.Linear(64, output_dim),
            nn.Tanh(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.ann(x)


def plot_regression(y_true: np.ndarray, y_pred: np.ndarray, title: str, path) -> None:
    """绘制预测值-真实值散点图并保存。"""
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
    parser = argparse.ArgumentParser(description="ANN 压溃性能回归示例")
    add_common_args(parser, epochs=False)
    parser.add_argument("--epochs", type=int, default=300, help="训练轮数")
    parser.add_argument("--batch-size", type=int, default=64, help="批大小")
    args = parser.parse_args(argv)

    set_seed(args.seed)
    device = get_device(args.device)
    output_dir = resolve_output_dir(args.output_dir, "ann")

    frame = load_crash_frame(data_root=args.data_root)
    x, y = normalize_frame(frame)
    x_train, x_val, y_train, y_val = split_dataset(x, y, seed=args.seed)

    x_train_tensor = torch.tensor(x_train, dtype=torch.float32)
    y_train_tensor = torch.tensor(y_train, dtype=torch.float32)
    x_val_tensor = torch.tensor(x_val, dtype=torch.float32)
    y_val_tensor = torch.tensor(y_val, dtype=torch.float32)

    train_loader = DataLoader(
        TensorDataset(x_train_tensor, y_train_tensor),
        batch_size=args.batch_size,
        shuffle=True,
        drop_last=len(x_train_tensor) >= args.batch_size,  # 小样本时避免空 batch
    )

    net = ANN(x_train.shape[1], y_train.shape[1]).to(device)
    optimizer = optim.Adam(net.parameters(), lr=1e-4, weight_decay=1e-5)
    criterion = nn.MSELoss()

    net.train()
    for epoch in range(args.epochs):
        epoch_loss = 0.0
        for step, (batch_x, batch_y) in enumerate(train_loader):
            if args.limit_batches is not None and step >= args.limit_batches:
                break
            optimizer.zero_grad()
            batch_x = batch_x.to(device)
            batch_y = batch_y.to(device)
            loss = criterion(net(batch_x), batch_y)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()
        if (epoch + 1) % 50 == 0 or epoch == 0:
            print(f"epoch {epoch + 1}/{args.epochs} loss={epoch_loss / (step + 1):.6f}")

    # 验证阶段：训练/验证张量都搬到同一设备
    net.eval()
    with torch.no_grad():
        y_train_pred = net(x_train_tensor.to(device)).cpu().numpy()
        y_val_pred = net(x_val_tensor.to(device)).cpu().numpy()

    print(f"Train R2: {r2_score(y_train, y_train_pred):.4f}")
    print(f"Val R2: {r2_score(y_val, y_val_pred):.4f}")
    plot_regression(y_val, y_val_pred, "ANN", output_dir / "ann_regression.png")
    print(f"Saved figures to {output_dir}")


if __name__ == "__main__":
    main()
