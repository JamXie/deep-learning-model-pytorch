"""RBF 神经网络：KMeans 初始化高斯核中心，线性输出层可训练。

运行方式（在仓库根目录）：
    python -m examples.rbf_network.train --epochs 500
"""

from __future__ import annotations

import argparse

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.cluster import KMeans
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


class RBFNet(nn.Module):
    """标准 RBF 网络：隐藏层为高斯核，输出层为线性组合。"""

    def __init__(self, input_dim: int, hidden_dim: int, output_dim: int) -> None:
        super().__init__()
        self.hidden_dim = hidden_dim
        self.centers: torch.Tensor | None = None
        self.sigma: float | None = None
        self.linear = nn.Linear(hidden_dim, output_dim, bias=False)

    def gaussian_rbf(self, x: torch.Tensor, centers: torch.Tensor, sigma: float) -> torch.Tensor:
        diff = x.unsqueeze(1) - centers.unsqueeze(0)
        dist = torch.sum(diff ** 2, dim=2)
        return torch.exp(-dist / (2 * sigma ** 2))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.centers is None or self.sigma is None:
            raise RuntimeError("请先调用 initialize_centers() 初始化 RBF 中心。")
        return self.linear(self.gaussian_rbf(x, self.centers, self.sigma))

    def initialize_centers(self, x_train: np.ndarray, device: torch.device | None = None) -> None:
        """用 KMeans 选择核中心，并显式放到目标设备上。"""
        device = device if device is not None else torch.device("cpu")
        kmeans = KMeans(n_clusters=self.hidden_dim, random_state=42, n_init=10).fit(x_train)
        centers = torch.tensor(kmeans.cluster_centers_, dtype=torch.float32, device=device)
        # 中心不参与反向传播，但随模型一起迁移设备
        self.register_buffer("_centers", centers)
        self.centers = self._centers

        # sigma 取所有中心两两距离的均值
        dists = torch.cdist(centers, centers)
        upper = dists[torch.triu(torch.ones_like(dists, dtype=torch.bool), diagonal=1)]
        self.sigma = float(upper.mean().item())


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
    parser = argparse.ArgumentParser(description="RBF 网络压溃性能回归示例")
    add_common_args(parser, epochs=False)
    parser.add_argument("--epochs", type=int, default=500, help="训练轮数")
    parser.add_argument("--batch-size", type=int, default=64, help="批大小")
    parser.add_argument("--hidden-dim", type=int, default=100, help="RBF 中心数")
    args = parser.parse_args(argv)

    set_seed(args.seed)
    device = get_device(args.device)
    output_dir = resolve_output_dir(args.output_dir, "rbf_network")

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
        drop_last=len(x_train_tensor) >= args.batch_size,
    )

    net = RBFNet(x_train.shape[1], args.hidden_dim, y_train.shape[1]).to(device)
    net.initialize_centers(x_train, device=device)

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
        if (epoch + 1) % 100 == 0 or epoch == 0:
            print(f"epoch {epoch + 1}/{args.epochs} loss={epoch_loss / (step + 1):.6f}")

    # 验证阶段：训练张量也需要搬到 device
    net.eval()
    with torch.no_grad():
        y_train_pred = net(x_train_tensor.to(device)).cpu().numpy()
        y_val_pred = net(x_val_tensor.to(device)).cpu().numpy()

    print(f"Train R2: {r2_score(y_train, y_train_pred):.4f}")
    print(f"Val R2: {r2_score(y_val, y_val_pred):.4f}")
    plot_regression(y_val, y_val_pred, "RBF Network", output_dir / "rbf_regression.png")
    print(f"Saved figures to {output_dir}")


if __name__ == "__main__":
    main()
