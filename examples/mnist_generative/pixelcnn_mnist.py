"""PixelCNN 风格的掩码卷积自编码器（MNIST）。

注意：原始算例只有掩码卷积 + 重建损失，没有标签条件，也不是真正的逐像素
自回归采样。这里保留这一教学形式，仅在文档中说明其局限：
  - 掩码保证每个位置的输出只依赖其“上方/左侧”的输入；
  - 训练目标为重建 MNIST 图像（MSE）。

运行方式（在仓库根目录，首次运行会自动下载 MNIST）：
    python -m examples.mnist_generative.pixelcnn_mnist --epochs 5
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader, random_split

from common.cli import add_common_args
from common.device import get_device
from common.paths import resolve_data_path, resolve_output_dir
from common.plotting import save_figure, use_headless_backend
from common.seed import set_seed


class MaskedCNN(nn.Conv2d):
    """带因果掩码的卷积：位置 (i, j) 只能看到自身左上方像素。

    掩码在 forward 中临时与权重相乘，避免原地修改参数破坏自动求导。
    """

    def __init__(self, mask_type: str, *args, **kwargs) -> None:
        if mask_type not in ["A", "B"]:
            raise ValueError("mask_type 必须是 'A' 或 'B'")
        self.mask_type = mask_type
        super().__init__(*args, **kwargs)
        mask = torch.ones_like(self.weight)
        _, _, height, width = self.weight.size()
        if mask_type == "A":
            # A 型：不包含中心像素本身
            mask[:, :, height // 2, width // 2:] = 0
            mask[:, :, height // 2 + 1:, :] = 0
        else:
            # B 型：包含中心像素
            mask[:, :, height // 2, width // 2 + 1:] = 0
            mask[:, :, height // 2 + 1:, :] = 0
        self.register_buffer("mask", mask)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # 每次都用权重与掩码的乘积，不修改 self.weight
        return F.conv2d(x, self.weight * self.mask, self.bias, self.stride, self.padding, self.dilation, self.groups)


class PixelCNN(nn.Module):
    """8 层掩码卷积网络，输出单通道重建图像。"""

    def __init__(self, channels: int = 64, kernel: int = 7) -> None:
        super().__init__()
        self.conv1 = MaskedCNN("A", 1, channels, kernel, 1, kernel // 2, bias=False)
        self.bn1 = nn.BatchNorm2d(channels)
        self.blocks = nn.ModuleList([
            nn.Sequential(
                MaskedCNN("B", channels, channels, kernel, 1, kernel // 2, bias=False),
                nn.BatchNorm2d(channels),
                nn.ReLU(True),
            )
            for _ in range(7)
        ])
        # 输出通道必须与输入一致（1 通道灰度图）
        self.out = nn.Conv2d(channels, 1, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = F.relu(self.bn1(self.conv1(x)))
        for block in self.blocks:
            x = block(x)
        return self.out(x)


def build_loaders(data_root: Path, batch_size: int):
    """按 80/10/10 划分训练、验证与测试集。"""
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,)),
    ])
    full = torchvision.datasets.MNIST(root=str(data_root), train=True, transform=transform, download=True)
    train_size = int(0.8 * len(full))
    val_size = int(0.1 * len(full))
    test_size = len(full) - train_size - val_size
    train_ds, val_ds, test_ds = random_split(full, [train_size, val_size, test_size])

    return (
        DataLoader(train_ds, batch_size=batch_size, shuffle=True),
        DataLoader(val_ds, batch_size=batch_size, shuffle=False),
        DataLoader(test_ds, batch_size=batch_size, shuffle=False),
    )


def compare_figure(x: torch.Tensor, pred: torch.Tensor, path) -> None:
    fig, axes = plt.subplots(2, 1, figsize=(10, 6))
    axes[0].imshow(torchvision.utils.make_grid(x.cpu())[0].clip(0, 1), cmap="Greys")
    axes[0].set_title("input")
    axes[0].axis("off")
    axes[1].imshow(torchvision.utils.make_grid(pred.cpu())[0].clip(0, 1), cmap="Greys")
    axes[1].set_title("reconstruction")
    axes[1].axis("off")
    save_figure(fig, path)


def main(argv: list[str] | None = None) -> None:
    use_headless_backend()
    parser = argparse.ArgumentParser(description="MNIST PixelCNN 示例")
    add_common_args(parser)
    args = parser.parse_args(argv)

    set_seed(args.seed)
    device = get_device(args.device)
    output_dir = resolve_output_dir(args.output_dir, "mnist_generative")
    data_root = Path(args.data_root).expanduser() if args.data_root else resolve_data_path("mnist")

    train_loader, val_loader, _ = build_loaders(data_root, args.batch_size)

    net = PixelCNN().to(device)
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(net.parameters(), lr=1e-3)

    train_losses: list[float] = []
    val_losses: list[float] = []

    for epoch in range(args.epochs):
        net.train()
        train_sum = 0.0
        for step, (x, _) in enumerate(train_loader):
            if args.limit_batches is not None and step >= args.limit_batches:
                break
            x = x.to(device)
            optimizer.zero_grad()
            loss = criterion(net(x), x)
            loss.backward()
            optimizer.step()
            train_sum += loss.item()
        train_losses.append(train_sum / (step + 1))

        net.eval()
        val_sum = 0.0
        with torch.no_grad():
            for v_i, (v_x, _) in enumerate(val_loader):
                if args.limit_batches is not None and v_i >= max(1, args.limit_batches // 2):
                    break
                v_x = v_x.to(device)
                val_sum += criterion(net(v_x), v_x).item()
        val_losses.append(val_sum / (v_i + 1))  # 使用验证集自己的计数

        print(f"epoch {epoch + 1}/{args.epochs} train={train_losses[-1]:.6f} val={val_losses[-1]:.6f}")

    fig, ax = plt.subplots(figsize=(6, 4))
    ax.plot(np.arange(1, len(train_losses) + 1), train_losses, "r", label="train")
    ax.plot(np.arange(1, len(val_losses) + 1), val_losses, "b", label="val")
    ax.set_xlabel("epoch")
    ax.set_ylabel("MSE loss")
    ax.set_title("PixelCNN loss")
    ax.legend()
    ax.grid(alpha=0.3)
    save_figure(fig, output_dir / "pixelcnn_loss.png")

    net.eval()
    with torch.no_grad():
        x, _ = next(iter(train_loader))
        x = x[:16].to(device)
        pred = net(x)
    compare_figure(x, pred, output_dir / "pixelcnn_reconstruction.png")

    torch.save(net.state_dict(), output_dir / "pixelcnn.pt")
    print(f"Saved outputs to {output_dir}")


if __name__ == "__main__":
    main()
