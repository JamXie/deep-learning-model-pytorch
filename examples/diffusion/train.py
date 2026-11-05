"""在 MNIST 上训练一个简化扩散模型（连续加噪 + 去噪回归）。

这是一份教学示例：
  - 用连续噪声强度 amount 把干净图像与随机噪声混合；
  - 网络学习从含噪图像恢复干净图像；
  - 采样时从纯噪声出发，分多步逐步向网络预测结果靠拢。

运行方式（在仓库根目录，首次运行会自动下载 MNIST）：
    python -m examples.diffusion.train --epochs 5
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader

from common.cli import add_common_args
from common.device import get_device
from common.paths import resolve_data_path, resolve_output_dir
from common.plotting import save_figure, use_headless_backend
from common.seed import set_seed


class BasicUNet(nn.Module):
    """最小 U-Net：三层下采样与三层上采样，带跳跃连接。"""

    def __init__(self, in_channels: int = 1, out_channels: int = 1) -> None:
        super().__init__()
        self.down_layers = nn.ModuleList([
            nn.Conv2d(in_channels, 32, kernel_size=5, padding=2),
            nn.Conv2d(32, 64, kernel_size=5, padding=2),
            nn.Conv2d(64, 64, kernel_size=5, padding=2),
        ])
        self.up_layers = nn.ModuleList([
            nn.Conv2d(64, 64, kernel_size=5, padding=2),
            nn.Conv2d(64, 32, kernel_size=5, padding=2),
            nn.Conv2d(32, out_channels, kernel_size=5, padding=2),
        ])
        self.act = nn.SiLU()
        self.downscale = nn.MaxPool2d(2)
        self.upscale = nn.Upsample(scale_factor=2)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = []
        for i, layer in enumerate(self.down_layers):
            x = self.act(layer(x))
            if i < 2:  # 前两层保存特征用于跳跃连接
                h.append(x)
                x = self.downscale(x)

        for i, layer in enumerate(self.up_layers):
            if i > 0:
                x = self.upscale(x)
                x = x + h.pop()  # 跳跃连接
            x = self.act(layer(x))
        return x


def corrupt(x: torch.Tensor, amount: torch.Tensor) -> torch.Tensor:
    """按 amount 把输入与均匀噪声混合；amount=0 保留原图，1 为纯噪声。"""
    noise = torch.rand_like(x)
    amount = amount.view(-1, 1, 1, 1)
    return x * (1 - amount) + noise * amount


def build_loader(data_root: Path, batch_size: int, train: bool = True, download: bool = True) -> DataLoader:
    """加载 MNIST 并归一化到 [-1, 1]。"""
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(0.5, 0.5),
    ])
    dataset = torchvision.datasets.MNIST(
        root=str(data_root),
        train=train,
        transform=transform,
        download=download,
    )
    return DataLoader(dataset, batch_size=batch_size, shuffle=train)


@torch.no_grad()
def sample(net: nn.Module, device: torch.device, count: int = 16, n_steps: int = 40) -> torch.Tensor:
    """从纯噪声出发，按 n_steps 步逐步预测并混合，返回 [0,1] 图像。"""
    x = torch.rand(count, 1, 28, 28, device=device)
    for i in range(n_steps):
        # 让噪声强度从高到低，避免每步都完全信任网络输出
        pred = net(x)
        mix_factor = 1 / (n_steps - i)
        x = x * (1 - mix_factor) + pred * mix_factor
    return x.clamp(0, 1).cpu()


def main(argv: list[str] | None = None) -> None:
    use_headless_backend()
    parser = argparse.ArgumentParser(description="MNIST 简化扩散模型示例")
    add_common_args(parser)
    parser.add_argument("--n-steps", type=int, default=40, help="采样步数")
    parser.add_argument("--sample-count", type=int, default=16, help="采样图片数量")
    args = parser.parse_args(argv)

    set_seed(args.seed)
    device = get_device(args.device)
    output_dir = resolve_output_dir(args.output_dir, "diffusion")
    data_root = Path(args.data_root).expanduser() if args.data_root else resolve_data_path("mnist")

    train_loader = build_loader(data_root, args.batch_size, train=True)

    net = BasicUNet().to(device)
    optimizer = torch.optim.Adam(net.parameters(), lr=1e-3)
    criterion = nn.MSELoss()

    losses: list[float] = []
    for epoch in range(args.epochs):
        net.train()
        running = 0.0
        for step, (x, _) in enumerate(train_loader):
            if args.limit_batches is not None and step >= args.limit_batches:
                break
            x = x.to(device)
            noise_amount = torch.rand(x.shape[0], device=device)
            noisy_x = corrupt(x, noise_amount)

            loss = criterion(net(noisy_x), x)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            losses.append(loss.item())
            running += loss.item()
        print(f"epoch {epoch + 1}/{args.epochs} loss={running / (step + 1):.6f}")

    # 损失曲线
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.plot(losses)
    ax.set_xlabel("iteration")
    ax.set_ylabel("MSE loss")
    ax.set_title("denoising loss")
    ax.grid(alpha=0.3)
    save_figure(fig, output_dir / "diffusion_loss.png")

    # 采样结果网格
    samples = sample(net, device, count=args.sample_count, n_steps=args.n_steps)
    grid = torchvision.utils.make_grid(samples, nrow=4)
    save_figure(_grid_figure(grid), output_dir / "diffusion_samples.png")

    torch.save(net.state_dict(), output_dir / "diffusion_unet.pt")
    print(f"Saved outputs to {output_dir}")


def _grid_figure(grid: torch.Tensor):
    fig, ax = plt.subplots(figsize=(6, 6))
    ax.imshow(grid.permute(1, 2, 0).numpy(), cmap="Greys")
    ax.set_title("generated samples")
    ax.axis("off")
    return fig


if __name__ == "__main__":
    main()
