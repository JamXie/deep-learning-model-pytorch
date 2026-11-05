"""在 MNIST 上训练 VQ-VAE（向量量化变分自编码器）。

运行方式（在仓库根目录，首次运行会自动下载 MNIST）：
    python -m examples.vqvae.train --epochs 5
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


class VectorQuantizer(nn.Module):
    """把连续特征映射到最近的码本向量，并返回 commitment/VQ 损失。"""

    def __init__(self, num_embeddings: int, embedding_dim: int, commitment_cost: float) -> None:
        super().__init__()
        self.embedding_dim = embedding_dim
        self.num_embeddings = num_embeddings
        self.commitment_cost = commitment_cost

        self.embeddings = nn.Embedding(num_embeddings, embedding_dim)
        self.embeddings.weight.data.uniform_(-1 / num_embeddings, 1 / num_embeddings)

    def forward(self, inputs: torch.Tensor):
        # NCHW -> NHWC，方便按像素位置量化
        inputs = inputs.permute(0, 2, 3, 1).contiguous()
        input_shape = inputs.shape
        flat_input = inputs.view(-1, self.embedding_dim)

        distances = (
            torch.sum(flat_input ** 2, dim=1, keepdim=True)
            + torch.sum(self.embeddings.weight ** 2, dim=1)
            - 2 * torch.matmul(flat_input, self.embeddings.weight.t())
        )
        encoding_indices = torch.argmin(distances, dim=1).unsqueeze(1)

        encodings = torch.zeros(encoding_indices.shape[0], self.num_embeddings, device=inputs.device)
        encodings.scatter_(1, encoding_indices, 1)

        quantized = torch.matmul(encodings, self.embeddings.weight).view(input_shape)

        e_latent_loss = F.mse_loss(quantized.detach(), inputs)
        q_latent_loss = F.mse_loss(quantized, inputs.detach())
        loss = q_latent_loss + self.commitment_cost * e_latent_loss

        # 直通估计：前向用量化值，反向梯度直接传回编码器
        quantized = inputs + (quantized - inputs).detach()
        return loss, quantized.permute(0, 3, 1, 2).contiguous(), encoding_indices


class Residual(nn.Module):
    def __init__(self, in_channels: int, num_hiddens: int) -> None:
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, num_hiddens, kernel_size=3, stride=1, padding=1)
        self.conv2 = nn.Conv2d(num_hiddens, in_channels, kernel_size=1, stride=1, padding=0)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        x = F.relu(inputs)
        x = F.relu(self.conv1(x))
        return inputs + self.conv2(x)


class ResidualStack(nn.Module):
    def __init__(self, in_channels: int, num_residual_layers: int, num_residual_hiddens: int) -> None:
        super().__init__()
        self.num_residual_layers = num_residual_layers
        self.layers = nn.ModuleList(
            [Residual(in_channels, num_residual_hiddens) for _ in range(num_residual_layers)]
        )

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        x = inputs
        for layer in self.layers:
            x = layer(x)
        return x


class Encoder(nn.Module):
    def __init__(self, in_channels: int, num_hiddens: int, num_residual_layers: int, num_residual_hiddens: int) -> None:
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, num_hiddens // 2, kernel_size=4, stride=2, padding=1)
        self.conv2 = nn.Conv2d(num_hiddens // 2, num_hiddens, kernel_size=4, stride=2, padding=1)
        self.conv3 = nn.Conv2d(num_hiddens, num_hiddens, kernel_size=3, stride=1, padding=1)
        self.residual_stack = ResidualStack(num_hiddens, num_residual_layers, num_residual_hiddens)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        x = F.relu(self.conv1(inputs))
        x = F.relu(self.conv2(x))
        x = self.conv3(x)
        return self.residual_stack(x)


class Decoder(nn.Module):
    def __init__(self, in_channels: int, num_hiddens: int, num_residual_layers: int, num_residual_hiddens: int) -> None:
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, num_hiddens, kernel_size=3, stride=1, padding=1)
        self.residual_stack = ResidualStack(num_hiddens, num_residual_layers, num_residual_hiddens)
        self.conv_trans1 = nn.ConvTranspose2d(num_hiddens, num_hiddens // 2, kernel_size=4, stride=2, padding=1)
        self.conv_trans2 = nn.ConvTranspose2d(num_hiddens // 2, 1, kernel_size=4, stride=2, padding=1)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        x = F.relu(self.conv1(inputs))
        x = self.residual_stack(x)
        x = F.relu(self.conv_trans1(x))
        return self.conv_trans2(x)


class VQVAE(nn.Module):
    def __init__(
        self,
        in_channels: int = 1,
        num_hiddens: int = 128,
        num_residual_layers: int = 2,
        num_residual_hiddens: int = 32,
        num_embeddings: int = 512,
        embedding_dim: int = 64,
        commitment_cost: float = 0.25,
    ) -> None:
        super().__init__()
        self.encoder = Encoder(in_channels, num_hiddens, num_residual_layers, num_residual_hiddens)
        self.pre_vq_conv = nn.Conv2d(num_hiddens, embedding_dim, kernel_size=1, stride=1)
        self.vq_layer = VectorQuantizer(num_embeddings, embedding_dim, commitment_cost)
        self.decoder = Decoder(embedding_dim, num_hiddens, num_residual_layers, num_residual_hiddens)

    def forward(self, x: torch.Tensor):
        z = self.encoder(x)
        z = self.pre_vq_conv(z)
        vq_loss, quantized, indices = self.vq_layer(z)
        recon = self.decoder(quantized)
        return vq_loss, recon, indices


def build_loaders(data_root: Path, batch_size: int):
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
    parser = argparse.ArgumentParser(description="MNIST VQ-VAE 示例")
    add_common_args(parser)
    args = parser.parse_args(argv)

    set_seed(args.seed)
    device = get_device(args.device)
    output_dir = resolve_output_dir(args.output_dir, "vqvae")
    data_root = Path(args.data_root).expanduser() if args.data_root else resolve_data_path("mnist")

    train_loader, val_loader, _ = build_loaders(data_root, args.batch_size)

    net = VQVAE().to(device)
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
            vq_loss, recon, _ = net(x)
            loss = criterion(recon, x) + vq_loss
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
                vq_loss, recon, _ = net(v_x)
                val_sum += (criterion(recon, v_x) + vq_loss).item()
        # 用验证集自己的批次数，而不是训练循环的 i
        val_losses.append(val_sum / (v_i + 1))

        print(f"epoch {epoch + 1}/{args.epochs} train={train_losses[-1]:.6f} val={val_losses[-1]:.6f}")

    fig, ax = plt.subplots(figsize=(6, 4))
    ax.plot(np.arange(1, len(train_losses) + 1), train_losses, "r", label="train")
    ax.plot(np.arange(1, len(val_losses) + 1), val_losses, "b", label="val")
    ax.set_xlabel("epoch")
    ax.set_ylabel("loss")
    ax.set_title("VQ-VAE loss")
    ax.legend()
    ax.grid(alpha=0.3)
    save_figure(fig, output_dir / "vqvae_loss.png")

    net.eval()
    with torch.no_grad():
        x, _ = next(iter(train_loader))
        x = x[:16].to(device)
        _, recon, _ = net(x)
    compare_figure(x, recon, output_dir / "vqvae_reconstruction.png")

    torch.save(net.state_dict(), output_dir / "vqvae.pt")
    print(f"Saved outputs to {output_dir}")


if __name__ == "__main__":
    main()
