"""MNIST 生成模型示例：前向形状、掩码因果性与损失有限性。"""

from __future__ import annotations

import torch

from examples.mnist_generative.mnist_diffusion import BasicUNet, corrupt
from examples.mnist_generative.pixelcnn_mnist import MaskedCNN, PixelCNN
from examples.mnist_generative.vqvae_mnist import VQVAE


def test_basic_unet_forward_shape():
    net = BasicUNet()
    out = net(torch.rand(2, 1, 28, 28))
    assert out.shape == (2, 1, 28, 28)


def test_corrupt_zero_amount_keeps_image():
    x = torch.rand(2, 1, 8, 8)
    amount = torch.zeros(2)
    torch.manual_seed(0)
    mixed = corrupt(x, amount)
    assert torch.allclose(mixed, x)


def test_masked_cnn_causality():
    """改动右下角像素不应影响左上角输出。"""
    conv = MaskedCNN("A", 1, 1, kernel_size=3, stride=1, padding=1)
    with torch.no_grad():
        conv.weight.fill_(1.0)
        conv.bias.zero_()

    x = torch.zeros(1, 1, 28, 28)
    x[0, 0, 20, 20] = 1.0
    x_modified = x.clone()
    x_modified[0, 0, 15:, 15:] = 5.0

    with torch.no_grad():
        out = conv(x)
        out_modified = conv(x_modified)

    assert torch.allclose(out[:, :, :10, :10], out_modified[:, :, :10, :10])


def test_pixelcnn_output_shape():
    net = PixelCNN(channels=8, kernel=3).eval()
    out = net(torch.rand(2, 1, 28, 28))
    assert out.shape == (2, 1, 28, 28)


def test_vqvae_forward_and_loss_finite():
    net = VQVAE(
        num_hiddens=16,
        num_residual_layers=1,
        num_residual_hiddens=8,
        num_embeddings=32,
        embedding_dim=8,
    )
    vq_loss, recon, indices = net(torch.rand(2, 1, 28, 28))

    assert recon.shape == (2, 1, 28, 28)
    assert torch.isfinite(vq_loss)
    assert indices.numel() == 2 * 7 * 7
