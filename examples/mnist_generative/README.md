# MNIST 生成模型

三个在 MNIST 上训练的教学级生成模型。首次运行会由 `torchvision` 自动下载数据到
`data/mnist/`（已忽略，不入库）。

## 文件

| 文件 | 模型 | 训练目标 |
| --- | --- | --- |
| `mnist_diffusion.py` | 简化 U-Net（`BasicUNet`） | 从“干净图像 + 连续噪声”的混合中恢复干净图像，采样时从纯噪声分步去噪 |
| `pixelcnn_mnist.py` | 掩码卷积自编码器（`MaskedCNN` / `PixelCNN`） | 用因果掩码卷积重建 MNIST（MSE） |
| `vqvae_mnist.py` | VQ-VAE（编码器 + 码本 + 解码器） | 重建损失 + VQ/commitment 损失 |

## 运行

```bash
python -m examples.mnist_generative.mnist_diffusion --epochs 5
python -m examples.mnist_generative.pixelcnn_mnist --epochs 5
python -m examples.mnist_generative.vqvae_mnist --epochs 5
```

公共参数见根目录 README。扩散示例另有：

- `--n-steps`：采样步数（默认 40）
- `--sample-count`：采样图片数量（默认 16）

## 输出

默认写入 `outputs/mnist_generative/`：

| 脚本 | 图片 | 权重 |
| --- | --- | --- |
| 扩散 | `diffusion_loss.png`、`diffusion_samples.png` | `diffusion_unet.pt` |
| PixelCNN | `pixelcnn_loss.png`、`pixelcnn_reconstruction.png` | `pixelcnn.pt` |
| VQ-VAE | `vqvae_loss.png`、`vqvae_reconstruction.png` | `vqvae.pt` |

## 说明与局限

- **扩散**：这是连续加噪/去噪的简化版，并不是完整的 DDPM 训练与采样流程；用于教学
  展示“加噪 → 去噪”的思想。
- **PixelCNN**：本示例保留了原算例的**无条件、重建式**形式。掩码卷积保证每个位置
  只能看到左上方像素，但训练目标是重建而非逐像素自回归采样，也没有类别标签条件。
  若需要一个真正的条件 PixelCNN，应在输出端改为 256 类 softmax 并逐像素采样。
- **VQ-VAE**：验证损失按验证 batch 数取平均；码本大小、隐层宽度等由脚本内固定，
  可在类构造函数中调整。
