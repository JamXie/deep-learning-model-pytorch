# VQ-VAE（MNIST）

向量量化变分自编码器：

- 编码器把图像压缩为连续特征；
- `VectorQuantizer` 把每个特征映射到码本中最近的向量；
- 解码器从量化后的表示重建图像；
- 损失 = 重建损失 + VQ 损失 + commitment 损失。

首次运行会由 `torchvision` 自动下载 MNIST 到 `data/mnist/`（已忽略，不入库）。

## 文件

| 文件 | 内容 |
| --- | --- |
| `train.py` | `VectorQuantizer`、编码器/解码器、训练与重建可视化 |

## 运行

```bash
python -m examples.vqvae.train --epochs 5
```

公共参数见仓库根目录 README。

## 输出

默认写入 `outputs/vqvae/`：

- `vqvae_loss.png`
- `vqvae_reconstruction.png`
- `vqvae.pt`

## 说明

- 验证损失按验证 batch 数取平均（使用独立的 `v_i` 计数，而不是训练循环变量）。
- 码本大小、隐层宽度等由类构造函数的参数控制，可在 `train.py` 中调整。
