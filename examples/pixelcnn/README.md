# PixelCNN（MNIST，重建式）

掩码卷积自编码器：

- `MaskedCNN` 使用类型 A/B 因果掩码，保证位置 `(i, j)` 的输出只依赖其上方/左侧输入；
- `PixelCNN` 堆叠若干掩码卷积，输出 1 通道重建结果；
- 训练目标为 MNIST 重建（MSE）。

首次运行会由 `torchvision` 自动下载 MNIST 到 `data/mnist/`（已忽略，不入库）。

## 文件

| 文件 | 内容 |
| --- | --- |
| `train.py` | `MaskedCNN`、`PixelCNN`、80/10/10 划分与训练 |

## 运行

```bash
python -m examples.pixelcnn.train --epochs 5
```

公共参数见仓库根目录 README。

## 输出

默认写入 `outputs/pixelcnn/`：

- `pixelcnn_loss.png`
- `pixelcnn_reconstruction.png`
- `pixelcnn.pt`

## 说明与局限

- 本示例保留原算例的**无条件、重建式**形式：掩码保证因果性，但训练目标是重建而非
  逐像素自回归采样，也没有类别标签条件。
- 若需要一个真正的条件 PixelCNN，应把输出端改为 256 类 softmax，并按像素顺序采样。
- 掩码在每次前向时用 `self.weight * self.mask` 临时计算，不原地修改参数。
