# 简化扩散模型（MNIST）

一个教学级的连续加噪 / 去噪示例：

- 用连续噪声强度 `amount` 把干净图像与随机噪声混合；
- 网络（`BasicUNet`）学习从含噪图像恢复干净图像；
- 采样时从纯噪声出发，分多步逐步向网络预测结果靠拢。

首次运行会由 `torchvision` 自动下载 MNIST 到 `data/mnist/`（已忽略，不入库）。

## 文件

| 文件 | 内容 |
| --- | --- |
| `train.py` | `BasicUNet`、加噪函数 `corrupt`、训练与采样 |

## 运行

```bash
python -m examples.diffusion.train --epochs 5
```

公共参数见仓库根目录 README；另有 `--n-steps`（采样步数，默认 40）与
`--sample-count`（采样图片数量，默认 16）。

## 输出

默认写入 `outputs/diffusion/`：

- `diffusion_loss.png`
- `diffusion_samples.png`
- `diffusion_unet.pt`

## 说明与局限

- 这是连续加噪/去噪的简化版，**不是**完整的 DDPM 训练与采样流程；用于展示
  “加噪 → 去噪”的基本思想。
- U-Net 为本仓库为演示目的编写的微型版本，层数与通道数都很小。
