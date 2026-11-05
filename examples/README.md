# 模型索引

本目录按“一个模型一个目录”组织单一模型示例；DeepONet 与碰撞安全 SciML 是多文件
项目或连续工作流，保留工程目录。每个目录都有独立 README。

建议先看 [docs/LEARNING_PATH.md](../docs/LEARNING_PATH.md)，按顺序从表格回归走到
科学机器学习。

## 基础回归与插值

共同任务：用 1CT 矩形薄壁梁的 6 个结构/材料参数预测 3 项压溃性能。数据与列含义见
[data/README.md](../data/README.md)；CSV 读取、归一化与划分在
[`_shared/crash_data.py`](_shared/crash_data.py)。

| 目录 | 模型 | 一句话说明 | 命令 |
| --- | --- | --- | --- |
| [`ann/`](ann/) | ANN | 三层全连接网络，BatchNorm + Dropout | `python -m examples.ann.train --epochs 300` |
| [`grnn/`](grnn/) | GRNN | 以训练样本为核中心，无需反向传播 | `python -m examples.grnn.run` |
| [`rbf_network/`](rbf_network/) | RBF 网络 | KMeans 选中心，线性输出层可训练 | `python -m examples.rbf_network.train --epochs 500` |
| [`interpolation/`](interpolation/) | RBF / griddata 插值 | 非参数基线，便于与神经网络对比 | `python -m examples.interpolation.rbf` |

## 概率模型

| 目录 | 模型 | 一句话说明 | 命令 |
| --- | --- | --- | --- |
| [`gpr/`](gpr/) | 高斯过程回归 | `cos(x)` 回归、超参数优化与观测噪声 | `python -m examples.gpr.gpr_basic` |

## 生成模型

| 目录 | 模型 | 一句话说明 | 命令 |
| --- | --- | --- | --- |
| [`diffusion/`](diffusion/) | 简化扩散 U-Net | 连续加噪 / 去噪的教学版本 | `python -m examples.diffusion.train --epochs 5` |
| [`pixelcnn/`](pixelcnn/) | 掩码卷积 PixelCNN | 因果掩码 + 重建目标 | `python -m examples.pixelcnn.train --epochs 5` |
| [`vqvae/`](vqvae/) | VQ-VAE | 码本量化 + 重建 | `python -m examples.vqvae.train --epochs 5` |

## 序列模型

| 目录 | 模型 | 一句话说明 | 命令 |
| --- | --- | --- | --- |
| [`moving_mnist/`](moving_mnist/) | Moving MNIST | 生成“数字移动”序列并可视化 | `python -m examples.moving_mnist.generate` |
| [`transformer/`](transformer/) | 时序 Transformer | 读取 5 帧历史，预测后 5 帧 | `python -m examples.transformer.train --epochs 10` |

## 算子学习与科学机器学习

| 目录 | 内容 | 命令 |
| --- | --- | --- |
| [`deeponet/`](deeponet/) | DeepONet + FNN 基线；一维积分、二维单摆、一维扩散 | 见目录 README |
| [`sciml_crash/`](sciml_crash/) | PINN → 碰撞算子 → PINO 微调 | 见目录 README |

## 通用约定

- 所有入口都在仓库根目录以 `python -m ...` 运行；
- 公用参数 `--seed`、`--device`、`--data-root`、`--output-dir`、`--limit-batches`
  见根目录 README；
- 训练类示例可用 `--epochs 1 --limit-batches 1` 做冒烟；
- 输出默认写到 `outputs/<模型名>/`，已被 `.gitignore` 排除。
