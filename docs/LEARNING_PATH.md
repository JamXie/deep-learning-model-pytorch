# 学习路径

本路径假设你已经会 Python 与 PyTorch 基础：张量、自动求导、`nn.Module` 和基本训练
循环。每个阶段先读对应 README，再跑冒烟命令，最后按默认轮数完整训练。

所有命令都在仓库根目录执行；公共参数说明见根目录 README。冒烟命令只确认流程可跑，
指标没有参考意义。

## 0. 环境与自检

```bash
python -m venv .venv
# Windows: .venv\Scripts\activate
# Linux/macOS: source .venv/bin/activate
pip install -r requirements-dev.txt
python -m pytest
```

预期 `77 passed`。测试不下载数据、不训练模型。

## 1. 表格回归：参数模型与非参数模型

先看 [examples/README.md](../examples/README.md#基础回归与插值) 与
[data/README.md](../data/README.md)。四个示例共用同一份 600 条仿真数据、同一套
固定归一化范围与同一个 80/20 划分，因此 R² 可以直接横向比较。

建议顺序：

1. **ANN**（[README](../examples/ann/README.md)）：最标准的全连接回归。
   ```bash
   python -m examples.ann.train --epochs 300
   ```
2. **GRNN**（[README](../examples/grnn/README.md)）：没有训练过程，训练样本就是
   核中心。观察 `--sigma` 对结果的影响。
   ```bash
   python -m examples.grnn.run --sigma 0.16
   ```
3. **RBF 网络**（[README](../examples/rbf_network/README.md)）：KMeans 选中心，
   线性输出层可训练。
   ```bash
   python -m examples.rbf_network.train --epochs 500 --hidden-dim 100
   ```
4. **插值基线**（[README](../examples/interpolation/README.md)）：RBF 插值与
   griddata 最近邻。
   ```bash
   python -m examples.interpolation.rbf
   python -m examples.interpolation.griddata
   ```

关注点：参数模型与非参数模型的差别；为什么固定归一化范围便于比较；`griddata` 的
`linear` / `cubic` 为什么会在凸包外产生 NaN。

## 2. 高斯过程回归

[examples/gpr/README.md](../examples/gpr/README.md)。三个脚本共用平方指数核：

```bash
python -m examples.gpr.gpr_basic
python -m examples.gpr.gpr_hyperparameters
python -m examples.gpr.gpr_uncertain_inputs
```

关注点：后验均值与 95% 置信带的含义；边际似然如何学习长度尺度与信号方差；观测噪声
方差如何影响预测区间。

## 3. 生成模型

首次运行会由 `torchvision` 下载 MNIST 到 `data/mnist/`。建议按“重建 → 因果 →
离散表示”的顺序：

1. **简化扩散**（[README](../examples/diffusion/README.md)）
   ```bash
   python -m examples.diffusion.train --epochs 5
   ```
2. **PixelCNN**（[README](../examples/pixelcnn/README.md)）
   ```bash
   python -m examples.pixelcnn.train --epochs 5
   ```
3. **VQ-VAE**（[README](../examples/vqvae/README.md)）
   ```bash
   python -m examples.vqvae.train --epochs 5
   ```

关注点：重建目标与生成目标的区别；掩码卷积的因果性；码本量化与 commitment loss。
注意 PixelCNN 示例是**无条件、重建式**的，不是逐像素自回归采样。

## 4. 序列模型

先生成 Moving MNIST，再训练 Transformer：

```bash
python -m examples.moving_mnist.generate --num-samples 600
python -m examples.moving_mnist.visualize --index 0
python -m examples.transformer.train --epochs 10
```

对应 README：[moving_mnist](../examples/moving_mnist/README.md)、
[transformer](../examples/transformer/README.md)。

关注点：帧编码器为什么必要；训练时的 teacher forcing 与推理时的自回归生成有何区别；
10 个 epoch 只用于演示流程。

## 5. 算子学习

[examples/deeponet/README.md](../examples/deeponet/README.md)。先选一个任务（`1d`
最简单），完成“生成数据 → 训练 → 验证”闭环：

```bash
python -m examples.deeponet.data_build.build_1d_integral --num-train 2000 --num-val 500
python -m examples.deeponet.train --task 1d --model deeponet --epochs 200
python -m examples.deeponet.validation --task 1d --split val
```

关注点：分支网络编码输入函数、主干网络编码查询坐标；DeepONet 与 FNN 基线的差别。

## 6. 科学机器学习

[examples/sciml_crash/README.md](../examples/sciml_crash/README.md)。按工作流顺序：

```bash
python -m examples.sciml_crash.pinns.case_linear --iters 5000
python -m examples.sciml_crash.operator.data_generator --num-samples 500
python -m examples.sciml_crash.operator.train_operator --epochs 300
python -m examples.sciml_crash.operator.eval_operator
python -m examples.sciml_crash.pino.benchmark --iters 200
```

关注点：PINN 强形式与能量形式的区别；RK45 参考解的作用；DeepONet 学习参数到响应的
映射；PINO 的物理残差如何微调预测。

## 冒烟与完整训练

- 冒烟：训练类用 `--epochs 1 --limit-batches 1`，PINN 用 `--iters 20`，
  数据生成用 `--num-samples 8` 等小规模参数；
- 完整训练：使用各脚本默认值，CPU 上可能需要较长时间；
- 所有输出都在 `outputs/`（或显式 `--output-dir`）下，已被 `.gitignore` 排除。

各步骤的详细命令与预期文件见
[docs/REPRODUCIBILITY.md](REPRODUCIBILITY.md)。
