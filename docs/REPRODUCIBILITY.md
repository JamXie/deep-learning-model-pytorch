# 可复现性与冒烟运行

本仓库追求的是**可运行、可复现的代码流程**，而不是原论文数值的精确复现。所有入口：

1. 通过 `common.seed.set_seed(seed)` 固定 `random`、NumPy 与 PyTorch 的种子；
2. 通过 `common.cli.add_common_args` 暴露 `--seed`、`--device`、`--data-root`、
   `--output-dir`、`--limit-batches`；
3. 把图片与权重写入显式的输出目录（默认 `outputs/`，已被 `.gitignore` 排除）；
4. 使用 matplotlib 的无界面后端 `Agg`，服务器或 CI 上无需图形环境。

`set_seed(..., deterministic=False)` 只保证随机数序列一致，**不**强制 CUDA 的
确定性内核；在 GPU 上训练时结果与 CPU 可能有细微差异，这是预期行为。

## 1. 测试（无需下载、无需训练）

```bash
python -m pytest
```

预期：`77 passed`，CPU 约 3 秒。测试覆盖：

| 测试文件 | 覆盖点 |
| --- | --- |
| `test_common.py` | 种子可复现、路径解析、PNG 保存、公共参数 |
| `test_tabular.py` | CSV 读取、归一化、80/20 划分、ANN/GRNN/RBF 前向 |
| `test_gpr.py` | 三种 GPR 变体拟合与预测形状 |
| `test_generative.py` | U-Net 形状、加噪零强度、掩码因果性、PixelCNN/VQ-VAE 前向 |
| `test_transformer.py` | teacher forcing 前向、自回归生成形状 |
| `test_moving_mnist.py` | 合成数字序列的确定性与 schema |
| `test_deeponet.py` | 1D/2D 前向、Excel 往返、积分正确性、数据确定性 |
| `test_sciml_crash.py` | PINN 导数、硬约束、RK45、数据确定性、算子与 PINO 残差 |
| `test_imports.py` | 全部入口模块可导入且无副作用 |

## 2. 经典示例冒烟

```bash
python -m examples.tabular_regression.ann --epochs 1 --limit-batches 1 --output-dir outputs/smoke/tabular
python -m examples.tabular_regression.grnn --output-dir outputs/smoke/tabular
python -m examples.tabular_regression.rbf_network --epochs 1 --limit-batches 1 --output-dir outputs/smoke/tabular
python -m examples.tabular_regression.interpolation_rbf --output-dir outputs/smoke/tabular
python -m examples.tabular_regression.interpolation_griddata --output-dir outputs/smoke/tabular

python -m examples.gpr.gpr_basic --output-dir outputs/smoke/gpr
python -m examples.gpr.gpr_hyperparameters --output-dir outputs/smoke/gpr
python -m examples.gpr.gpr_uncertain_inputs --output-dir outputs/smoke/gpr
```

预期图片（写入指定 `--output-dir`）：`ann_regression.png`、`grnn_regression.png`、
`rbf_regression.png`、`interpolation_rbf.png`、`interpolation_griddata.png`、
`gpr_basic.png`、`gpr_hyperparameters.png`、`gpr_uncertain_inputs.png`。

## 3. 生成模型与 Transformer

首次运行会从 `torchvision` 下载 MNIST：

```bash
python -m examples.mnist_generative.mnist_diffusion --epochs 1 --limit-batches 2 --output-dir outputs/smoke/gen
python -m examples.mnist_generative.pixelcnn_mnist --epochs 1 --limit-batches 2 --output-dir outputs/smoke/gen
python -m examples.mnist_generative.vqvae_mnist --epochs 1 --limit-batches 2 --output-dir outputs/smoke/gen
```

预期：`diffusion_loss.png` / `diffusion_samples.png` / `diffusion_unet.pt`，
`pixelcnn_loss.png` / `pixelcnn_reconstruction.png` / `pixelcnn.pt`，
`vqvae_loss.png` / `vqvae_reconstruction.png` / `vqvae.pt`。

Moving MNIST 与 Transformer：

```bash
python -m examples.temporal_transformer.generate_moving_mnist --num-samples 8 --output outputs/smoke/moving_mnist.pt
python -m examples.temporal_transformer.visualize_moving_mnist --data-file outputs/smoke/moving_mnist.pt --index 0 --output-dir outputs/smoke/transformer
python -m examples.temporal_transformer.temporal_transformer --data-file outputs/smoke/moving_mnist.pt --epochs 1 --limit-batches 1 --output-dir outputs/smoke/transformer
```

预期：`moving_mnist_sample_0.png`、`transformer_loss.png`、
`transformer_prediction.png`、`temporal_transformer.pt`。

## 4. DeepONet 冒烟

```bash
# 生成小数据集（8 条训练 + 4 条验证 + 4 条外推）
python -m examples.deeponet.data_build.build_1d_integral --num-train 8 --num-val 4 --num-extra 4 --output-dir outputs/smoke/deeponet_1d

# 训练 1 个 epoch、1 个 batch
python -m examples.deeponet.train --task 1d --epochs 1 --limit-batches 1 \
  --data-dir outputs/smoke/deeponet_1d --output-dir outputs/smoke/deeponet_1d

# 验证
python -m examples.deeponet.validation --task 1d --split val \
  --data-dir outputs/smoke/deeponet_1d --output-dir outputs/smoke/deeponet_1d
```

2D / 3D 同理，把 `build_1d_integral` 换成 `build_2d_pendulum` /
`build_3d_diffusion`，并相应修改 `--task`。预期输出：`train.xlsx`、`val.xlsx`、
`extra.xlsx`、`deeponet_loss.png`、`deeponet.pth`、`deeponet_prediction_val.png`、
`deeponet_error_distribution_val.png`。

## 5. 碰撞安全 SciML 冒烟

```bash
# 单个 PINN 案例（20 次 Adam 迭代，仅验证流程）
python -m examples.sciml_crash.pinns.case_linear --iters 20 --output-dir outputs/smoke/sciml

# 小数据集 + 1 个 epoch 训练
python -m examples.sciml_crash.operator.data_generator --num-samples 8 --num-time 50 --output-dir outputs/smoke/sciml
python -m examples.sciml_crash.operator.train_operator --epochs 1 --limit-batches 2 \
  --data-path outputs/smoke/sciml/crash_dataset.pkl \
  --checkpoint outputs/smoke/sciml/deeponet_crash.pt \
  --output-dir outputs/smoke/sciml

# 评估与 PINO 对比
python -m examples.sciml_crash.operator.eval_operator \
  --checkpoint outputs/smoke/sciml/deeponet_crash.pt \
  --num-samples 10 --n-scenarios 10 --rk45-samples 5 \
  --output-dir outputs/smoke/sciml
python -m examples.sciml_crash.pino.benchmark \
  --checkpoint outputs/smoke/sciml/deeponet_crash.pt \
  --iters 3 --output-dir outputs/smoke/sciml
```

预期：`case_linear_results.png`、`crash_dataset.pkl`、`deeponet_crash.pt`、
`deeponet_predictions.png`、`pino_benchmark_results.png`。另外两个 PINN 案例是
`case_bilinear`（默认 8000 次迭代）与 `case_rate_dependent`（默认 6000 次）。

## 6. 关于数值

- 冒烟运行（1 个 epoch / 20 次迭代）只用于确认流程可跑，指标没有参考意义。
- 完整训练的超参数保留在各脚本的默认值中；本仓库未提交训练日志或论文数值。
- `docs/assets/` 中的图片来自作者早期实验，用于展示图表形态，不是当前代码的
  精确复现结果。
