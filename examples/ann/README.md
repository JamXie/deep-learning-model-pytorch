# ANN：全连接回归

用 1CT 矩形薄壁梁的 6 个结构/材料参数预测 3 项压溃性能：

- 输入：`l, w, h, t, LoadAngle, sigma`
- 输出：`MeanCrashingForce, PeakCrashingForce, SpecificEnergyAbsorption`

网络为三层全连接 `6 -> 128 -> 64 -> 3`，带 BatchNorm、LeakyReLU、Dropout 与 Tanh
输出。数据为 600 条有限元仿真结果，列含义见
[data/README.md](../../data/README.md)。

## 文件

| 文件 | 内容 |
| --- | --- |
| `train.py` | 训练、验证与“预测值-真实值”散点图 |

## 运行

```bash
python -m examples.ann.train --epochs 300
```

公共参数（`--seed`、`--device`、`--data-root`、`--output-dir`、`--limit-batches`）
见仓库根目录 README；训练参数另有 `--epochs`、`--batch-size`。

## 输出

默认写入 `outputs/ann/ann_regression.png`。图中三个子图分别对应三个输出，包含
`x = y` 参考线；终端同时打印训练/验证 R²。

## 说明

- 数据读取、固定归一化范围与 80/20 划分来自
  [`examples/_shared/crash_data.py`](../_shared/crash_data.py)，因此可与 GRNN、
  RBF 网络和插值基线直接比较。
- 冒烟运行（`--epochs 1 --limit-batches 1`）只为验证流程，指标无参考意义。
