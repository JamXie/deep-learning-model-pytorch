# GRNN：广义回归神经网络

GRNN 没有反向传播训练：每个训练样本就是一个高斯核中心，预测值是训练目标的核加权
平均。`--sigma` 控制核宽度，默认 `0.16`。

输入为 6 个结构/材料参数，输出为 3 项压溃性能；数据与列含义见
[data/README.md](../../data/README.md)。

## 文件

| 文件 | 内容 |
| --- | --- |
| `run.py` | 构建 GRNN、预测并绘制散点图（无训练循环） |

## 运行

```bash
python -m examples.grnn.run --sigma 0.16
```

公共参数见仓库根目录 README。

## 输出

默认写入 `outputs/grnn/grnn_regression.png`，终端打印训练/验证 R²。

## 说明

- GRNN 需要保存全部训练样本，样本量很大时内存/显存开销明显；本数据集只有 480 条
  训练样本，开销很小。
- 与 ANN、RBF 网络共用
  [`examples/_shared/crash_data.py`](../_shared/crash_data.py) 的归一化与划分。
