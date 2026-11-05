# RBF 网络

RBF 网络把输入映射到一组高斯核上，再用线性层组合输出：

- 隐藏层中心由 KMeans 在训练集上初始化，初始化后不再更新；
- 核宽度 `sigma` 取所有中心两两距离的均值；
- 输出层为可训练的线性组合。

输入为 6 个结构/材料参数，输出为 3 项压溃性能；数据见
[data/README.md](../../data/README.md)。

## 文件

| 文件 | 内容 |
| --- | --- |
| `train.py` | KMeans 初始化中心、训练线性输出层、绘制散点图 |

## 运行

```bash
python -m examples.rbf_network.train --epochs 500 --hidden-dim 100
```

`--hidden-dim` 是 RBF 中心数量（默认 100）。公共参数见仓库根目录 README。

## 输出

默认写入 `outputs/rbf_network/rbf_regression.png`，终端打印训练/验证 R²。

## 说明

- `initialize_centers(..., device=...)` 会把 KMeans 在 CPU 上求得的中心显式放到
  目标设备，避免模型与中心设备不一致。
- 与 ANN、GRNN、插值基线共用
  [`examples/_shared/crash_data.py`](../_shared/crash_data.py) 的归一化与划分。
