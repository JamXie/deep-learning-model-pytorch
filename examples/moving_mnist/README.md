# Moving MNIST 数据生成

每个序列：一个 MNIST 数字在 64×64 画布上匀速移动、碰边反弹，默认 10 帧。Transformer
示例读取前 5 帧，预测后 5 帧。

## 文件

| 文件 | 内容 |
| --- | --- |
| `generate.py` | 从 MNIST 训练集生成序列数据集 |
| `visualize.py` | 把一条序列的所有帧拼成网格图 |

## 生成数据

```bash
python -m examples.moving_mnist.generate --num-samples 600 --seq-len 10
```

默认输出 `data/moving_mnist/moving_mnist_sequences.pt`。固定 schema（`sequences`、
`conditions`、`meta`）见 [data/README.md](../../data/README.md) 第 3 节。生成过程使用
显式 `RandomState(seed)`，同一种子结果一致。

## 可视化

```bash
python -m examples.moving_mnist.visualize --index 0
```

默认读取 `data/moving_mnist/moving_mnist_sequences.pt`，图片写入
`outputs/moving_mnist/moving_mnist_sample_0.png`。

## 说明

- 可用 `--data-file` 指定其他数据集路径，用 `--output` 指定生成位置。
- 本目录只负责数据；训练与预测见 [../transformer/README.md](../transformer/README.md)。
