# Moving MNIST 与时序 Transformer

一个数字在 64×64 画布上匀速移动、碰边反弹，形成 10 帧序列。Transformer 读取前 5 帧
历史，预测后 5 帧。

## 文件

| 文件 | 内容 |
| --- | --- |
| `generate_moving_mnist.py` | 从 MNIST 训练集生成序列数据集 |
| `visualize_moving_mnist.py` | 把一条序列的所有帧拼成一张图 |
| `temporal_transformer.py` | 帧级编码器-解码器 Transformer |

## 生成数据

```bash
python -m examples.temporal_transformer.generate_moving_mnist --num-samples 600 --seq-len 10
```

默认输出 `data/moving_mnist/moving_mnist_sequences.pt`，schema 见
`docs/DATA.md`。生成过程使用显式 `RandomState(seed)`，同一种子结果一致。

```bash
python -m examples.temporal_transformer.visualize_moving_mnist --index 0
```

## 训练与预测

```bash
python -m examples.temporal_transformer.temporal_transformer --epochs 10
```

可调参数：`--history`（默认 5）、`--horizon`（默认 5）、`--d-model`（128）、
`--nhead`（4）、`--num-layers`（2）、`--data-file`（自定义数据集路径）。

## 模型设计

- 每帧 64×64 先展平为 4096 维，经 `Linear(4096, d_model)` 编码为 token；
- 可学习位置编码加到帧 token 上；
- 编码器读取历史帧；
- 解码器输入为 `[SOS, target[:-1]]`，使用因果掩码做 teacher forcing；
- 输出头 `Linear(d_model, 4096)` + sigmoid，reshape 回 `[B, T, 1, 64, 64]`；
- 推理时 `generate()` 从 SOS 开始逐帧自回归生成。

> 与原始脚本相比，这里增加了帧编码器并统一了训练/推理路径。原因见
> `docs/FIXED_ISSUES.md` 第 4 节。

## 输出

默认写入 `outputs/temporal_transformer/`：

- `transformer_loss.png`：训练/验证 MSE 曲线
- `transformer_prediction.png`：历史帧 + 真实后续帧 + 预测后续帧
- `temporal_transformer.pt`：模型权重

## 说明

- 没有生成 GIF；如需动画，可把 `transformer_prediction.png` 或逐帧图片自行合成。
- 训练 10 个 epoch 只用于演示流程，预测质量有限。
