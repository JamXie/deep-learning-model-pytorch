# 时序 Transformer

用编码器-解码器 Transformer 预测 Moving MNIST 的后续 5 帧。先按
[moving_mnist 的说明](../moving_mnist/README.md)生成数据。

## 文件

| 文件 | 内容 |
| --- | --- |
| `train.py` | `TemporalTransformer`、训练、teacher forcing 与自回归生成 |

## 运行

```bash
python -m examples.moving_mnist.generate --num-samples 600
python -m examples.transformer.train --epochs 10
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

## 输出

默认写入 `outputs/transformer/`：

- `transformer_loss.png`：训练/验证 MSE 曲线
- `transformer_prediction.png`：历史帧 + 真实后续帧 + 预测后续帧
- `temporal_transformer.pt`：模型权重

## 说明

- 与原始脚本相比，这里增加了帧编码器并统一了训练/推理路径。原因见
  [docs/FIXED_ISSUES.md](../../docs/FIXED_ISSUES.md) 第 4 节。
- 训练 10 个 epoch 只用于演示流程，预测质量有限。
- 没有生成 GIF；如需动画，可把 `transformer_prediction.png` 或逐帧图片自行合成。
