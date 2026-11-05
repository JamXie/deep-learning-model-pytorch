"""用编码器-解码器 Transformer 预测 Moving MNIST 的后续帧。

设计（相对原始算例做了必要重写，原因见 docs/FIXED_ISSUES.md）：
  - 每帧 64×64 先展平为 4096 维，再用 Linear 映射到 d_model，避免
    原始代码直接拿 4096 维当 d_model 导致注意力层过大；
  - 编码器读取历史帧，解码器输入 [SOS, target[:-1]] 做 teacher forcing；
  - 推理时用 5 帧历史自回归生成后续帧；
  - 默认 d_model=128、nhead=4、num_layers=2、历史 5 帧、预测 5 帧。

运行方式（在仓库根目录，先运行 examples.moving_mnist.generate）：
    python -m examples.transformer.train --epochs 10
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset, random_split

from common.cli import add_common_args
from common.device import get_device
from common.paths import resolve_data_path, resolve_output_dir
from common.plotting import save_figure, use_headless_backend
from common.seed import set_seed


class PositionalEncoding(nn.Module):
    """可学习的位置编码，加到帧 token 上。"""

    def __init__(self, d_model: int, max_len: int = 64) -> None:
        super().__init__()
        self.embedding = nn.Embedding(max_len, d_model)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        positions = torch.arange(x.size(1), device=x.device)
        return x + self.embedding(positions).unsqueeze(0)


class TemporalTransformer(nn.Module):
    """帧级编码器-解码器 Transformer。"""

    def __init__(
        self,
        image_size: int = 64,
        d_model: int = 128,
        nhead: int = 4,
        num_layers: int = 2,
        max_len: int = 32,
    ) -> None:
        super().__init__()
        self.image_size = image_size
        self.d_model = d_model
        self.max_len = max_len

        self.frame_encoder = nn.Linear(image_size * image_size, d_model)
        self.pos_encoding = PositionalEncoding(d_model, max_len=max_len)
        self.sos = nn.Parameter(torch.zeros(1, 1, d_model))

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=nhead, dim_feedforward=d_model * 4, dropout=0.1, batch_first=True
        )
        decoder_layer = nn.TransformerDecoderLayer(
            d_model=d_model, nhead=nhead, dim_feedforward=d_model * 4, dropout=0.1, batch_first=True
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.decoder = nn.TransformerDecoder(decoder_layer, num_layers=num_layers)
        self.output = nn.Linear(d_model, image_size * image_size)

    def encode_frames(self, frames: torch.Tensor) -> torch.Tensor:
        """[B,T,1,H,W] -> [B,T,d_model]。"""
        tokens = self.frame_encoder(frames.flatten(2))
        return self.pos_encoding(tokens)

    def decode_tokens(self, tokens: torch.Tensor, memory: torch.Tensor) -> torch.Tensor:
        """对解码器输入做掩码自注意力，返回 [B,T,d_model]。"""
        length = tokens.size(1)
        mask = nn.Transformer.generate_square_subsequent_mask(length, device=tokens.device)
        return self.decoder(tokens, memory, tgt_mask=mask)

    def forward(self, src: torch.Tensor, tgt: torch.Tensor) -> torch.Tensor:
        """Teacher forcing 前向：src 为历史帧，tgt 为待预测帧。"""
        batch = src.size(0)
        memory = self.encoder(self.encode_frames(src))

        # 解码器输入为 [SOS, tgt_0, ..., tgt_{T-2}]
        tgt_tokens = self.encode_frames(tgt[:, :-1])
        dec_in = torch.cat([self.sos.expand(batch, 1, -1), tgt_tokens], dim=1)
        dec_out = self.decode_tokens(dec_in, memory)

        logits = self.output(dec_out)
        return torch.sigmoid(logits).view(batch, tgt.size(1), 1, self.image_size, self.image_size)

    @torch.no_grad()
    def generate(self, src: torch.Tensor, n_frames: int) -> torch.Tensor:
        """自回归生成：只用历史帧，逐帧预测。"""
        batch = src.size(0)
        memory = self.encoder(self.encode_frames(src))
        generated: list[torch.Tensor] = []

        for _ in range(n_frames):
            if not generated:
                dec_in = self.sos.expand(batch, 1, -1)
            else:
                prev = torch.stack(generated, dim=1)  # [B,T,1,H,W]
                dec_in = torch.cat([self.sos.expand(batch, 1, -1), self.encode_frames(prev)], dim=1)

            dec_out = self.decode_tokens(dec_in, memory)
            next_frame = torch.sigmoid(self.output(dec_out[:, -1]))
            generated.append(next_frame.view(batch, 1, self.image_size, self.image_size))

        return torch.stack(generated, dim=1)


def load_sequences(data_path: Path) -> torch.Tensor:
    if not data_path.exists():
        raise FileNotFoundError(
            f"未找到 {data_path}，请先运行：\n"
            "  python -m examples.moving_mnist.generate"
        )
    data = torch.load(data_path, map_location="cpu")
    return data["sequences"].float()


def plot_comparison(src: torch.Tensor, true: torch.Tensor, pred: torch.Tensor, path) -> None:
    """历史帧 + 真实后续帧 + 预测后续帧，三行网格。"""
    src = src[0].cpu()
    true = true[0].cpu()
    pred = pred[0].cpu()
    columns = src.shape[0] + true.shape[0]
    fig, axes = plt.subplots(3, columns, figsize=(1.5 * columns, 5))
    for ax in axes.ravel():
        ax.axis("off")

    for t in range(src.shape[0]):
        axes[0, t].imshow(src[t, 0], cmap="gray", vmin=0, vmax=1)
        axes[0, t].set_title("history")
    for t in range(true.shape[0]):
        axes[1, src.shape[0] + t].imshow(true[t, 0], cmap="gray", vmin=0, vmax=1)
        axes[1, src.shape[0] + t].set_title("true")
        axes[2, src.shape[0] + t].imshow(pred[t, 0], cmap="gray", vmin=0, vmax=1)
        axes[2, src.shape[0] + t].set_title("pred")
    save_figure(fig, path)


def main(argv: list[str] | None = None) -> None:
    use_headless_backend()
    parser = argparse.ArgumentParser(description="Moving MNIST 时序 Transformer")
    add_common_args(parser)
    parser.add_argument("--data-file", default=None, help="Moving MNIST .pt 路径")
    parser.add_argument("--history", type=int, default=5, help="输入历史帧数")
    parser.add_argument("--horizon", type=int, default=5, help="预测帧数")
    parser.add_argument("--d-model", type=int, default=128, help="Transformer 隐层维度")
    parser.add_argument("--nhead", type=int, default=4, help="注意力头数")
    parser.add_argument("--num-layers", type=int, default=2, help="编码器/解码器层数")
    args = parser.parse_args(argv)

    set_seed(args.seed)
    device = get_device(args.device)
    output_dir = resolve_output_dir(args.output_dir, "transformer")

    data_path = (
        Path(args.data_file).expanduser()
        if args.data_file
        else resolve_data_path("moving_mnist", "moving_mnist_sequences.pt")
    )
    sequences = load_sequences(data_path)

    required = args.history + args.horizon
    if sequences.size(1) < required:
        raise ValueError(f"序列长度 {sequences.size(1)} 小于 history+horizon={required}")

    inputs = sequences[:, :args.history]
    targets = sequences[:, args.history:required]
    dataset = TensorDataset(inputs, targets)

    train_size = int(0.8 * len(dataset))
    val_size = len(dataset) - train_size
    train_ds, val_ds = random_split(dataset, [train_size, val_size])

    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False)

    model = TemporalTransformer(
        image_size=sequences.size(-1),
        d_model=args.d_model,
        nhead=args.nhead,
        num_layers=args.num_layers,
    ).to(device)
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)

    train_losses: list[float] = []
    val_losses: list[float] = []
    for epoch in range(args.epochs):
        model.train()
        train_sum = 0.0
        for step, (src, tgt) in enumerate(train_loader):
            if args.limit_batches is not None and step >= args.limit_batches:
                break
            src, tgt = src.to(device), tgt.to(device)
            optimizer.zero_grad()
            loss = criterion(model(src, tgt), tgt)
            loss.backward()
            optimizer.step()
            train_sum += loss.item()
        train_losses.append(train_sum / (step + 1))

        model.eval()
        val_sum = 0.0
        with torch.no_grad():
            for v_i, (src, tgt) in enumerate(val_loader):
                if args.limit_batches is not None and v_i >= max(1, args.limit_batches // 2):
                    break
                src, tgt = src.to(device), tgt.to(device)
                val_sum += criterion(model(src, tgt), tgt).item()
        val_losses.append(val_sum / (v_i + 1))
        print(f"epoch {epoch + 1}/{args.epochs} train={train_losses[-1]:.6f} val={val_losses[-1]:.6f}")

    fig, ax = plt.subplots(figsize=(6, 4))
    ax.plot(range(1, len(train_losses) + 1), train_losses, "r", label="train")
    ax.plot(range(1, len(val_losses) + 1), val_losses, "b", label="val")
    ax.set_xlabel("epoch")
    ax.set_ylabel("MSE loss")
    ax.set_title("temporal transformer")
    ax.legend()
    ax.grid(alpha=0.3)
    save_figure(fig, output_dir / "transformer_loss.png")

    # 自回归生成一个验证样本并保存对比图
    model.eval()
    src, true = next(iter(val_loader))
    src = src[:1].to(device)
    pred = model.generate(src, args.horizon)
    plot_comparison(src.cpu(), true[:1], pred.cpu(), output_dir / "transformer_prediction.png")

    torch.save(model.state_dict(), output_dir / "temporal_transformer.pt")
    print(f"Saved outputs to {output_dir}")


if __name__ == "__main__":
    main()
