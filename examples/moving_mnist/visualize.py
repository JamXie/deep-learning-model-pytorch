"""把一条 Moving MNIST 序列的所有帧拼成网格图保存。

运行方式（在仓库根目录，先运行 examples.moving_mnist.generate）：
    python -m examples.moving_mnist.visualize --index 0
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import torch

from common.cli import add_common_args
from common.paths import resolve_data_path, resolve_output_dir
from common.plotting import save_figure, use_headless_backend


def main(argv: list[str] | None = None) -> None:
    use_headless_backend()
    parser = argparse.ArgumentParser(description="可视化一条 Moving MNIST 序列")
    add_common_args(parser, epochs=False)
    parser.add_argument("--data-file", default=None, help="数据集 .pt 路径")
    parser.add_argument("--index", type=int, default=0, help="样本序号")
    args = parser.parse_args(argv)

    data_path = (
        Path(args.data_file).expanduser()
        if args.data_file
        else resolve_data_path("moving_mnist", "moving_mnist_sequences.pt")
    )
    if not data_path.exists():
        raise FileNotFoundError(
            f"未找到 {data_path}，请先运行：\n"
            "  python -m examples.moving_mnist.generate"
        )

    data = torch.load(data_path, map_location="cpu")
    sequences = data["sequences"]
    if not 0 <= args.index < len(sequences):
        raise IndexError(f"--index 超出范围：数据集共 {len(sequences)} 条")

    frames = sequences[args.index]
    count = frames.shape[0]
    fig, axes = plt.subplots(1, count, figsize=(1.6 * count, 2))
    axes = list(axes) if count > 1 else [axes]
    for t, ax in enumerate(axes):
        ax.imshow(frames[t, 0].numpy(), cmap="gray", vmin=0, vmax=1)
        ax.set_title(f"t={t}")
        ax.axis("off")
    fig.suptitle(f"sample {args.index} / {len(sequences)}")
    output_dir = resolve_output_dir(args.output_dir, "moving_mnist")
    save_figure(fig, output_dir / f"moving_mnist_sample_{args.index}.png")
    print(f"Saved preview to {output_dir}")


if __name__ == "__main__":
    main()
