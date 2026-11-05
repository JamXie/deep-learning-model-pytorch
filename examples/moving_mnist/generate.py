"""生成 Moving MNIST 序列数据集。

每个序列：一个 MNIST 数字在 64×64 画布上匀速移动并循环反弹。
固定随机种子后结果可复现，输出 schema：

    {
      "sequences": FloatTensor [N, L, 1, 64, 64],
      "conditions": [{"initial_position": [x, y], "velocity": [vx, vy]}, ...],
      "meta": {"num_samples": N, "seq_len": L, "canvas_size": 64,
               "image_size": 28, "seed": 42, "source": "MNIST train"}
    }

运行方式（在仓库根目录）：
    python -m examples.moving_mnist.generate --num-samples 600
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import torch
import torchvision
import torchvision.transforms as transforms

from common.cli import add_common_args
from common.paths import resolve_data_path
from common.seed import set_seed


def generate_sequence(
    digit: torch.Tensor,
    seq_len: int = 10,
    canvas_size: int = 64,
    image_size: int = 28,
    rng: np.random.RandomState | None = None,
) -> tuple[torch.Tensor, dict]:
    """让单个数字在画布上移动，返回 (L,1,H,W) 序列与初始条件。"""
    rng = rng if rng is not None else np.random.RandomState(0)

    x0 = int(rng.randint(0, canvas_size - image_size))
    y0 = int(rng.randint(0, canvas_size - image_size))
    vx0 = int(rng.randint(-2, 3))
    vy0 = int(rng.randint(-2, 3))

    x, y, vx, vy = x0, y0, vx0, vy0
    sequence = torch.zeros((seq_len, 1, canvas_size, canvas_size))

    for t in range(seq_len):
        # 先夹紧到合法范围，再贴图
        x = max(0, min(x, canvas_size - image_size))
        y = max(0, min(y, canvas_size - image_size))
        sequence[t, 0, x:x + image_size, y:y + image_size] = digit

        x += vx
        y += vy
        # 越界则反向
        if x < 0 or x + image_size > canvas_size:
            vx = -vx
        if y < 0 or y + image_size > canvas_size:
            vy = -vy

    condition = {
        "initial_position": [x0, y0],
        "velocity": [vx0, vy0],
    }
    return sequence, condition


def save_moving_mnist(
    output_path: Path,
    num_samples: int = 600,
    seq_len: int = 10,
    canvas_size: int = 64,
    image_size: int = 28,
    seed: int = 42,
    data_root: str | Path | None = None,
) -> dict:
    """从 MNIST 训练集生成 Moving MNIST 并保存为 .pt 文件。"""
    mnist_root = resolve_data_path("mnist", root=data_root)
    mnist = torchvision.datasets.MNIST(
        root=str(mnist_root),
        train=True,
        transform=transforms.ToTensor(),
        download=True,
    )

    rng = np.random.RandomState(seed)
    indices = rng.permutation(len(mnist))[:num_samples]

    sequences = []
    conditions = []
    for idx in indices:
        img, _ = mnist[int(idx)]
        sequence, condition = generate_sequence(
            img[0], seq_len=seq_len, canvas_size=canvas_size, image_size=image_size, rng=rng
        )
        sequences.append(sequence)
        conditions.append(condition)

    payload = {
        "sequences": torch.stack(sequences, dim=0),  # [N, L, 1, H, W]
        "conditions": conditions,
        "meta": {
            "num_samples": num_samples,
            "seq_len": seq_len,
            "canvas_size": canvas_size,
            "image_size": image_size,
            "seed": seed,
            "source": "MNIST train",
        },
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(payload, output_path)
    return payload


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="生成 Moving MNIST 序列")
    add_common_args(parser, epochs=False)
    parser.add_argument("--num-samples", type=int, default=600, help="序列数量")
    parser.add_argument("--seq-len", type=int, default=10, help="每个序列的帧数")
    parser.add_argument("--canvas-size", type=int, default=64, help="画布边长")
    parser.add_argument("--image-size", type=int, default=28, help="数字边长")
    parser.add_argument("--output", default=None, help="输出 .pt 路径")
    args = parser.parse_args(argv)

    set_seed(args.seed)
    output_path = (
        Path(args.output).expanduser()
        if args.output
        else resolve_data_path("moving_mnist", "moving_mnist_sequences.pt")
    )

    payload = save_moving_mnist(
        output_path,
        num_samples=args.num_samples,
        seq_len=args.seq_len,
        canvas_size=args.canvas_size,
        image_size=args.image_size,
        seed=args.seed,
        data_root=args.data_root,
    )
    print(f"Saved {payload['meta']['num_samples']} sequences to {output_path}")


if __name__ == "__main__":
    main()
