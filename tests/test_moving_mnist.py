"""Moving MNIST 生成：用合成数字张量验证固定种子与输出 schema（不依赖 MNIST）。"""

from __future__ import annotations

import numpy as np
import torch

from examples.temporal_transformer.generate_moving_mnist import generate_sequence


def _synthetic_digit(size: int = 8) -> torch.Tensor:
    """构造一个取值在 (0,1] 的合成“数字”补丁。"""
    torch.manual_seed(0)
    return torch.rand(size, size)


def test_generate_sequence_is_deterministic():
    """相同随机种子应产生完全相同的序列与初始条件。"""
    digit = _synthetic_digit()

    seq_a, cond_a = generate_sequence(
        digit, seq_len=5, canvas_size=32, image_size=8, rng=np.random.RandomState(0)
    )
    seq_b, cond_b = generate_sequence(
        digit, seq_len=5, canvas_size=32, image_size=8, rng=np.random.RandomState(0)
    )

    assert torch.equal(seq_a, seq_b)
    assert cond_a == cond_b


def test_generate_sequence_schema():
    """序列形状为 (L,1,H,W)，初始条件包含位置与速度。"""
    digit = _synthetic_digit()
    sequence, condition = generate_sequence(
        digit, seq_len=5, canvas_size=32, image_size=8, rng=np.random.RandomState(0)
    )

    assert sequence.shape == (5, 1, 32, 32)
    assert set(condition) == {"initial_position", "velocity"}

    x0, y0 = condition["initial_position"]
    vx0, vy0 = condition["velocity"]
    assert 0 <= x0 <= 32 - 8
    assert 0 <= y0 <= 32 - 8
    assert -2 <= vx0 <= 2
    assert -2 <= vy0 <= 2

    # 每一帧都应确实贴上了数字（非全零）
    for t in range(sequence.size(0)):
        assert sequence[t].sum() > 0
