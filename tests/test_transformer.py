"""时序 Transformer：teacher forcing 前向与自回归生成的形状。"""

from __future__ import annotations

import torch

from examples.temporal_transformer.temporal_transformer import TemporalTransformer


def _small_model() -> TemporalTransformer:
    return TemporalTransformer(
        image_size=16, d_model=32, nhead=4, num_layers=1, max_len=16
    ).eval()


def test_teacher_forcing_forward_shape():
    """历史 5 帧、目标 5 帧时，输出应与目标同形状且取值在 [0,1]。"""
    model = _small_model()
    src = torch.rand(2, 5, 1, 16, 16)
    tgt = torch.rand(2, 5, 1, 16, 16)

    with torch.no_grad():
        out = model(src, tgt)

    assert out.shape == (2, 5, 1, 16, 16)
    assert out.min() >= 0.0
    assert out.max() <= 1.0


def test_autoregressive_generate_shape():
    """只用历史帧自回归生成 5 帧，形状应正确。"""
    model = _small_model()
    src = torch.rand(2, 5, 1, 16, 16)

    pred = model.generate(src, n_frames=5)

    assert pred.shape == (2, 5, 1, 16, 16)
