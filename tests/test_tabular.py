"""表格回归示例：数据读取、归一化、划分与网络前向形状。"""

from __future__ import annotations

import numpy as np
import torch

from examples._shared.crash_data import (
    INPUT_COLUMNS,
    OUTPUT_COLUMNS,
    load_crash_frame,
    normalize_frame,
    split_dataset,
)
from examples.ann.train import ANN
from examples.grnn.run import GRNN
from examples.rbf_network.train import RBFNet


def test_load_crash_frame():
    """仓库自带 CSV 应能被读取并包含固定列。"""
    frame = load_crash_frame()
    assert frame.shape == (600, 11)
    for column in INPUT_COLUMNS + OUTPUT_COLUMNS:
        assert column in frame.columns


def test_normalize_and_split():
    """归一化后输入/输出形状正确，划分比例为 80/20。"""
    frame = load_crash_frame()
    x, y = normalize_frame(frame)
    assert x.shape == (600, 6)
    assert y.shape == (600, 3)

    x_train, x_val, y_train, y_val = split_dataset(x, y, seed=42)
    assert x_train.shape == (480, 6)
    assert x_val.shape == (120, 6)
    assert y_train.shape == (480, 3)
    assert y_val.shape == (120, 3)


def test_ann_forward_shape():
    net = ANN(input_dim=6, output_dim=3).eval()
    out = net(torch.rand(4, 6))
    assert out.shape == (4, 3)


def test_grnn_forward_shape():
    train_inputs = torch.rand(20, 6)
    train_targets = torch.rand(20, 3)
    model = GRNN(train_inputs, train_targets, sigma=0.16)
    out = model(torch.rand(5, 6))
    assert out.shape == (5, 3)
    assert torch.isfinite(out).all()


def test_rbf_forward_shape():
    x_train = np.random.RandomState(0).rand(10, 6).astype(np.float32)
    net = RBFNet(input_dim=6, hidden_dim=5, output_dim=3)
    net.initialize_centers(x_train, device=torch.device("cpu"))
    out = net(torch.tensor(x_train[:4]))
    assert out.shape == (4, 3)
