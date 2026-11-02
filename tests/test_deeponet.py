"""DeepONet 示例：前向形状、Excel 往返、积分算子与数据生成确定性。"""

from __future__ import annotations

import numpy as np
import torch

from examples.deeponet.data_build.build_1d_integral import (
    compute_integral,
    generate_dataset,
)
from examples.deeponet.data_build.common import save_to_excel
from examples.deeponet.dataload import load_from_excel
from examples.deeponet.model.deeponet import DeepONet
from examples.deeponet.model.fnn import FNNBaseline


def test_deeponet_forward_shape():
    """(u, y) 输入经 branch/trunk 内积后得到 (batch, 1)。"""
    model = DeepONet(branch_layers=[5, 8, 4], trunk_layers=[2, 8, 4])
    u = torch.rand(3, 5)
    y = torch.rand(3, 2)
    out = model((u, y))
    assert out.shape == (3, 1)


def test_fnn_baseline_query_dims():
    """1D 与 3D 查询点都能拼接进基线网络。"""
    u = torch.rand(4, 6)

    net_1d = FNNBaseline(m=6, query_dim=1, hidden_dim=8)
    assert net_1d(u, torch.rand(4, 1)).shape == (4, 1)

    net_3d = FNNBaseline(m=6, query_dim=3, hidden_dim=8)
    assert net_3d(u, torch.rand(4, 3)).shape == (4, 1)


def test_excel_roundtrip_both_branch_sheets(tmp_path):
    """u_function 与 f_function 两种 branch sheet 都能被读回。"""
    n, m = 8, 6
    branch = np.random.RandomState(0).rand(n, m).astype(np.float32)
    query = np.random.RandomState(1).rand(n, 1).astype(np.float32)
    output = np.random.RandomState(2).rand(n, 1).astype(np.float32)

    for sheet in ("u_function", "f_function"):
        path = tmp_path / f"{sheet}.xlsx"
        save_to_excel(branch, query, output, path, branch_sheet=sheet)

        u_back, y_back, s_back = load_from_excel(path)
        assert u_back.shape == (n, m)
        assert y_back.shape == (n, 1)
        assert s_back.shape == (n, 1)
        assert np.allclose(u_back, branch)
        assert np.allclose(y_back, query)
        assert np.allclose(s_back, output)


def test_compute_integral_constant_field():
    """常值场 u≡1 在 [0,y] 上的积分应等于 y（y 取在网格节点上）。"""
    x_grid = np.linspace(0.0, 2.0, 21)  # 步长 0.1
    u = np.ones((2, x_grid.size))
    y = np.array([[0.5], [1.0]])

    s = compute_integral(u, y, x_grid)

    assert np.allclose(s, y)


def test_generate_dataset_is_deterministic():
    """固定种子下生成的 (u, y, s) 完全一致。"""
    u_a, y_a, s_a = generate_dataset(4, 10, seed=0)
    u_b, y_b, s_b = generate_dataset(4, 10, seed=0)

    assert np.array_equal(u_a, u_b)
    assert np.array_equal(y_a, y_b)
    assert np.array_equal(s_a, s_b)
    assert u_a.shape == (4, 10)
    assert y_a.shape == (4, 1)
    assert s_a.shape == (4, 1)
