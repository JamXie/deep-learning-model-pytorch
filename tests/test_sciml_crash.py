"""碰撞安全 SciML：PINN 自动微分、RK45 参考解、数据生成、算子与 PINO 残差。"""

from __future__ import annotations

import numpy as np
import torch

from examples.sciml_crash.operator.data_generator import generate_crash_dataset
from examples.sciml_crash.operator.deeponet_model import DeepONet, ParameterNormalizer
from examples.sciml_crash.pinns.models import DerivativeNetwork, FNN, HardConstraintWrapper
from examples.sciml_crash.pinns.utils import linear_spring, solve_spring_mass_rk45
from examples.sciml_crash.pino.pino_pipeline import compute_physics_residual


def test_derivative_network_shapes():
    """t -> (x, v, a) 三者形状均为 (batch, 1)。"""
    net = DerivativeNetwork(FNN(hidden_layers=[8, 8]))
    t = torch.linspace(0.0, 1.0, 5).reshape(-1, 1)

    x, v, a = net(t)

    assert x.shape == v.shape == a.shape == (5, 1)


def test_hard_constraint_satisfies_initial_condition():
    """硬约束包装器在 t=0 处满足 x(0)=x0。"""
    wrapper = HardConstraintWrapper(FNN(hidden_layers=[8]), x0=1.0, v0=2.0)
    x = wrapper(torch.zeros(3, 1))
    assert torch.allclose(x, torch.ones(3, 1), atol=1e-6)


def test_rk45_matches_cosine():
    """m=1、c=0、k=1、无外力时，解析解为 x=cos(t)。"""
    t, x, v, _ = solve_spring_mass_rk45(
        m=1.0,
        c=0.0,
        k_func=lambda x: linear_spring(x, k=1.0),
        F_func=lambda t: 0.0,
        x0=1.0,
        v0=0.0,
        t_span=(0.0, 1.0),
        n_points=50,
    )

    assert np.allclose(x, np.cos(t), atol=1e-4)
    assert np.allclose(v, -np.sin(t), atol=1e-3)


def test_crash_dataset_deterministic():
    """3 个样本、固定种子时两次生成结果完全一致。"""
    data_a = generate_crash_dataset(n_samples=3, n_time_points=20, seed=42)
    data_b = generate_crash_dataset(n_samples=3, n_time_points=20, seed=42)

    assert np.array_equal(data_a["params"], data_b["params"])
    assert np.array_equal(data_a["x_data"], data_b["x_data"])
    assert data_a["params"].shape == (3, 6)
    assert data_a["x_data"].shape == (3, 20)


def test_operator_predict_trajectory_shape():
    """新参数下的完整时间历程形状为 (batch, n_time)。"""
    model = DeepONet(n_params=6, p=8, branch_dims=[16, 16], trunk_dims=[16, 16])
    params = torch.rand(2, 6)
    t_grid = torch.linspace(0.0, 0.12, 20).reshape(-1, 1)

    x_pred = model.predict_trajectory(params, t_grid)

    assert x_pred.shape == (2, 20)
    assert torch.isfinite(x_pred).all()


def test_pino_physics_residual_finite():
    """PINO 物理残差形状正确且数值有限。"""
    model = DeepONet(n_params=6, p=8, branch_dims=[16, 16], trunk_dims=[16, 16])
    mins = np.array([800.0, 300.0, 150e3, 10e3, 0.02, -20.0], dtype=np.float32)
    maxs = np.array([2000.0, 1000.0, 500e3, 50e3, 0.06, -5.0], dtype=np.float32)
    normalizer = ParameterNormalizer(mins, maxs)

    params_raw = torch.tensor(
        [
            [1200.0, 500.0, 300e3, 30e3, 0.04, -12.0],
            [1600.0, 700.0, 400e3, 40e3, 0.05, -8.0],
        ],
        dtype=torch.float32,
    )
    t_grid_raw = torch.linspace(0.0, 0.12, 20)

    residual, x, v, a = compute_physics_residual(
        model, normalizer, params_raw, t_grid_raw
    )

    assert residual.shape == (2, 20)
    assert torch.isfinite(residual).all()
    assert torch.isfinite(x).all()
    assert x.shape == (2, 20)
    assert v.shape == (2, 20)
    assert a.shape == (2, 20)
