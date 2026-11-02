"""用于碰撞安全算子学习的 DeepONet 架构。

实现深度算子网络（Deep Operator Network, DeepONet），参考：
  Lu et al. (2021a), Nature Machine Intelligence
  综述论文第 2.2.1 节，式 (9)-(11)

架构：
  G(u)(y) = sum_{k=1}^p NN_B_k(u) * NN_T_k(y) + bias

  - 分支网络（Branch Net）NN_B：接收碰撞参数 -> 输出系数 beta_k
  - 主干网络（Trunk Net）NN_T：接收时间坐标 -> 输出基函数 phi_k(y)
  - 输出：点积 beta · phi → 位移 x(t)

该网络学习算子：{碰撞参数} -> {时间响应}。
训练完成后，模型可以在毫秒级别预测新参数的碰撞响应
（而 FEM/PINNs 需要秒或分钟级）。
"""

from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn


# ==============================================================================
# 分支网络 — 编码输入参数
# ==============================================================================

class BranchNet(nn.Module):
    """将碰撞参数编码为特征系数。

    输入：[m, c, k1, k2, x_yield, v0] — 6 个标量参数
    输出：p 维特征向量（系数 beta）
    """

    def __init__(self, n_params=6, p=100, hidden_dims=None):
        super().__init__()
        if hidden_dims is None:
            hidden_dims = [128, 128, 128, 128]

        layers = []
        prev_dim = n_params
        for h_dim in hidden_dims:
            layers.append(nn.Linear(prev_dim, h_dim))
            layers.append(nn.Tanh())
            prev_dim = h_dim
        layers.append(nn.Linear(prev_dim, p))

        self.net = nn.Sequential(*layers)

    def forward(self, params):
        """将碰撞参数编码为系数。

        参数：
            params: (batch, n_params) — 归一化后的参数
        返回：
            beta: (batch, p) — 基函数系数
        """
        return self.net(params)


# ==============================================================================
# 主干网络 — 编码时间坐标
# ==============================================================================

class TrunkNet(nn.Module):
    """将时间坐标编码为基函数。

    输入：时间 t — 标量坐标
    输出：p 个基函数值 phi_1(t), ..., phi_p(t)

    这类似于传统函数逼近中选择基函数，但 DeepONet 从数据中学习最优基函数。
    """

    def __init__(self, n_time=1, p=100, hidden_dims=None):
        super().__init__()
        if hidden_dims is None:
            hidden_dims = [128, 128, 128, 128]

        layers = []
        prev_dim = n_time
        for h_dim in hidden_dims:
            layers.append(nn.Linear(prev_dim, h_dim))
            layers.append(nn.Tanh())
            prev_dim = h_dim
        layers.append(nn.Linear(prev_dim, p))

        self.net = nn.Sequential(*layers)

    def forward(self, t):
        """将时间坐标编码为基函数。

        参数：
            t: (batch, 1) — 时间点
        返回：
            phi: (batch, p) — 基函数值
        """
        return self.net(t)


# ==============================================================================
# DeepONet — 组合分支网络 + 主干网络
# ==============================================================================

class DeepONet(nn.Module):
    """用于碰撞响应预测的深度算子网络。

    G(params)(t) = sum_{k=1}^p branch_k(params) * trunk_k(t) + bias

    训练：
        输入：params (batch, 6) + t (batch, 1)
        目标：x(batch, 1) — 时间 t 处的位移

    推理：
        对于新参数，G(params)(t) 对 t_grid 中所有 t 计算 → 完整时间响应。
    """

    def __init__(self, n_params=6, p=100, branch_dims=None, trunk_dims=None):
        super().__init__()
        self.p = p
        self.branch = BranchNet(n_params, p, branch_dims)
        self.trunk = TrunkNet(1, p, trunk_dims)
        self.bias = nn.Parameter(torch.zeros(1))

    def forward(self, params, t):
        """DeepONet 的前向传播。

        参数：
            params: (batch, n_params) 碰撞参数
            t: (batch, 1) 时间坐标

        返回：
            x_pred: (batch, 1) 预测的位移
        """
        beta = self.branch(params)  # (batch, p)
        phi = self.trunk(t)         # (batch, p)

        # 点积：G(params)(t) = sum(beta_k * phi_k) + bias
        return torch.sum(beta * phi, dim=1, keepdim=True) + self.bias

    def predict_trajectory(self, params, t_grid_tensor):
        """预测给定参数的完整位移时间历程。

        参数：
            params: (batch, n_params) 或 (n_params,)
            t_grid_tensor: (n_time, 1) 时间网格

        返回：
            x_pred: (batch, n_time) 预测的位移历程
        """
        if params.dim() == 1:
            params = params.unsqueeze(0)

        batch_size = params.shape[0]
        n_time = t_grid_tensor.shape[0]

        # 每个参数样本与每个时间点配对
        params_expanded = params.unsqueeze(1).expand(-1, n_time, -1)
        t_expanded = t_grid_tensor.unsqueeze(0).expand(batch_size, -1, -1)

        params_flat = params_expanded.reshape(-1, params.shape[1])
        t_flat = t_expanded.reshape(-1, 1)

        x_flat = self.forward(params_flat, t_flat)
        return x_flat.reshape(batch_size, n_time)


# ==============================================================================
# 参数归一化器
# ==============================================================================

class ParameterNormalizer:
    """将碰撞参数归一化到 [-1, 1] 范围以确保训练稳定。"""

    def __init__(self, param_mins, param_maxs):
        self.param_mins = np.asarray(param_mins, dtype=np.float32)
        self.param_maxs = np.asarray(param_maxs, dtype=np.float32)

    def normalize(self, params):
        """NumPy 参数归一化到 [-1, 1]。"""
        return 2.0 * (params - self.param_mins) / (self.param_maxs - self.param_mins) - 1.0

    def normalize_tensor(self, params_tensor):
        """张量参数归一化到 [-1, 1]（保持设备一致）。"""
        mins = torch.as_tensor(self.param_mins, device=params_tensor.device)
        maxs = torch.as_tensor(self.param_maxs, device=params_tensor.device)
        return 2.0 * (params_tensor - mins) / (maxs - mins) - 1.0

    def denormalize(self, params_normalized):
        """将归一化参数还原到原始量纲。"""
        return self.param_mins + (params_normalized + 1.0) / 2.0 * (self.param_maxs - self.param_mins)
