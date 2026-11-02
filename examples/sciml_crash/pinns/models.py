"""碰撞安全应用中的神经网络架构。

定义了 PINNs 中用作近似函数的标准 FNN（全连接神经网络），对应文献：
  AI for PDEs in solid mechanics: A review (Wang et al., 2025)
  Section 2.1 — 神经网络近似函数替代传统形函数
"""

from __future__ import annotations

import torch
import torch.nn as nn


class FNN(nn.Module):
    """具有可配置激活函数的全连接神经网络。

    这是 PINNs 的标准网络架构：
        u(t; theta) = L_K ∘ sigma ∘ L_{K-1} ∘ ... ∘ sigma ∘ L_1(t)

    其中 L_i 是线性层，sigma 是激活函数。

    论文中的关键设计选择（第 3.1 节）：
    - 使用 Tanh 激活函数（而非 ReLU）—— ReLU 的二阶导数为零，
      会破坏微分算子的反向传播；
    - 自适应激活（第 3.1 节，式 20）：每层具有可训练的
      斜率参数 a 以调整激活形状。

    参数：
        input_dim: 输入维度（ODE 问题中通常为 1，表示时间）
        output_dim: 输出维度（1D 位移问题中为 1）
        hidden_layers: 隐藏层宽度列表，例如 [64, 64, 64]
        activation: 'tanh'（默认）、'sin'、'swish'
        use_adaptive_activation: 是否启用可训练激活斜率
    """

    def __init__(
        self,
        input_dim=1,
        output_dim=1,
        hidden_layers=None,
        activation="tanh",
        use_adaptive_activation=False,
    ):
        super().__init__()

        if hidden_layers is None:
            hidden_layers = [64, 64, 64, 64]

        self.use_adaptive_activation = use_adaptive_activation

        # 构建层列表
        layers = []
        prev_dim = input_dim
        for h_dim in hidden_layers:
            layers.append(nn.Linear(prev_dim, h_dim))
            prev_dim = h_dim

        self.hidden_layers = nn.ModuleList(layers)
        self.output_layer = nn.Linear(prev_dim, output_dim)

        # 自适应激活参数（论文第 3.1 节，式 20）
        if use_adaptive_activation:
            self.activation_slopes = nn.ParameterList(
                [nn.Parameter(torch.ones(1)) for _ in hidden_layers]
            )

        # 选择激活函数
        if activation == "tanh":
            self.activation = torch.tanh
        elif activation == "sin":
            self.activation = torch.sin
        elif activation == "swish":
            self.activation = lambda x: x * torch.sigmoid(x)
        else:
            raise ValueError(f"Unknown activation: {activation}")

        # 使用 Xavier 初始化方案初始化权重
        self._init_weights()

    def _init_weights(self):
        """Tanh 激活函数的 Xavier 初始化。"""
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_normal_(m.weight)
                nn.init.zeros_(m.bias)

    def forward(self, t):
        """前向传播：t -> u(t)。

        参数：
            t: 输入张量 (batch_size, input_dim)
        返回：
            u: 输出张量 (batch_size, output_dim)
        """
        x = t
        for i, layer in enumerate(self.hidden_layers):
            x = layer(x)
            if self.use_adaptive_activation:
                x = self.activation(self.activation_slopes[i] * x)
            else:
                x = self.activation(x)
        return self.output_layer(x)


class DerivativeNetwork(nn.Module):
    """通过自动微分计算 ODE 残差的包装器。

    给定一个映射 t -> x(t) 的 FNN，该模块计算：
        dx/dt、d^2x/dt^2 以及 ODE 残差。
    """

    def __init__(self, model):
        super().__init__()
        self.model = model

    def forward(self, t):
        """计算 x(t)、v(t)=dx/dt、a(t)=d^2x/dt^2。

        参数：
            t: 时间点，形状 (batch, 1)
        返回：
            x: 位移
            v: 速度（一阶导数）
            a: 加速度（二阶导数）
        """
        t.requires_grad_(True)
        x = self.model(t)

        # 一阶导数：dx/dt
        v = torch.autograd.grad(
            x, t, grad_outputs=torch.ones_like(x), create_graph=True, retain_graph=True
        )[0]

        # 二阶导数：d^2x/dt^2
        a = torch.autograd.grad(
            v, t, grad_outputs=torch.ones_like(v), create_graph=True, retain_graph=True
        )[0]

        return x, v, a

    def compute_residual(self, t, m, c, k_func_torch, F_func_torch):
        """计算 ODE 残差：r = m*a + c*v + F_spring(x) - F(t)。

        这是 PINN 损失函数的核心（第 2.1.1 节，式 2）。

        参数：
            t: 时间点
            m, c: 标量质量和阻尼（或张量）
            k_func_torch: 弹簧力函数（可微分）
            F_func_torch: 外力函数（可微分）

        返回：
            residual: ODE 残差 (batch, 1)
        """
        x, v, a = self.forward(t)
        residual = m * a + c * v + k_func_torch(x) - F_func_torch(t)
        return residual, x, v, a


class HardConstraintWrapper(nn.Module):
    """将初始条件作为硬约束施加。

    遵循综述论文第 3.4 节（式 37）：
        u(x) = h(x) + D(x) * u_g(x; theta)

    对于初始条件为 x(0)=x0、v(0)=v0 的弹簧-质量 ODE：
        x(t) = x0 + v0*t + t^2 * NN(t)

    因为在 t=0 时：
        x(0) = x0 + 0 + 0 = x0 [满足]
        x'(0) = v0 + 0 = v0 [满足]（因为 t^2*NN(t) 在 t=0 处的导数为 0）

    这消除了在初始条件上添加惩罚项的需要，降低了超参数敏感性。
    """

    def __init__(self, model, x0=0.0, v0=0.0):
        super().__init__()
        self.model = model
        self.x0 = x0
        self.v0 = v0

    def forward(self, t):
        """计算施加了硬约束初始条件的 x(t)。

        参数：
            t: 时间点 (batch, 1)
        返回：
            x: 满足 x(0)=x0、x'(0)=v0 的位移
        """
        nn_out = self.model(t)
        return self.x0 + self.v0 * t + t**2 * nn_out
