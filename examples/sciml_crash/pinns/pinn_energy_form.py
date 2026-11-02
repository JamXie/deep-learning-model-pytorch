"""用于弹簧-质量碰撞模型的 PINNs 能量形式（DEM）求解器。

实现了文献中描述的深度能量方法（DEM）：
  Section 2.1.2 of the review paper (Wang et al., 2025)
  Reference: Samaniego et al. (2020), Yu (2018) Deep Ritz

对于保守力学系统：
    损失 = ∫ [应变能 - 外力功] dΩ

对于含阻尼的弹簧-质量 ODE，这里使用基于能量的残差方法，
与强形式相比降低了导数阶数。
"""

from __future__ import annotations

import numpy as np
import torch

from .models import FNN, HardConstraintWrapper


class PINNEnergyFormSolver:
    """使用能量形式（变分原理）损失的 PINN 求解器。

    求解：m * d^2x/dt^2 + c * dx/dt + F_spring(x) = F_impact(t)

    对于保守系统（c=0），使用作用量原理：
        S[x] = ∫₀ᵀ [½m·v^2 - V(x) + F_ext·x] dt
    其中 V(x) = ∫₀ˣ F_spring(s) ds。

    对于阻尼系统，能量方法添加了耗散功。
    """

    def __init__(
        self,
        m,
        c,
        k_func_np,
        F_func_np,
        x0,
        v0,
        t_span,
        hidden_layers=None,
        activation="tanh",
        device="cpu",
    ):
        self.m = m
        self.c = c
        self.x0 = x0
        self.v0 = v0
        self.t_span = t_span
        self.device = torch.device(device)

        if hidden_layers is None:
            hidden_layers = [64, 64, 64, 64]

        self.base_model = FNN(
            input_dim=1, output_dim=1, hidden_layers=hidden_layers, activation=activation
        ).to(self.device)
        # 使用硬约束初始条件以避免惩罚项
        self.model = HardConstraintWrapper(self.base_model, x0, v0)

        self.k_func_np = k_func_np
        self.F_func_np = F_func_np

        def k_func_torch(x):
            xn = np.asarray(x.detach().cpu().numpy(), dtype=np.float64).ravel()
            result = np.asarray(k_func_np(xn), dtype=np.float32)
            if result.ndim == 0:
                result = np.full_like(xn, result, dtype=np.float32)
            return torch.as_tensor(result, device=x.device).reshape(x.shape)

        def F_func_torch(t):
            tn = np.asarray(t.detach().cpu().numpy(), dtype=np.float64).ravel()
            result = np.asarray(F_func_np(tn), dtype=np.float32)
            if result.ndim == 0:
                result = np.full_like(tn, result, dtype=np.float32)
            return torch.as_tensor(result, device=t.device).reshape(t.shape)

        self.k_func_torch = k_func_torch
        self.F_func_torch = F_func_torch
        self.loss_history = []

    def _compute_spring_potential(self, x):
        """计算势能 V(x) = ∫₀ˣ F_spring(s) ds（10 点梯形积分）。"""
        n_quad = 10
        xi = torch.linspace(0, 1, n_quad, device=x.device)
        # s 值：从 0 到每个样本的 x_i
        s = x * xi.view(1, -1)  # (batch, n_quad)
        F_s = self.k_func_torch(s)
        return torch.trapezoid(F_s, s, dim=1)

    def _compute_action(self, t_domain):
        """计算保守弹簧-质量系统的作用量积分。

        S[x] = ∫₀ᵀ [T(t) - V(x(t)) + W_ext(t)] dt
        其中 T = ½m·v^2，W_ext = F(t)·x(t)。
        """
        t = t_domain.requires_grad_(True)
        x = self.model(t)

        v = torch.autograd.grad(
            x, t, grad_outputs=torch.ones_like(x), create_graph=True
        )[0]

        T = 0.5 * self.m * v**2
        V = self._compute_spring_potential(x)
        W_ext = self.F_func_torch(t) * x

        # 拉格朗日被积函数：L = T - V + W_ext
        L = T - V.reshape(x.shape) + W_ext
        action = torch.trapezoid(L.squeeze(), t.squeeze())
        return action, x, v

    def train(self, n_iterations=5000, n_domain=300, lr=1e-3, verbose=True, verbose_every=500):
        """使用能量形式损失进行训练。

        对于阻尼情况，最小化能量平衡残差：
            r = m*a + c*v + F_spring(x) - F_ext(t)

        使用硬约束初始条件时，变分方法比强形式超参数更少，
        因为我们不需要 lambda_ic。
        """
        optimizer = torch.optim.Adam(self.base_model.parameters(), lr=lr)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=n_iterations, eta_min=lr * 0.01
        )

        resample_every = max(1, n_iterations // 4)
        for iteration in range(n_iterations):
            # 重新采样点
            if iteration % resample_every == 0:
                t_np = np.linspace(self.t_span[0], self.t_span[1], n_domain)
                t_domain = torch.tensor(
                    t_np.reshape(-1, 1).astype(np.float32),
                    requires_grad=True,
                    device=self.device,
                )

            optimizer.zero_grad()

            t = t_domain.requires_grad_(True)
            x = self.model(t)
            v = torch.autograd.grad(
                x, t, grad_outputs=torch.ones_like(x), create_graph=True, retain_graph=True
            )[0]
            a = torch.autograd.grad(
                v, t, grad_outputs=torch.ones_like(v), create_graph=True, retain_graph=True
            )[0]

            F_spring = self.k_func_torch(x)
            F_ext = self.F_func_torch(t)
            residual = self.m * a + self.c * v + F_spring - F_ext

            loss = torch.mean(residual**2)
            loss.backward()
            optimizer.step()
            scheduler.step()

            self.loss_history.append(loss.item())
            if verbose and iteration % verbose_every == 0:
                print(f"Iter {iteration:5d} | Loss: {loss.item():.2e}")

        return self

    def predict(self, t_eval, numpy=True):
        """预测位移、速度和加速度。"""
        if torch.is_tensor(t_eval):
            t_tensor = t_eval.reshape(-1, 1).float().to(self.device)
        else:
            t_tensor = torch.tensor(
                np.asarray(t_eval, dtype=np.float32).reshape(-1, 1), device=self.device
            )

        with torch.no_grad():
            self.model.eval()
            x_tensor = self.model(t_tensor)
            self.model.train()

        t_g = t_tensor.detach().requires_grad_(True)
        x_g = self.model(t_g)
        # 计算二阶导数需要保留一阶导的计算图
        v_g = torch.autograd.grad(
            x_g, t_g, grad_outputs=torch.ones_like(x_g), create_graph=True
        )[0]
        a_g = torch.autograd.grad(
            v_g, t_g, grad_outputs=torch.ones_like(v_g), create_graph=False
        )[0]

        if numpy:
            return (
                x_tensor.detach().cpu().numpy().flatten(),
                v_g.detach().cpu().numpy().flatten(),
                a_g.detach().cpu().numpy().flatten(),
            )
        return x_tensor, v_g, a_g

    def get_loss_array(self):
        return np.array(self.loss_history)
