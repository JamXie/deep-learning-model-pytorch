"""用于弹簧-质量碰撞模型的 PINNs 强形式求解器。

实现了综述论文中描述的 PINN 强形式：
  Section 2.1.1 of the review paper (Wang et al., 2025)
  Reference: Raissi et al. (2019)

该求解器用神经网络替代近似函数（传统上为 FEM 形函数），
并将 PDE 残差最小化作为损失函数。
"""

from __future__ import annotations

import numpy as np
import torch

from .losses import compute_ntk_weights, pinn_strong_form_loss
from .models import DerivativeNetwork, FNN, HardConstraintWrapper
from .utils import sample_collocation_points


class PINNStrongFormSolver:
    """使用强形式（PDE 残差）损失的 PINN 求解器。

    求解：m * d^2x/dt^2 + c * dx/dt + F_spring(x) = F_impact(t)
    初始条件：x(0) = x0, dx/dt(0) = v0
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
        use_hard_ic=True,
        use_adaptive_weights=False,
        device="cpu",
    ):
        """
        参数：
            m, c: 质量和阻尼（标量）
            k_func_np: 弹簧力 F_spring(x)，可被 NumPy 调用
            F_func_np: 外力 F_impact(t)，可被 NumPy 调用
            x0, v0: 初始条件
            t_span: (t_start, t_end)
            hidden_layers: 隐藏层宽度列表
            activation: 'tanh'、'sin'、'swish'
            use_hard_ic: 若为 True，通过 t^2*NN(t) 构造施加硬约束初始条件
            use_adaptive_weights: 若为 True，使用基于 NTK 的权重平衡
            device: 计算设备
        """
        self.m = m
        self.c = c
        self.x0 = x0
        self.v0 = v0
        self.t_span = t_span
        self.use_hard_ic = use_hard_ic
        self.use_adaptive_weights = use_adaptive_weights
        self.device = torch.device(device)

        # 默认网络：4 个隐藏层 × 64 个神经元
        if hidden_layers is None:
            hidden_layers = [64, 64, 64, 64]

        # 构建网络
        self.base_model = FNN(
            input_dim=1, output_dim=1, hidden_layers=hidden_layers, activation=activation
        ).to(self.device)

        if use_hard_ic:
            self.model_wrapper = HardConstraintWrapper(self.base_model, x0, v0)
        else:
            self.model_wrapper = self.base_model

        self.derivative_net = DerivativeNetwork(self.model_wrapper)

        self.k_func_np = k_func_np
        self.F_func_np = F_func_np

        # 将 NumPy 弹簧力/外力包装成张量函数（保持原实现的做法）
        def k_func_torch(x):
            x_np = np.asarray(x.detach().cpu().numpy(), dtype=np.float64).ravel()
            result = np.asarray(k_func_np(x_np), dtype=np.float32)
            if result.ndim == 0:
                result = np.full_like(x_np, result, dtype=np.float32)
            return torch.as_tensor(result, device=x.device).reshape(x.shape)

        def F_func_torch(t):
            t_np = np.asarray(t.detach().cpu().numpy(), dtype=np.float64).ravel()
            result = np.asarray(F_func_np(t_np), dtype=np.float32)
            if result.ndim == 0:
                result = np.full_like(t_np, result, dtype=np.float32)
            return torch.as_tensor(result, device=t.device).reshape(t.shape)

        self.k_func_torch = k_func_torch
        self.F_func_torch = F_func_torch

        # 追踪损失历史
        self.loss_history = []

    def train(
        self,
        n_iterations=5000,
        n_domain=200,
        n_ic=2,
        lr=1e-3,
        lambda_pde=1.0,
        lambda_ic=10.0,
        sampler="uniform",
        verbose=True,
        verbose_every=500,
    ):
        """训练 PINN 模型。

        参数：
            n_iterations: Adam 迭代次数
            n_domain: 域内配点数量
            n_ic: 初始条件点数量
            lr: 学习率
            lambda_pde, lambda_ic: 初始损失权重
            sampler: 'uniform' / 'random' / 'chebyshev' / 'latin_hypercube'
            verbose: 是否打印进度
            verbose_every: 打印频率

        返回：
            self（已训练的模型）
        """
        optimizer = torch.optim.Adam(self.base_model.parameters(), lr=lr)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=n_iterations, eta_min=lr * 0.01
        )

        x0_tensor = torch.tensor([self.x0], dtype=torch.float32, device=self.device)
        v0_tensor = torch.tensor([self.v0], dtype=torch.float32, device=self.device)

        # 定期重新采样点以获得更好的覆盖（第 3.2 节）
        resample_every = max(1, n_iterations // 3)

        for iteration in range(n_iterations):
            # 重新采样配点
            if iteration % resample_every == 0:
                t_domain_np, t_ic_np = sample_collocation_points(
                    self.t_span, n_domain, n_ic, scheme=sampler
                )
                t_domain = torch.tensor(
                    t_domain_np, requires_grad=True, device=self.device
                )
                t_ic = torch.tensor(t_ic_np, requires_grad=True, device=self.device)

            # 自适应权重平衡（第 3.3 节）
            if self.use_adaptive_weights and iteration > 0 and iteration % 200 == 0:
                lambda_pde, lambda_ic = compute_ntk_weights(
                    self.model_wrapper,
                    t_domain,
                    t_ic,
                    self.m,
                    self.c,
                    self.k_func_torch,
                    self.F_func_torch,
                    x0_tensor,
                    v0_tensor,
                )

            optimizer.zero_grad()
            loss, (loss_pde, loss_ic) = pinn_strong_form_loss(
                self.derivative_net,
                t_domain,
                t_ic,
                self.m,
                self.c,
                self.k_func_torch,
                self.F_func_torch,
                x0_tensor,
                v0_tensor,
                lambda_pde,
                lambda_ic,
            )

            loss.backward()
            optimizer.step()
            scheduler.step()

            self.loss_history.append(
                {"total": loss.item(), "pde": loss_pde.item(), "ic": loss_ic.item()}
            )

            if verbose and iteration % verbose_every == 0:
                print(
                    f"Iter {iteration:5d} | Loss: {loss.item():.2e} | "
                    f"PDE: {loss_pde.item():.2e} | IC: {loss_ic.item():.2e}"
                )

        return self

    def predict(self, t_eval, numpy=True):
        """预测给定时间点处的位移、速度和加速度。"""
        if torch.is_tensor(t_eval):
            t_tensor = t_eval.reshape(-1, 1).float().to(self.device)
        else:
            t_tensor = torch.tensor(
                np.asarray(t_eval, dtype=np.float32).reshape(-1, 1), device=self.device
            )

        self.model_wrapper.eval()
        t_grad = t_tensor.detach().requires_grad_(True)
        x_tensor = self.model_wrapper(t_grad)

        v_tensor = torch.autograd.grad(
            x_tensor,
            t_grad,
            grad_outputs=torch.ones_like(x_tensor),
            create_graph=True,
            retain_graph=True,
        )[0]
        a_tensor = torch.autograd.grad(
            v_tensor, t_grad, grad_outputs=torch.ones_like(v_tensor), create_graph=False
        )[0]
        self.model_wrapper.train()

        if numpy:
            return (
                x_tensor.detach().cpu().numpy().flatten(),
                v_tensor.detach().cpu().numpy().flatten(),
                a_tensor.detach().cpu().numpy().flatten(),
            )
        return x_tensor, v_tensor, a_tensor

    def get_loss_array(self, key="total"):
        """将损失历史提取为 NumPy 数组。"""
        return np.array([d[key] for d in self.loss_history])
