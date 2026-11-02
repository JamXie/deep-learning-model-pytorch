"""应用于弹簧-质量碰撞模型的 PINNs 损失函数。

涵盖强形式（PDE 残差）和能量形式（变分原理）损失构建，对应文献：
  AI for PDEs in solid mechanics: A review (Wang et al., 2025)
  Section 2.1.1（强形式，式 4）和 2.1.2（能量形式，式 5）
"""

from __future__ import annotations

import torch


# ==============================================================================
# 强形式损失 — 第 2.1.1 节，式 (4)
# ==============================================================================

def pinn_strong_form_loss(
    derivative_net,
    t_domain,
    t_ic,
    m,
    c,
    k_func_torch,
    F_func_torch,
    x0_true,
    v0_true,
    lambda_pde=1.0,
    lambda_ic=10.0,
):
    """弹簧-质量 ODE 的 PINN 强形式损失。

    损失 = lambda_pde * MSE_pde + lambda_ic * MSE_ic

    其中：
        MSE_pde = mean( (m*a + c*v + F_spring(x) - F(t))^2 )   在域点上
        MSE_ic  = mean( (x(t0)-x0)^2 + (v(t0)-v0)^2 )          在 t=0 处

    参数：
        derivative_net: 包装 FNN 的 DerivativeNetwork
        t_domain: 域内的配点，形状 (N_d, 1)
        t_ic: t=0 处的配点，形状 (N_ic, 1)
        m, c: 质量和阻尼（标量）
        k_func_torch: 弹簧力函数（可被 PyTorch 调用）
        F_func_torch: 外力函数（可被 PyTorch 调用）
        x0_true, v0_true: 真实初始条件
        lambda_pde, lambda_ic: 损失权重

    返回：
        total_loss, (loss_pde, loss_ic)
    """
    # 域点处的 PDE 残差
    residual, _, _, _ = derivative_net.compute_residual(
        t_domain, m, c, k_func_torch, F_func_torch
    )
    loss_pde = torch.mean(residual**2)

    # 初始条件损失
    x_ic, v_ic, _ = derivative_net.forward(t_ic)
    loss_ic = torch.mean((x_ic - x0_true) ** 2) + torch.mean((v_ic - v0_true) ** 2)

    total_loss = lambda_pde * loss_pde + lambda_ic * loss_ic
    return total_loss, (loss_pde, loss_ic)


# ==============================================================================
# 能量形式损失 — 第 2.1.2 节，式 (5)
# ==============================================================================

def spring_potential_energy(x, k_func_torch, x0_tensor):
    """计算弹簧的势能：∫ F_spring(s) ds 从 x0 到 x。

    对于线性弹簧 F=kx：PE = 0.5*k*(x - x0)^2
    对于非线性弹簧：通过中点积分近似。
    """
    # 梯形积分近似，使用 10 个求积点
    n_quad = 10
    if x.dim() == 0:
        x = x.unsqueeze(0)

    x_flat = x.reshape(-1)

    # 求积：对 F(s) 从 0 到每个 x_i 进行积分
    quad_points = torch.linspace(0, 1, n_quad, device=x.device).view(1, -1)
    s = x_flat.view(-1, 1) * quad_points  # (batch, n_quad)
    F_s = k_func_torch(s)
    pe = torch.trapezoid(F_s, s, dim=1)
    return pe.mean()


def pinn_energy_form_loss(model, t_domain, m, c, k_func_torch, F_func_torch, x0, v0):
    """弹簧-质量 ODE 的 PINN 能量形式（DEM）损失。

    使用 Hamilton 原理 / 最小作用量：
        作用量 = ∫ [动能 - 势能] dt - ∫ F_ext * x dt

    对于具有非保守阻尼的弹簧-质量-阻尼器，这里使用
    第 2.1.2 节中的能量残差方法：

        损失 = 能量残差^2

    注意：这是一种简化的能量形式。关于含耗散系统的完整能量方法，
          请参见 Fuhg & Bouklas (2022) 的 mDEM（综述论文第 4.3 节）。

    参数：
        model: FNN（如果需要，带 HardConstraintWrapper）
        t_domain: 配点 (N, 1)，已排序
        m, c: 质量和阻尼
        k_func_torch: 弹簧力函数
        F_func_torch: 外力函数
        x0, v0: 初始条件

    返回：
        total_loss, (loss_energy, loss_ic)
    """
    t_domain = t_domain.requires_grad_(True)
    x = model(t_domain)

    # 通过自动微分计算速度与加速度
    v = torch.autograd.grad(
        x, t_domain, grad_outputs=torch.ones_like(x), create_graph=True
    )[0]
    dvdt = torch.autograd.grad(
        v, t_domain, grad_outputs=torch.ones_like(v), create_graph=True
    )[0]

    # 能量残差：m*a + c*v + F_spring - F_ext = 0
    # 这等价于 ODE 的强形式；对于含阻尼的动力学问题，强形式更直接
    F_spring = k_func_torch(x)
    residual_energy = m * dvdt + c * v + F_spring - F_func_torch(t_domain)
    loss_energy = torch.mean(residual_energy**2)

    # 初始条件施加
    t0 = torch.zeros(1, 1, device=t_domain.device, requires_grad=True)
    x0_pred = model(t0)
    v0_pred = torch.autograd.grad(
        x0_pred, t0, grad_outputs=torch.ones_like(x0_pred), create_graph=True
    )[0]

    loss_ic = torch.mean((x0_pred - x0) ** 2) + torch.mean((v0_pred - v0) ** 2)

    total_loss = loss_energy + 10.0 * loss_ic
    return total_loss, (loss_energy, loss_ic)


# ==============================================================================
# 基于 NTK 的自适应权重 — 第 3.3 节，式 (27)
# ==============================================================================

def compute_ntk_weights(model, t_domain, t_ic, m, c, k_func_torch, F_func_torch, x0, v0):
    """使用神经正切核特征值计算自适应损失权重。

    遵循综述第 3.3 节（式 27，Wang S 等人 2022 年）：
        lambda_b = Tr(K_rb) / Tr(K_b)

    这里用梯度范数比值作廉价近似（论文式 25）。

    返回：
        lambda_pde, lambda_ic
    """
    t_domain_g = t_domain.detach().requires_grad_(True)
    t_ic_g = t_ic.detach().requires_grad_(True)

    # PDE 梯度范数
    pde_residual, _, _, _ = _compute_residual_static(
        model, t_domain_g, m, c, k_func_torch, F_func_torch
    )
    loss_pde_g = torch.mean(pde_residual**2)
    pde_grads = torch.autograd.grad(
        loss_pde_g, model.parameters(), create_graph=False, retain_graph=True
    )

    # 初始条件梯度范数
    x_ic, v_ic, _ = _forward_static(model, t_ic_g)
    loss_ic_g = torch.mean((x_ic - x0) ** 2) + torch.mean((v_ic - v0) ** 2)
    ic_grads = torch.autograd.grad(loss_ic_g, model.parameters(), create_graph=False)

    pde_norm = sum(g.norm().item() ** 2 for g in pde_grads if g is not None) ** 0.5
    ic_norm = sum(g.norm().item() ** 2 for g in ic_grads if g is not None) ** 0.5

    # 式 (25)：lambda_o = max|∇_theta L_r| / mean|∇_theta L_o|
    lambda_ic_new = pde_norm / ic_norm if (ic_norm > 0 and pde_norm > 0) else 10.0
    return 1.0, lambda_ic_new


def _compute_residual_static(model, t, m, c, k_func, F_func):
    """NTK 计算的静态辅助函数：返回残差与 x/v/a。"""
    t_g = t.requires_grad_(True)
    x = model(t_g)
    v = torch.autograd.grad(x, t_g, grad_outputs=torch.ones_like(x), create_graph=True)[0]
    a = torch.autograd.grad(v, t_g, grad_outputs=torch.ones_like(v), create_graph=True)[0]
    residual = m * a + c * v + k_func(x) - F_func(t_g)
    return residual, x, v, a


def _forward_static(model, t):
    """NTK 计算的静态辅助函数：返回 x/v/a。"""
    t_g = t.requires_grad_(True)
    x = model(t_g)
    v = torch.autograd.grad(x, t_g, grad_outputs=torch.ones_like(x), create_graph=True)[0]
    a = torch.autograd.grad(v, t_g, grad_outputs=torch.ones_like(v), create_graph=True)[0]
    return x, v, a
