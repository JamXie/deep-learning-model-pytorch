"""案例 C：率相关弹簧 — 应变率敏感材料。

物理模型：m*x'' + c*x' + k0*x*(1 + beta*|v|^n) = F_impact(t)

这是最与碰撞相关的弹簧 — 汽车用钢（DP、TRIP、硼钢）
在高应变率下（10–500 /s）表现出显著的强度增加。

使用 PINN 时，率相关弹簧通过弹簧力耦合了位置和速度，
使 ODE 具有更强的非线性。

运行（在仓库根目录）：
    python -m examples.sciml_crash.pinns.case_rate_dependent --iters 6000
"""

from __future__ import annotations

import argparse

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
from scipy.integrate import solve_ivp

from common.cli import add_common_args
from common.device import get_device
from common.paths import resolve_output_dir
from common.plotting import save_figure, use_headless_backend
from common.seed import set_seed
from examples.sciml_crash.pinns.pinn_strong_form import PINNStrongFormSolver
from examples.sciml_crash.pinns.utils import compute_metrics


class RateDependentDerivativeNet(nn.Module):
    """具有率相关弹簧力的导数网络。

    与通用 DerivativeNetwork 的区别：弹簧力同时依赖位移 x 和速度 v，
    因此残差需要在这里单独计算。
    """

    def __init__(self, model, m, c, k0, beta, n):
        super().__init__()
        self.model = model
        self.m = m
        self.c = c
        self.k0 = k0
        self.beta = beta
        self.n = n

    def forward(self, t):
        """计算 x(t)、v(t)、a(t)。"""
        t.requires_grad_(True)
        x = self.model(t)
        v = torch.autograd.grad(
            x, t, grad_outputs=torch.ones_like(x), create_graph=True, retain_graph=True
        )[0]
        a = torch.autograd.grad(
            v, t, grad_outputs=torch.ones_like(v), create_graph=True, retain_graph=True
        )[0]
        return x, v, a

    def compute_residual(self, t, F_func_torch):
        """计算率相关 ODE 残差：m*a + c*v + k0*x*(1+beta*|v|^n) - F(t)。"""
        x, v, a = self.forward(t)
        F_spring = self.k0 * x * (1.0 + self.beta * torch.abs(v) ** self.n)
        residual = self.m * a + self.c * v + F_spring - F_func_torch(t)
        return residual, x, v, a


def main(argv: list[str] | None = None) -> None:
    use_headless_backend()
    parser = argparse.ArgumentParser(description="案例 C：率相关弹簧 PINN 强形式")
    add_common_args(parser, epochs=False)
    parser.add_argument("--iters", type=int, default=6000, help="Adam 迭代次数")
    args = parser.parse_args(argv)

    set_seed(args.seed)
    device = get_device(args.device)
    output_dir = resolve_output_dir(args.output_dir, "sciml_crash", "pinns")

    print("=" * 60)
    print("Case C: Rate-Dependent Spring — PINNs Strong Form")
    print("=" * 60)

    m = 1200.0
    c = 500.0
    k0 = 200000.0     # 基础刚度 [N/m]
    beta = 0.05       # 率敏感系数
    n = 1.0           # 率指数（线性率相关）

    x0 = 0.0
    v0 = -15.0        # 54 km/h 碰撞
    t_span = (0.0, 0.12)

    def F_impact(t):
        return 0.0

    print(f"\nPhysics: m={m} kg, c={c} Ns/m")
    print(f"Spring: k0={k0/1000:.0f} kN/m, beta={beta}, n={n}")
    print(f"Rate effect at v=10 m/s: F_rate/F_static = {1+beta*10**n:.2f}x")
    print(f"IC: v0={abs(v0)*3.6:.1f} km/h")

    # 参考解（率相关效应直接写进 ODE 右端）
    print("\nComputing reference (RK45)...")

    def ode_rhs_rate(t, y):
        x, v = y
        F_spring = k0 * x * (1.0 + beta * abs(v) ** n)
        return [v, (F_impact(t) - c * v - F_spring) / m]

    t_eval = np.linspace(t_span[0], t_span[1], 500)
    sol = solve_ivp(ode_rhs_rate, t_span, [x0, v0], t_eval=t_eval, method="RK45", rtol=1e-8, atol=1e-10)
    t_ref = sol.t
    x_ref = sol.y[0]
    v_ref = sol.y[1]
    a_ref = np.array(
        [
            (F_impact(t) - c * v_ref[i] - k0 * x_ref[i] * (1.0 + beta * abs(v_ref[i]) ** n)) / m
            for i, t in enumerate(t_ref)
        ]
    )
    print(f"  Max displacement: {np.max(np.abs(x_ref))*1000:.1f} mm")
    print(f"  Max acceleration: {np.max(np.abs(a_ref))/9.81:.1f} g")

    # 训练 PINN：先构造基础求解器，再替换为率相关导数网络
    print("\nTraining PINNs (rate-dependent spring)...")
    solver = PINNStrongFormSolver(
        m=m,
        c=c,
        k_func_np=lambda x: k0 * x,  # 基础弹簧（率效应在下面的导数网络中处理）
        F_func_np=F_impact,
        x0=x0,
        v0=v0,
        t_span=t_span,
        hidden_layers=[128, 128, 128, 128],
        activation="tanh",
        use_hard_ic=True,
        use_adaptive_weights=True,
        device=device,
    )
    solver.derivative_net = RateDependentDerivativeNet(
        solver.model_wrapper, m, c, k0, beta, n
    )

    # 使用率相关损失的自定义训练循环
    optimizer = torch.optim.Adam(solver.base_model.parameters(), lr=1e-3)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.iters, eta_min=1e-5)

    x0_t = torch.tensor([x0], dtype=torch.float32, device=device)
    v0_t = torch.tensor([v0], dtype=torch.float32, device=device)
    verbose_every = max(1, args.iters // 12)

    for iteration in range(args.iters):
        t_domain_np = np.linspace(t_span[0], t_span[1], 300)
        t_ic_np = np.array([t_span[0]] * 5)
        t_domain = torch.tensor(
            t_domain_np.reshape(-1, 1).astype(np.float32),
            requires_grad=True,
            device=device,
        )
        t_ic = torch.tensor(
            t_ic_np.reshape(-1, 1).astype(np.float32),
            requires_grad=True,
            device=device,
        )

        optimizer.zero_grad()
        residual, _, _, _ = solver.derivative_net.compute_residual(t_domain, solver.F_func_torch)
        loss_pde = torch.mean(residual**2)

        x_ic, v_ic, _ = solver.derivative_net.forward(t_ic)
        loss_ic = torch.mean((x_ic - x0_t) ** 2) + torch.mean((v_ic - v0_t) ** 2)

        loss = loss_pde + 10.0 * loss_ic
        loss.backward()
        optimizer.step()
        scheduler.step()

        solver.loss_history.append(
            {"total": loss.item(), "pde": loss_pde.item(), "ic": loss_ic.item()}
        )
        if iteration % verbose_every == 0:
            print(
                f"Iter {iteration:5d} | Loss: {loss.item():.2e} | "
                f"PDE: {loss_pde.item():.2e} | IC: {loss_ic.item():.2e}"
            )

    # 评估
    print("\nEvaluating...")
    t_pred = np.linspace(t_span[0], t_span[1], 300)
    x_pred, v_pred, a_pred = solver.predict(t_pred)
    metrics = compute_metrics(t_ref, x_ref, t_pred, x_pred)

    v_ref_interp = np.interp(t_pred, t_ref, v_ref)
    a_ref_interp = np.interp(t_pred, t_ref, a_ref)
    rmse_v = float(np.sqrt(np.mean((v_pred - v_ref_interp) ** 2)))
    rmse_a = float(np.sqrt(np.mean((a_pred - a_ref_interp) ** 2)))

    print(f"  RMSE (displacement): {metrics['rmse']:.2e} m")
    print(f"  Max error: {metrics['max_err']:.2e} m")
    print(
        f"  Relative RMSE: {metrics['nrmse']*100:.2f}% | "
        f"Accuracy (1-NRMSE): {metrics['accuracy']*100:.2f}% | "
        f"R^2: {metrics['r2']:.6f} | Peak RE: {metrics['peak_re']*100:.2f}%"
    )
    print(f"  RMSE (velocity): {rmse_v:.2e} m/s")
    print(f"  RMSE (acceleration): {rmse_a:.2e} m/s^2")

    # 可视化
    fig, axes = plt.subplots(2, 3, figsize=(18, 10))

    ax = axes[0, 0]
    ax.plot(t_ref * 1000, x_ref * 1000, "k-", lw=2, label="RK45")
    ax.plot(t_pred * 1000, x_pred * 1000, "r--", lw=2, label="PINNs (rate-dep.)")
    ax.set_xlabel("Time [ms]")
    ax.set_ylabel("Displacement [mm]")
    ax.set_title("Rate-Dependent Spring Displacement")
    ax.legend()
    ax.grid(alpha=0.3)

    ax = axes[0, 1]
    ax.plot(t_ref * 1000, v_ref, "k-", lw=2, label="RK45")
    ax.plot(t_pred * 1000, v_pred, "y--", lw=2, label="PINNs")
    ax.set_xlabel("Time [ms]")
    ax.set_ylabel("Velocity [m/s]")
    ax.set_title("Velocity")
    ax.legend()
    ax.grid(alpha=0.3)

    ax = axes[0, 2]
    ax.plot(t_ref * 1000, a_ref / 9.81, "k-", lw=2, label="RK45")
    ax.plot(t_pred * 1000, a_pred / 9.81, "g--", lw=2, label="PINNs")
    ax.set_xlabel("Time [ms]")
    ax.set_ylabel("Acceleration [g]")
    ax.set_title("Crash Pulse (Rate-Dependent)")
    ax.legend()
    ax.grid(alpha=0.3)

    ax = axes[1, 0]
    ax.plot(t_pred * 1000, (x_pred - metrics["x_ref_interp"]) * 1000, "r-", lw=1.5)
    ax.set_xlabel("Time [ms]")
    ax.set_ylabel("Error [mm]")
    ax.set_title(f"Error (RMSE={metrics['rmse']*1000:.3f} mm)")
    ax.grid(alpha=0.3)

    ax = axes[1, 1]
    ax.semilogy(solver.get_loss_array("total"), "b-", lw=1)
    ax.set_xlabel("Iteration")
    ax.set_ylabel("Loss")
    ax.set_title("Training Convergence")
    ax.grid(alpha=0.3)
    axes[1, 2].set_visible(False)

    fig.tight_layout()
    save_figure(fig, output_dir / "case_rate_dependent_results.png")

    print(f"\n{'='*60}")
    print("Summary: Rate-Dependent Spring")
    print(f"{'='*60}")
    print(f"  Displacement RMSE: {metrics['rmse']*1000:.3f} mm ({metrics['nrmse']*100:.2f}%)")
    print(
        f"  Accuracy (1-NRMSE): {metrics['accuracy']*100:.2f}% | "
        f"R^2: {metrics['r2']:.6f} | Peak RE: {metrics['peak_re']*100:.2f}%"
    )
    print(f"  Saved figure to {output_dir / 'case_rate_dependent_results.png'}")


if __name__ == "__main__":
    main()
