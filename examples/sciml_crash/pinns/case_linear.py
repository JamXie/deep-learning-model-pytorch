"""案例 A：线性弹簧-质量-阻尼器 — PINN 强形式验证基准。

物理模型：m*x'' + c*x' + k*x = 0
初始条件：x(0) = 0, v(0) = v0

这是在转向非线性碰撞弹簧之前最简单的验证案例。
无阻尼自由振动存在解析解，RK45 为一般情况提供参考解。

运行（在仓库根目录）：
    python -m examples.sciml_crash.pinns.case_linear --iters 5000
"""

from __future__ import annotations

import argparse

import matplotlib.pyplot as plt
import numpy as np

from common.cli import add_common_args
from common.device import get_device
from common.paths import resolve_output_dir
from common.plotting import save_figure, use_headless_backend
from common.seed import set_seed
from examples.sciml_crash.pinns.pinn_strong_form import PINNStrongFormSolver
from examples.sciml_crash.pinns.utils import compute_metrics, linear_spring, solve_spring_mass_rk45


def main(argv: list[str] | None = None) -> None:
    use_headless_backend()
    parser = argparse.ArgumentParser(description="案例 A：线性弹簧 PINN 强形式")
    add_common_args(parser, epochs=False)
    parser.add_argument("--iters", type=int, default=5000, help="Adam 迭代次数")
    args = parser.parse_args(argv)

    set_seed(args.seed)
    device = get_device(args.device)
    output_dir = resolve_output_dir(args.output_dir, "sciml_crash", "pinns")

    print("=" * 60)
    print("Case A: Linear Spring-Mass-Damper — PINNs Strong Form")
    print("=" * 60)

    # 车辆碰撞简化参数
    m = 1200.0       # 质量 [kg]
    c = 500.0        # 阻尼 [N·s/m]
    k = 200000.0     # 弹簧刚度 [N/m]

    x0 = 0.0         # 初始位移 [m]
    v0 = -15.0       # 初始速度 [m/s]（约 54 km/h 碰撞）
    t_span = (0.0, 0.12)

    def F_impact(t):
        """碰撞过程中无外力（带初始条件的自由振动）。"""
        return 0.0

    print(f"\nPhysics: m={m} kg, c={c} Ns/m, k={k/1000:.1f} kN/m")
    print(f"IC: x(0)={x0}, v(0)={v0} m/s = {abs(v0)*3.6:.1f} km/h")
    print(f"Time: {t_span[0]}–{t_span[1]} s")

    # 参考解（RK45）
    print("\nComputing reference solution (RK45)...")
    k_func = lambda x: linear_spring(x, k=k)
    t_ref, x_ref, v_ref, a_ref = solve_spring_mass_rk45(
        m, c, k_func, F_impact, x0, v0, t_span, n_points=500
    )
    print(f"  Max displacement: {np.max(np.abs(x_ref)):.4f} m")
    print(f"  Max acceleration: {np.max(np.abs(a_ref))/9.81:.1f} g")

    # 训练 PINN
    print("\nTraining PINNs strong form...")
    solver = PINNStrongFormSolver(
        m=m,
        c=c,
        k_func_np=k_func,
        F_func_np=F_impact,
        x0=x0,
        v0=v0,
        t_span=t_span,
        hidden_layers=[64, 64, 64, 64],
        activation="tanh",
        use_hard_ic=True,
        use_adaptive_weights=True,
        device=device,
    )
    solver.train(
        n_iterations=args.iters,
        n_domain=200,
        n_ic=5,
        lr=1e-3,
        lambda_pde=1.0,
        lambda_ic=10.0,
        sampler="uniform",
        verbose=True,
        verbose_every=max(1, args.iters // 10),
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
    ax.plot(t_ref * 1000, x_ref * 1000, "k-", linewidth=2, label="RK45 Reference")
    ax.plot(t_pred * 1000, x_pred * 1000, "g--", linewidth=2, label="PINNs Prediction")
    ax.set_xlabel("Time [ms]")
    ax.set_ylabel("Displacement [mm]")
    ax.set_title("Displacement Comparison: Linear Spring")
    ax.legend()
    ax.grid(alpha=0.3)

    ax = axes[0, 1]
    ax.plot(t_ref * 1000, v_ref, "k-", linewidth=2, label="RK45")
    ax.plot(t_pred * 1000, v_pred, "y--", linewidth=2, label="PINNs")
    ax.set_xlabel("Time [ms]")
    ax.set_ylabel("Velocity [m/s]")
    ax.set_title("Velocity Comparison")
    ax.legend()
    ax.grid(alpha=0.3)

    ax = axes[0, 2]
    ax.plot(t_ref * 1000, a_ref, "k-", linewidth=2, label="RK45")
    ax.plot(t_pred * 1000, a_pred, "c--", linewidth=2, label="PINNs")
    ax.set_xlabel("Time [ms]")
    ax.set_ylabel("Acceleration [m/s^2]")
    ax.set_title("Acceleration Comparison")
    ax.legend()
    ax.grid(alpha=0.3)

    ax = axes[1, 0]
    ax.plot(t_pred * 1000, (x_pred - metrics["x_ref_interp"]) * 1000, "r-", linewidth=1.5)
    ax.set_xlabel("Time [ms]")
    ax.set_ylabel("Position Error [mm]")
    ax.set_title(f"PINN Error (RMSE={metrics['rmse']*1000:.3f} mm)")
    ax.grid(alpha=0.3)

    ax = axes[1, 1]
    ax.semilogy(solver.get_loss_array("total"), "b-", linewidth=1)
    ax.set_xlabel("Iteration")
    ax.set_ylabel("Total Loss")
    ax.set_title("Training Loss Convergence")
    ax.grid(alpha=0.3)
    axes[1, 2].set_visible(False)

    fig.tight_layout()
    save_figure(fig, output_dir / "case_linear_results.png")

    print(f"\n{'='*60}")
    print("Summary")
    print(f"{'='*60}")
    print(f"  Displacement RMSE: {metrics['rmse']*1000:.3f} mm")
    print(
        f"  Accuracy (1-NRMSE): {metrics['accuracy']*100:.2f}% | "
        f"R^2: {metrics['r2']:.6f} | Peak RE: {metrics['peak_re']*100:.2f}%"
    )
    print(f"  Saved figure to {output_dir / 'case_linear_results.png'}")


if __name__ == "__main__":
    main()
