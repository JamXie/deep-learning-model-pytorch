"""案例 B：双线性弹塑性弹簧 — 碰撞能量吸收。

物理模型：m*x'' + c*x' + F_spring(x) = F_impact(t)
其中 F_spring 为双线性弹塑性：
  - 弹性区：  F = k1 * x                                  |x| <= x_yield
  - 塑性平台：F = k1*x_yield + k2*(|x|-x_yield) * sign(x)  |x| >  x_yield

这模拟了典型的车辆前部结构碰撞行为：
  保险杠压溃 -> 碰撞盒坍塌 -> 纵梁屈曲

运行（在仓库根目录）：
    python -m examples.sciml_crash.pinns.case_bilinear --iters 8000
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
from examples.sciml_crash.pinns.utils import (
    bilinear_spring,
    compute_metrics,
    solve_spring_mass_rk45,
)


def main(argv: list[str] | None = None) -> None:
    use_headless_backend()
    parser = argparse.ArgumentParser(description="案例 B：双线性弹塑性弹簧 PINN 强形式")
    add_common_args(parser, epochs=False)
    parser.add_argument("--iters", type=int, default=8000, help="Adam 迭代次数")
    args = parser.parse_args(argv)

    set_seed(args.seed)
    device = get_device(args.device)
    output_dir = resolve_output_dir(args.output_dir, "sciml_crash", "pinns")

    print("=" * 60)
    print("Case B: Bilinear Elastoplastic Spring — PINNs Strong Form")
    print("=" * 60)

    # 带双线性弹簧的车辆碰撞参数
    m = 1200.0        # 质量 [kg]
    c = 800.0         # 阻尼 [N·s/m]（较高，结构阻尼）
    k1 = 300000.0     # 弹性刚度 [N/m]
    k2 = 30000.0      # 塑性刚度 [N/m]（弹性的 1/10）
    x_yield = 0.04    # 屈服位移 [m] = 塑性铰前压溃 40 mm

    x0 = 0.0
    v0 = -15.0        # 初始速度 [m/s] = 54 km/h
    t_span = (0.0, 0.12)

    def F_impact(t):
        """碰撞脉冲通过初始条件建模。"""
        return 0.0

    print(f"\nPhysics: m={m} kg, c={c} Ns/m")
    print(
        f"Spring: k1={k1/1000:.0f} kN/m, k2={k2/1000:.0f} kN/m, "
        f"x_yield={x_yield*1000:.1f} mm"
    )
    print(f"IC: v0={abs(v0)*3.6:.1f} km/h")

    # 参考解
    print("\nComputing reference solution (RK45)...")
    k_func = lambda x: bilinear_spring(x, k1=k1, k2=k2, x_yield=x_yield)
    t_ref, x_ref, v_ref, a_ref = solve_spring_mass_rk45(
        m, c, k_func, F_impact, x0, v0, t_span, n_points=500
    )
    print(f"  Max displacement: {np.max(np.abs(x_ref))*1000:.1f} mm")
    print(f"  Max acceleration: {np.max(np.abs(a_ref))/9.81:.1f} g")
    print(f"  Yield exceeded: {'YES' if np.max(np.abs(x_ref)) > x_yield else 'NO'}")

    # 训练 PINN（非线性问题需要更深的网络）
    print("\nTraining PINNs strong form (nonlinear spring)...")
    solver = PINNStrongFormSolver(
        m=m,
        c=c,
        k_func_np=k_func,
        F_func_np=F_impact,
        x0=x0,
        v0=v0,
        t_span=t_span,
        hidden_layers=[128, 128, 128, 128, 64],
        activation="tanh",
        use_hard_ic=True,
        use_adaptive_weights=True,
        device=device,
    )
    solver.train(
        n_iterations=args.iters,
        n_domain=300,
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

    print(f"  RMSE: {metrics['rmse']*1000:.3f} mm")
    print(f"  Peak error: {metrics['max_err']*1000:.3f} mm")
    print(
        f"  Relative RMSE: {metrics['nrmse']*100:.2f}% | "
        f"Accuracy (1-NRMSE): {metrics['accuracy']*100:.2f}% | "
        f"R^2: {metrics['r2']:.6f} | Peak RE: {metrics['peak_re']*100:.2f}%"
    )
    print(f"  RMSE (velocity): {rmse_v:.2e} m/s")
    print(f"  RMSE (acceleration): {rmse_a:.2e} m/s^2")

    idx_max = int(np.argmax(np.abs(x_ref)))
    print(f"  x_max ref: {x_ref[idx_max]*1000:.2f} mm")
    print(f"  x_max pred: {np.interp(t_ref[idx_max], t_pred, x_pred)*1000:.2f} mm")

    # 可视化
    fig, axes = plt.subplots(2, 3, figsize=(18, 10))

    ax = axes[0, 0]
    ax.plot(t_ref * 1000, x_ref * 1000, "k-", lw=2, label="RK45")
    ax.plot(t_pred * 1000, x_pred * 1000, "#ffb3b3", lw=2.5, linestyle="--", alpha=0.8, label="PINNs")
    ax.axhline(y=-x_yield * 1000, color="gray", linestyle=":", alpha=0.5, label="Yield")
    ax.set_xlabel("Time [ms]")
    ax.set_ylabel("Disp [mm]")
    ax.set_title("Displacement: Bilinear Spring")
    ax.legend()
    ax.grid(alpha=0.3)

    ax = axes[0, 1]
    ax.plot(t_ref * 1000, v_ref, "k-", lw=2, label="RK45")
    ax.plot(t_pred * 1000, v_pred, "#b3d9b3", lw=2.5, linestyle="--", alpha=0.8, label="PINNs")
    ax.set_xlabel("Time [ms]")
    ax.set_ylabel("Velocity [m/s]")
    ax.set_title("Velocity")
    ax.legend()
    ax.grid(alpha=0.3)

    ax = axes[0, 2]
    ax.plot(t_ref * 1000, a_ref / 9.81, "k-", lw=2, label="RK45")
    ax.plot(t_pred * 1000, a_pred / 9.81, "#b3d9f0", lw=2.5, linestyle="--", alpha=0.8, label="PINNs")
    ax.set_xlabel("Time [ms]")
    ax.set_ylabel("Acceleration [g]")
    ax.set_title("Acceleration (Crash Pulse)")
    ax.legend()
    ax.grid(alpha=0.3)

    ax = axes[1, 0]
    ax.plot(t_pred * 1000, (x_pred - metrics["x_ref_interp"]) * 1000, "r-", lw=1.5)
    ax.set_xlabel("Time [ms]")
    ax.set_ylabel("Error [mm]")
    ax.set_title(f"Position Error (RMSE={metrics['rmse']*1000:.3f} mm)")
    ax.grid(alpha=0.3)

    ax = axes[1, 1]
    ax.plot(x_ref * 1000, k_func(x_ref) / 1000, "k-", lw=2, label="True F-x")
    ax.plot(x_pred * 1000, k_func(x_pred) / 1000, "r--", lw=1.5, label="PINN path")
    ax.set_xlabel("Displacement [mm]")
    ax.set_ylabel("Spring Force [kN]")
    ax.set_title("Force-Displacement Curve")
    ax.legend()
    ax.grid(alpha=0.3)

    ax = axes[1, 2]
    ax.semilogy(solver.get_loss_array("total"), "b-", lw=1)
    ax.set_xlabel("Iteration")
    ax.set_ylabel("Loss")
    ax.set_title("Training Convergence")
    ax.grid(alpha=0.3)

    fig.tight_layout()
    save_figure(fig, output_dir / "case_bilinear_results.png")

    print(f"\n{'='*60}")
    print("Summary: Bilinear Elastoplastic Spring")
    print(f"{'='*60}")
    print(f"  Displacement RMSE: {metrics['rmse']*1000:.3f} mm ({metrics['nrmse']*100:.2f}%)")
    print(
        f"  Accuracy (1-NRMSE): {metrics['accuracy']*100:.2f}% | "
        f"R^2: {metrics['r2']:.6f} | Peak RE: {metrics['peak_re']*100:.2f}%"
    )
    print(f"  Saved figure to {output_dir / 'case_bilinear_results.png'}")


if __name__ == "__main__":
    main()
