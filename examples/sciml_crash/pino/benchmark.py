"""PINO 基准测试：比较 DeepONet 与 PINO 在碰撞响应预测中的表现。

评估内容：
  1. DeepONet 的零样本精度和速度
  2. PINO 物理微调带来的改进
  3. 在不同碰撞场景下的性能表现

运行（在仓库根目录）：
    python -m examples.sciml_crash.pino.benchmark
"""

from __future__ import annotations

import argparse
import time
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch

from common.cli import add_common_args
from common.device import get_device
from common.paths import resolve_output_dir
from common.plotting import save_figure, use_headless_backend
from common.seed import set_seed
from examples.sciml_crash.operator.data_generator import solve_crash_scenario
from examples.sciml_crash.operator.eval_operator import load_model
from examples.sciml_crash.pino.pino_pipeline import PINOFineTuner


def run_benchmark(checkpoint_path, output_dir, device="cpu", n_iterations=200):
    """依次在若干典型场景上对比 DeepONet 与 PINO。"""
    print(f"Device: {device}")
    print("=" * 70)
    print("PINO Benchmark: Crash Response Prediction")
    print("=" * 70)

    # 测试场景
    scenarios = [
        ("Sedan 50km/h", 1400, 600, 250e3, 30e3, 0.04, -13.9),
        ("SUV 60km/h", 2000, 800, 400e3, 45e3, 0.05, -16.7),
        ("Light car 70km/h", 900, 400, 200e3, 20e3, 0.03, -19.4),
        ("Mid-size 54km/h", 1500, 700, 300e3, 35e3, 0.045, -15.0),
    ]

    t_span = (0.0, 0.12)
    t_grid = np.linspace(0, 0.12, 200)
    T_max = float(t_grid[-1])
    t_t = torch.tensor((t_grid / T_max).reshape(-1, 1).astype("float32"), device=device)

    results = []

    for name, *pvals in scenarios:
        params_np = np.array(pvals, dtype=np.float32)
        m, c, k1, k2, xy, v0 = map(float, pvals)
        print(f"\n{'='*60}")
        print(f"Scenario: {name} | m={m:.0f}kg, v0={abs(v0)*3.6:.0f}km/h")

        # 参考解（RK45）
        _, x_ref, _, _ = solve_crash_scenario(m, c, k1, k2, xy, v0, t_span, 200)

        # --- A. DeepONet 零样本（干净副本） ---
        model_fresh, normalizer, _ = load_model(checkpoint_path, device)
        params_t = torch.tensor(normalizer.normalize(params_np.reshape(1, -1)), device=device)
        model_fresh.eval()
        t0 = time.time()
        with torch.no_grad():
            x_data = model_fresh.predict_trajectory(params_t, t_t).cpu().numpy().flatten()
        t_data = time.time() - t0

        # --- B. PINO 物理微调（另一个干净副本） ---
        model_pino, _, _ = load_model(checkpoint_path, device)
        tuner = PINOFineTuner(model_pino, normalizer, device)
        t0 = time.time()
        x_pino, _ = tuner.fine_tune(
            params_np,
            t_grid,
            n_iterations=n_iterations,
            lr=5e-5,
            lambda_data=5.0,
            freeze_branch=True,
            verbose=True,
        )
        t_pino = time.time() - t0

        # --- 指标 ---
        rmse_data = float(np.sqrt(np.mean((x_data - x_ref) ** 2)))
        rmse_pino = float(np.sqrt(np.mean((x_pino - x_ref) ** 2)))
        improvement = (rmse_data - rmse_pino) / (rmse_data + 1e-10) * 100

        print(f"  DeepONet: RMSE={rmse_data*1000:.2f}mm ({t_data*1000:.1f}ms)")
        print(f"  PINO:     RMSE={rmse_pino*1000:.2f}mm ({t_pino*1000:.0f}ms)")
        print(f"  Improvement: {improvement:.1f}%")

        results.append(
            {
                "name": name,
                "rmse_data": rmse_data,
                "rmse_pino": rmse_pino,
                "t_data": t_data,
                "t_pino": t_pino,
                "x_ref": x_ref,
                "x_data": x_data,
                "x_pino": x_pino,
            }
        )

    # 可视化
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    for ax, res in zip(axes.flatten(), results):
        ax.plot(t_grid * 1000, res["x_ref"] * 1000, "k-", lw=2, label="True (RK45)")
        ax.plot(t_grid * 1000, res["x_data"] * 1000, "--", color="#8ECae6", lw=1.8,
                alpha=0.95, label="DeepONet")
        ax.plot(t_grid * 1000, res["x_pino"] * 1000, "-", color="#FFB4A2", lw=1.8,
                alpha=0.95, label="PINO")
        ax.set_title(
            f"{res['name']} | Data RMSE={res['rmse_data']*1000:.1f}mm, "
            f"PINO RMSE={res['rmse_pino']*1000:.1f}mm",
            fontsize=11,
        )
        ax.set_xlabel("Time [ms]")
        ax.set_ylabel("Displacement [mm]")
        ax.legend(fontsize=9)
        ax.grid(alpha=0.3)

    fig.suptitle(
        "PINO: Physics-Informed Neural Operator for Crash Safety",
        fontsize=14,
        fontweight="bold",
    )
    fig.tight_layout()
    save_figure(fig, output_dir / "pino_benchmark_results.png")

    # 汇总
    print(f"\n{'='*70}")
    print("SUMMARY")
    print(f"{'='*70}")
    print(f"  {'Scenario':<20} {'DeepONet':>10} {'PINO':>10} {'Gain':>8}")
    print(f"  {'-'*52}")
    for res in results:
        gain = (res["rmse_data"] - res["rmse_pino"]) / res["rmse_data"] * 100
        print(
            f"  {res['name']:<20} {res['rmse_data']*1000:>7.2f}mm "
            f"{res['rmse_pino']*1000:>7.2f}mm {gain:>7.1f}%"
        )
    print(f"\nSaved: {output_dir / 'pino_benchmark_results.png'}")

    return results


def main(argv: list[str] | None = None) -> None:
    use_headless_backend()
    parser = argparse.ArgumentParser(description="PINO 物理微调基准")
    add_common_args(parser, epochs=False)
    parser.add_argument("--checkpoint", default=None, help="DeepONet 权重路径")
    parser.add_argument("--iters", type=int, default=200, help="每个场景的 PINO 微调迭代次数")
    args = parser.parse_args(argv)

    set_seed(args.seed)
    device = get_device(args.device)
    output_dir = resolve_output_dir(args.output_dir, "sciml_crash")
    checkpoint_path = Path(args.checkpoint) if args.checkpoint else output_dir / "deeponet_crash.pt"

    if not checkpoint_path.exists():
        raise FileNotFoundError(
            f"未找到权重 {checkpoint_path}\n"
            "请先运行：python -m examples.sciml_crash.operator.train_operator"
        )

    run_benchmark(checkpoint_path, output_dir, device=device, n_iterations=args.iters)


if __name__ == "__main__":
    main()
