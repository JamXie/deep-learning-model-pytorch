"""在碰撞测试场景上评估训练好的 DeepONet。

展示：
1. 速度对比：DeepONet vs RK45 求解器
2. 测试集上的精度
3. 泛化能力：预测未见过的碰撞参数的响应
4. 位移的预测质量

运行（在仓库根目录）：
    python -m examples.sciml_crash.operator.eval_operator
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
from examples.sciml_crash.operator.data_generator import (
    generate_crash_dataset,
    solve_crash_scenario,
    split_dataset,
)
from examples.sciml_crash.operator.deeponet_model import DeepONet, ParameterNormalizer


def load_model(model_path, device="cpu"):
    """加载训练好的 DeepONet 与参数归一化器。"""
    checkpoint = torch.load(model_path, map_location=device)
    p = checkpoint["p"]
    model = DeepONet(n_params=6, p=p).to(device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    ns = checkpoint["normalizer_state"]
    normalizer = ParameterNormalizer(ns["mins"], ns["maxs"])
    return model, normalizer, checkpoint.get("history", {})


def benchmark_speed(model, normalizer, n_scenarios=100, rk45_samples=20, device="cpu", seed=123):
    """对比 DeepONet 与传统 RK45 求解器的推理速度。"""
    print("\n--- Speed Benchmark ---")
    rng = np.random.RandomState(seed)
    test_params = np.column_stack(
        [
            rng.uniform(800, 2000, n_scenarios),      # m
            rng.uniform(300, 1000, n_scenarios),      # c
            rng.uniform(150e3, 500e3, n_scenarios),   # k1
            rng.uniform(10e3, 50e3, n_scenarios),     # k2
            rng.uniform(0.02, 0.06, n_scenarios),     # xy
            rng.uniform(-20, -5, n_scenarios),        # v0
        ]
    )

    t_span = (0.0, 0.12)
    n_t = 200

    # 计时：RK45 求解器（限制样本数以免太慢）
    rk45_times = []
    for i in range(min(rk45_samples, n_scenarios)):
        m, c, k1, k2, xy, v0 = test_params[i]
        t0 = time.time()
        solve_crash_scenario(m, c, k1, k2, xy, v0, t_span, n_t)
        rk45_times.append(time.time() - t0)
    avg_rk45 = float(np.mean(rk45_times))

    # 计时：DeepONet（含预热）
    test_params_norm = normalizer.normalize(test_params.astype(np.float32))
    test_params_t = torch.tensor(test_params_norm, device=device)
    t_grid = np.linspace(t_span[0], t_span[1], n_t)
    t_norm = t_grid / t_grid[-1]
    t_t = torch.tensor(t_norm.reshape(-1, 1).astype(np.float32), device=device)

    with torch.no_grad():
        _ = model.predict_trajectory(test_params_t[:10], t_t)

    t0 = time.time()
    with torch.no_grad():
        _ = model.predict_trajectory(test_params_t, t_t)
    deeponet_time = (time.time() - t0) / n_scenarios

    print(f"  RK45 average: {avg_rk45*1000:.1f} ms/scenario")
    print(f"  DeepONet:     {deeponet_time*1000:.2f} ms/scenario")
    print(f"  Speedup:      {avg_rk45/deeponet_time:.0f}x")
    return avg_rk45 / deeponet_time


def evaluate_accuracy(model, normalizer, test_data, device="cpu"):
    """评估测试集上的预测精度。"""
    print("\n--- Accuracy Evaluation ---")

    params = test_data["params"].astype(np.float32)
    t_grid = test_data["t_grid"]
    x_true = test_data["x_data"]

    params_norm = normalizer.normalize(params)
    t_norm = t_grid / t_grid[-1]

    with torch.no_grad():
        x_pred = model.predict_trajectory(
            torch.tensor(params_norm, device=device),
            torch.tensor(t_norm.reshape(-1, 1).astype(np.float32), device=device),
        ).cpu().numpy()

    rmse_list = []
    peak_re_list = []
    for i in range(len(params)):
        rmse_list.append(np.sqrt(np.mean((x_pred[i] - x_true[i]) ** 2)))
        peak_true = np.max(np.abs(x_true[i]))
        peak_pred = np.max(np.abs(x_pred[i]))
        peak_re_list.append(abs(peak_pred - peak_true) / (peak_true + 1e-8))

    rmse_arr = np.array(rmse_list)
    peak_arr = np.array(peak_re_list)

    print(f"  Displacement RMSE: {np.mean(rmse_arr)*1000:.3f} +/- {np.std(rmse_arr)*1000:.3f} mm")
    print(f"  Peak displacement RE: {np.mean(peak_arr)*100:.1f}% +/- {np.std(peak_arr)*100:.1f}%")
    print(f"  Max RMSE: {np.max(rmse_arr)*1000:.3f} mm | Min RMSE: {np.min(rmse_arr)*1000:.3f} mm")

    return rmse_arr, peak_arr, x_pred


def plot_predictions(model, normalizer, test_params, test_x, test_t, save_path):
    """绘制样本场景的预测轨迹与真实轨迹对比图。"""
    device = next(model.parameters()).device

    params_norm = normalizer.normalize(test_params.astype(np.float32))
    t_norm = test_t / test_t[-1]

    with torch.no_grad():
        pred = model.predict_trajectory(
            torch.tensor(params_norm, device=device),
            torch.tensor(t_norm.reshape(-1, 1).astype(np.float32), device=device),
        ).cpu().numpy()

    n_plot = min(6, len(test_params))
    fig, axes = plt.subplots(2, 3, figsize=(16, 10))
    axes = axes.flatten()

    for i in range(n_plot):
        ax = axes[i]
        ax.plot(test_t * 1000, test_x[i] * 1000, "k-", lw=2, label="True (RK45)")
        ax.plot(test_t * 1000, pred[i] * 1000, "r--", lw=2, label="DeepONet")

        p = test_params[i]
        rmse = np.sqrt(np.mean((pred[i] - test_x[i]) ** 2))
        ax.set_title(
            f"m={p[0]:.0f}kg, v0={abs(p[5])*3.6:.0f}km/h, "
            f"k1={p[2]/1000:.0f}kN/m | RMSE={rmse*1000:.2f}mm"
        )
        ax.set_xlabel("Time [ms]")
        ax.set_ylabel("Displacement [mm]")
        ax.legend(fontsize=9)
        ax.grid(alpha=0.3)

    # 隐藏未使用的子图
    for i in range(n_plot, len(axes)):
        axes[i].set_visible(False)

    fig.suptitle("DeepONet: Crash Response Prediction", fontsize=15, fontweight="bold")
    fig.tight_layout()
    save_figure(fig, save_path)
    print(f"Predictions plot saved to {save_path}")
    return pred


def main(argv: list[str] | None = None) -> None:
    use_headless_backend()
    parser = argparse.ArgumentParser(description="评估碰撞响应 DeepONet")
    add_common_args(parser, epochs=False)
    parser.add_argument("--checkpoint", default=None, help="权重路径")
    parser.add_argument("--data-path", default=None, help="测试数据 pkl（可选）")
    parser.add_argument("--num-samples", type=int, default=200, help="无数据文件时生成的测试样本数")
    parser.add_argument("--n-scenarios", type=int, default=100, help="速度基准的场景数")
    parser.add_argument("--rk45-samples", type=int, default=20, help="RK45 计时样本数")
    args = parser.parse_args(argv)

    set_seed(args.seed)
    device = get_device(args.device)
    output_dir = resolve_output_dir(args.output_dir, "sciml_crash")
    checkpoint_path = Path(args.checkpoint) if args.checkpoint else output_dir / "deeponet_crash.pt"

    print(f"Using device: {device}")
    print("=" * 60)
    print("Evaluating DeepONet for Crash Response")
    print("=" * 60)

    if not checkpoint_path.exists():
        raise FileNotFoundError(
            f"未找到权重 {checkpoint_path}\n"
            "请先运行：python -m examples.sciml_crash.operator.train_operator"
        )
    model, normalizer, _ = load_model(checkpoint_path, device)

    # 测试集：优先使用已有 pkl 的 test 划分，否则现场生成
    if args.data_path:
        from examples.sciml_crash.operator.data_generator import load_dataset

        payload = load_dataset(args.data_path)
        test = payload.get("test")
        if test is None:
            raise KeyError(f"{args.data_path} 中没有 'test' 划分。")
    else:
        data = generate_crash_dataset(
            n_samples=args.num_samples, n_time_points=200, seed=999
        )
        _, _, test = split_dataset(data, train_ratio=0.0, val_ratio=0.0, seed=args.seed)

    speedup = benchmark_speed(
        model,
        normalizer,
        n_scenarios=args.n_scenarios,
        rk45_samples=args.rk45_samples,
        device=device,
        seed=args.seed,
    )
    rmse_arr, peak_arr, _ = evaluate_accuracy(model, normalizer, test, device=device)

    plot_predictions(
        model,
        normalizer,
        test["params"][:6],
        test["x_data"][:6],
        test["t_grid"],
        output_dir / "deeponet_predictions.png",
    )

    print(f"\n{'='*60}")
    print("Evaluation Summary")
    print(f"{'='*60}")
    print(f"  Speedup vs RK45: {speedup:.0f}x")
    print(f"  Avg RMSE: {np.mean(rmse_arr)*1000:.2f} mm")
    print(f"  Peak error: {np.mean(peak_arr)*100:.1f}%")


if __name__ == "__main__":
    main()
