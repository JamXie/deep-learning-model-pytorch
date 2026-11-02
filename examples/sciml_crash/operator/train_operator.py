"""训练 DeepONet 进行碰撞响应预测。

学习算子：(碰撞参数) -> (位移时间历程)

运行（在仓库根目录）：
    # 先生成数据（可选；缺失时会自动生成）
    python -m examples.sciml_crash.operator.data_generator --num-samples 500
    # 再训练
    python -m examples.sciml_crash.operator.train_operator --epochs 300
"""

from __future__ import annotations

import argparse
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn

from common.cli import add_common_args
from common.device import get_device
from common.paths import resolve_output_dir
from common.seed import set_seed
from examples.sciml_crash.operator.data_generator import (
    generate_crash_dataset,
    load_dataset,
    save_dataset,
    split_dataset,
)
from examples.sciml_crash.operator.deeponet_model import DeepONet, ParameterNormalizer


def prepare_training_data(data, normalizer, device="cpu"):
    """准备 DeepONet 训练数据（逐点形式）。

    每个训练点 = (params_i, t_j, x_i(t_j))；
    每批次随机采样（参数，时间）对。
    """
    params = data["params"].astype(np.float32)
    t_grid = data["t_grid"].astype(np.float32)
    x_data = data["x_data"].astype(np.float32)

    # 归一化参数，并把时间归一化到 [0, 1]
    params_norm = normalizer.normalize(params)
    t_norm = t_grid / t_grid[-1]

    n_samples, n_time = x_data.shape

    # 展平为逐点形式：每行 = (params_i, t_j, x_i(t_j))
    params_rep = np.repeat(params_norm, n_time, axis=0)
    t_rep = np.tile(t_norm, n_samples).reshape(-1, 1)
    x_flat = x_data.reshape(-1, 1)

    params_t = torch.tensor(params_rep, dtype=torch.float32, device=device)
    t_t = torch.tensor(t_rep, dtype=torch.float32, device=device)
    x_t = torch.tensor(x_flat, dtype=torch.float32, device=device)

    return params_t, t_t, x_t, t_grid


def train_deeponet(
    train_data,
    val_data,
    n_params=6,
    p=100,
    batch_size=1024,
    n_epochs=300,
    lr=1e-3,
    device="cpu",
    limit_batches=None,
):
    """在碰撞数据集上训练 DeepONet。

    返回：
        model: 训练好的 DeepONet
        normalizer: 参数归一化器
        history: 训练历史记录字典
    """
    # 用训练集 + 验证集统计参数范围
    all_params = np.vstack([train_data["params"], val_data["params"]])
    normalizer = ParameterNormalizer(all_params.min(axis=0), all_params.max(axis=0))

    print("Preparing training data...")
    params_t, t_t, x_t, _ = prepare_training_data(train_data, normalizer, device)
    dataset_size = params_t.shape[0]
    print(f"  Training points: {dataset_size}")

    # 验证数据
    val_params_norm = normalizer.normalize(val_data["params"].astype(np.float32))
    val_t_norm = val_data["t_grid"].astype(np.float32) / val_data["t_grid"][-1]
    val_x = val_data["x_data"].astype(np.float32)

    # 构建模型
    model = DeepONet(n_params=n_params, p=p).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", factor=0.5, patience=20, min_lr=1e-6
    )
    loss_fn = nn.MSELoss()

    history = {"train_loss": [], "val_rmse": [], "epoch_time": []}

    print(f"\nTraining DeepONet (p={p}, epochs={n_epochs}, batch={batch_size})...")
    print(f"  Model params: {sum(p.numel() for p in model.parameters()):,}")

    for epoch in range(n_epochs):
        epoch_start = time.time()
        model.train()

        perm = torch.randperm(dataset_size, device=device)
        epoch_loss = 0.0
        n_batches = 0

        for i in range(0, dataset_size, batch_size):
            if limit_batches is not None and n_batches >= limit_batches:
                break
            idx = perm[i : i + batch_size]
            optimizer.zero_grad()
            x_pred = model(params_t[idx], t_t[idx])
            loss = loss_fn(x_pred, x_t[idx])
            loss.backward()
            optimizer.step()

            epoch_loss += loss.item()
            n_batches += 1

        avg_loss = epoch_loss / n_batches
        history["train_loss"].append(avg_loss)
        history["epoch_time"].append(time.time() - epoch_start)

        # 定期验证
        if epoch % 10 == 0 or epoch == n_epochs - 1:
            model.eval()
            with torch.no_grad():
                val_params_t = torch.tensor(val_params_norm, dtype=torch.float32, device=device)
                val_t_t = torch.tensor(
                    val_t_norm.reshape(-1, 1), dtype=torch.float32, device=device
                )
                val_pred = model.predict_trajectory(val_params_t, val_t_t)

                val_x_t = torch.tensor(val_x, dtype=torch.float32, device=device)
                val_rmse = torch.sqrt(torch.mean((val_pred - val_x_t) ** 2)).item()
                history["val_rmse"].append(val_rmse)

                # 峰值位移误差
                peak_true = torch.max(torch.abs(val_x_t), dim=1)[0]
                peak_pred = torch.max(torch.abs(val_pred), dim=1)[0]
                peak_re = torch.mean(torch.abs(peak_pred - peak_true) / (peak_true + 1e-8)).item()
            scheduler.step(val_rmse)

            print(
                f"  Epoch {epoch:4d} | Loss: {avg_loss:.4e} | "
                f"Val RMSE: {val_rmse*1000:.3f} mm | "
                f"Peak RE: {peak_re*100:.1f}% | "
                f"Time: {history['epoch_time'][-1]:.1f}s"
            )

    print("\nTraining complete!")
    print(f"  Final val RMSE: {history['val_rmse'][-1]*1000:.3f} mm")
    return model, normalizer, history


def save_checkpoint(model, normalizer, history, save_path):
    """保存训练好的模型和元数据（不含 NumPy 对象，便于跨版本加载）。"""
    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "p": model.p,
            "normalizer_state": {
                "mins": normalizer.param_mins.tolist(),
                "maxs": normalizer.param_maxs.tolist(),
            },
            "history": history,
        },
        save_path,
    )
    print(f"Model saved to {save_path}")


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="训练碰撞响应 DeepONet")
    add_common_args(parser, epochs=False)
    parser.add_argument("--epochs", type=int, default=300)
    parser.add_argument("--batch-size", type=int, default=1024)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--p", type=int, default=100, help="隐空间维度（基函数数量）")
    parser.add_argument("--data-path", default=None, help="crash_dataset.pkl 路径")
    parser.add_argument("--checkpoint", default=None, help="权重输出路径")
    parser.add_argument("--num-samples", type=int, default=500, help="无数据文件时自动生成的样本数")
    parser.add_argument("--num-time", type=int, default=200, help="无数据文件时的时间点数")
    args = parser.parse_args(argv)

    set_seed(args.seed)
    device = get_device(args.device)
    output_dir = resolve_output_dir(args.output_dir, "sciml_crash")
    data_path = Path(args.data_path) if args.data_path else output_dir / "crash_dataset.pkl"
    checkpoint_path = Path(args.checkpoint) if args.checkpoint else output_dir / "deeponet_crash.pt"

    print(f"Using device: {device}")
    print("=" * 60)
    print("Training DeepONet for Crash Response Operator Learning")
    print("=" * 60)

    # 加载或生成数据集
    if data_path.exists():
        payload = load_dataset(data_path)
    else:
        data = generate_crash_dataset(
            n_samples=args.num_samples, n_time_points=args.num_time, seed=args.seed
        )
        train, val, test = split_dataset(data, seed=args.seed)
        payload = {"full": data, "train": train, "val": val, "test": test}
        save_dataset(data_path, payload)

    train, val, test = payload["train"], payload["val"], payload["test"]
    print(f"  Train: {len(train['params'])} | Val: {len(val['params'])} | Test: {len(test['params'])}")

    model, normalizer, history = train_deeponet(
        train,
        val,
        n_params=6,
        p=args.p,
        batch_size=args.batch_size,
        n_epochs=args.epochs,
        lr=args.lr,
        device=device,
        limit_batches=args.limit_batches,
    )
    save_checkpoint(model, normalizer, history, checkpoint_path)

    # 在未见参数上快速测试
    print("\n--- Quick test on unseen parameters ---")
    model.eval()
    with torch.no_grad():
        test_params = test["params"][:5].astype(np.float32)
        test_x = test["x_data"][:5]
        test_t = test["t_grid"]

        test_params_norm = normalizer.normalize(test_params)
        test_t_norm = test_t.astype(np.float32) / test_t[-1]
        pred = model.predict_trajectory(
            torch.tensor(test_params_norm, dtype=torch.float32, device=device),
            torch.tensor(test_t_norm.reshape(-1, 1), dtype=torch.float32, device=device),
        ).cpu().numpy()

    for i in range(min(5, len(test_params))):
        rmse = np.sqrt(np.mean((pred[i] - test_x[i]) ** 2))
        print(
            f"  Sample {i}: m={test_params[i, 0]:.0f}kg, "
            f"v0={abs(test_params[i, 5])*3.6:.0f}km/h -> RMSE={rmse*1000:.3f}mm"
        )


if __name__ == "__main__":
    main()
