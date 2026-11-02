"""训练 DeepONet（或 FNN 基线）。

运行方式（在仓库根目录）：
    # 先生成数据
    python -m examples.deeponet.data_build.build_1d_integral --num-train 2000 --num-val 500
    # 再训练
    python -m examples.deeponet.train --task 1d --epochs 200
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import torch

from common.cli import add_common_args
from common.device import get_device
from common.paths import resolve_output_dir
from common.plotting import save_figure, use_headless_backend
from common.seed import set_seed
from examples.deeponet.config import TASKS, Config
from examples.deeponet.dataload import get_dataloader, load_from_excel
from examples.deeponet.model.fnn import FNNBaseline
from examples.deeponet.model.deeponet import DeepONet


def build_model(config: Config, model_name: str) -> torch.nn.Module:
    if model_name == "deeponet":
        return DeepONet(config.branch_layers, config.trunk_layers)
    if model_name == "fnn":
        return FNNBaseline(config.m, config.query_dim)
    raise ValueError(f"未知模型 {model_name}")


def predict(model: torch.nn.Module, u: torch.Tensor, y: torch.Tensor, model_name: str) -> torch.Tensor:
    """统一 DeepONet 的 tuple 输入与 FNN 的双参数输入。"""
    if model_name == "fnn":
        return model(u, y)
    return model((u, y))


def evaluate(model, u, y, s, device, model_name: str) -> float:
    model.eval()
    with torch.no_grad():
        u_t = torch.as_tensor(u, dtype=torch.float32, device=device)
        y_t = torch.as_tensor(y, dtype=torch.float32, device=device)
        s_t = torch.as_tensor(s, dtype=torch.float32, device=device)
        pred = predict(model, u_t, y_t, model_name)
        return torch.mean((pred - s_t) ** 2).item()


def main(argv: list[str] | None = None) -> None:
    use_headless_backend()
    parser = argparse.ArgumentParser(description="训练 DeepONet / FNN 算子模型")
    add_common_args(parser, epochs=False)
    parser.add_argument("--task", choices=TASKS, default="1d", help="任务类型")
    parser.add_argument("--model", choices=["deeponet", "fnn"], default="deeponet")
    parser.add_argument("--epochs", type=int, default=200)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--m", type=int, default=50, help="分支网络输入离散点数")
    parser.add_argument("--p", type=int, default=50, help="隐层输出维度")
    parser.add_argument("--data-dir", default=None, help="数据目录（含 train.xlsx/val.xlsx）")
    args = parser.parse_args(argv)

    set_seed(args.seed)
    device = get_device(args.device)

    config = Config(
        task=args.task,
        m=args.m,
        p=args.p,
        data_dir=args.data_dir,
        lr=args.lr,
        epochs=args.epochs,
        batch_size=args.batch_size,
    )
    output_dir = resolve_output_dir(args.output_dir, "deeponet", args.task)
    config.output_dir = output_dir
    config.checkpoint_path = output_dir / f"{args.model}.pth"

    if not config.train_path.exists():
        raise FileNotFoundError(
            f"未找到 {config.train_path}\n"
            f"请先运行：python -m examples.deeponet.data_build.build_{args.task}_"
            f"{'integral' if args.task == '1d' else 'pendulum' if args.task == '2d' else 'diffusion'}"
        )

    train_loader = get_dataloader(config.train_path, args.batch_size, shuffle=True)
    u_val, y_val, s_val = load_from_excel(config.test_path)

    model = build_model(config, args.model).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=config.lr)
    loss_fn = torch.nn.MSELoss()

    train_losses: list[float] = []
    val_losses: list[float] = []
    for epoch in range(config.epochs):
        model.train()
        total = 0.0
        for step, ((u, y), s) in enumerate(train_loader):
            if args.limit_batches is not None and step >= args.limit_batches:
                break
            u, y, s = u.to(device), y.to(device), s.to(device)
            pred = predict(model, u, y, args.model)
            loss = loss_fn(pred, s)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total += loss.item()

        train_losses.append(total / (step + 1))
        val_losses.append(evaluate(model, u_val, y_val, s_val, device, args.model))
        if (epoch + 1) % 20 == 0 or epoch == 0:
            print(f"epoch {epoch + 1}/{config.epochs} train={train_losses[-1]:.6f} val={val_losses[-1]:.6f}")

    fig, ax = plt.subplots(figsize=(6, 4))
    ax.plot(range(1, len(train_losses) + 1), train_losses, label="train")
    ax.plot(range(1, len(val_losses) + 1), val_losses, label="val")
    ax.set_xlabel("epoch")
    ax.set_ylabel("MSE loss")
    ax.set_title(f"{args.task} / {args.model}")
    ax.legend()
    ax.grid(alpha=0.3)
    save_figure(fig, output_dir / f"{args.model}_loss.png")

    torch.save(
        {
            "state_dict": model.state_dict(),
            "task": args.task,
            "model": args.model,
            "m": config.m,
            "p": config.p,
            "query_dim": config.query_dim,
        },
        config.checkpoint_path,
    )
    print(f"Saved checkpoint to {config.checkpoint_path}")


if __name__ == "__main__":
    main()
