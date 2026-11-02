"""加载训练好的 DeepONet / FNN 并评估相对 L2 误差。

运行方式（在仓库根目录）：
    python -m examples.deeponet.validation --task 1d --split val
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch

from common.cli import add_common_args
from common.device import get_device
from common.paths import resolve_output_dir
from common.plotting import save_figure, use_headless_backend
from examples.deeponet.config import TASKS, Config
from examples.deeponet.dataload import load_from_excel
from examples.deeponet.model.deeponet import DeepONet
from examples.deeponet.model.fnn import FNNBaseline
from examples.deeponet.train import predict


def relative_l2_error(pred: np.ndarray, true: np.ndarray) -> float:
    return float(np.linalg.norm(pred - true) / (np.linalg.norm(true) + 1e-12))


def main(argv: list[str] | None = None) -> None:
    use_headless_backend()
    parser = argparse.ArgumentParser(description="评估 DeepONet / FNN")
    add_common_args(parser, epochs=False)
    parser.add_argument("--task", choices=TASKS, default="1d")
    parser.add_argument("--model", choices=["deeponet", "fnn"], default="deeponet")
    parser.add_argument("--checkpoint", default=None, help="权重路径")
    parser.add_argument("--split", choices=["train", "val", "extra"], default="val")
    parser.add_argument("--data-dir", default=None, help="数据目录")
    parser.add_argument("--m", type=int, default=50)
    parser.add_argument("--p", type=int, default=50)
    args = parser.parse_args(argv)

    device = get_device(args.device)
    config = Config(task=args.task, m=args.m, p=args.p, data_dir=args.data_dir)
    output_dir = resolve_output_dir(args.output_dir, "deeponet", args.task)

    checkpoint = Path(args.checkpoint) if args.checkpoint else output_dir / f"{args.model}.pth"
    if not checkpoint.exists():
        raise FileNotFoundError(f"未找到权重 {checkpoint}，请先运行训练脚本。")

    data_path = {"train": config.train_path, "val": config.test_path, "extra": config.test_path_extra}[args.split]
    u, y, s_true = load_from_excel(data_path)

    if args.model == "fnn":
        model = FNNBaseline(config.m, config.query_dim)
    else:
        model = DeepONet(config.branch_layers, config.trunk_layers)
    model.to(device)

    payload = torch.load(checkpoint, map_location=device)
    state_dict = payload["state_dict"] if isinstance(payload, dict) and "state_dict" in payload else payload
    model.load_state_dict(state_dict)
    model.eval()

    with torch.no_grad():
        u_t = torch.as_tensor(u, dtype=torch.float32, device=device)
        y_t = torch.as_tensor(y, dtype=torch.float32, device=device)
        s_pred = predict(model, u_t, y_t, args.model).cpu().numpy()

    error = relative_l2_error(s_pred, s_true)
    print(f"Relative L2 error ({args.split}): {error:.4e}")

    # 预测 vs 真值
    min_val = float(min(s_true.min(), s_pred.min()))
    max_val = float(max(s_true.max(), s_pred.max()))
    fig, ax = plt.subplots(figsize=(5, 5))
    ax.scatter(s_true.ravel(), s_pred.ravel(), s=6, alpha=0.6)
    ax.plot([min_val, max_val], [min_val, max_val], "r--", label="y = x")
    ax.set_xlabel("true")
    ax.set_ylabel("predicted")
    ax.set_title(f"{args.task} / {args.split}\nRelative L2 = {error:.3e}")
    ax.legend()
    ax.grid(alpha=0.3)
    save_figure(fig, output_dir / f"{args.model}_prediction_{args.split}.png")

    # 相对误差分布
    rel = np.abs(s_pred - s_true) / (np.abs(s_true) + 1e-8)
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.hist(rel.ravel(), bins=50)
    ax.set_xlabel("relative error")
    ax.set_ylabel("count")
    ax.set_title("relative error distribution")
    ax.grid(alpha=0.3)
    save_figure(fig, output_dir / f"{args.model}_error_distribution_{args.split}.png")
    print(f"Saved results to {output_dir}")


if __name__ == "__main__":
    main()
