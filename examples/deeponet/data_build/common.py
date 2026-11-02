"""数据构建共用工具：GRF 采样与 Excel 保存。"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd


def generate_grf(
    n: int,
    grid: np.ndarray,
    length_scale: float = 0.2,
    rng: np.random.RandomState | None = None,
) -> np.ndarray:
    """用平方指数协方差采样 n 条高斯随机场曲线，形状 (n, len(grid))。"""
    rng = rng if rng is not None else np.random.RandomState(0)
    x1, x2 = np.meshgrid(grid, grid)
    cov = np.exp(-((x1 - x2) ** 2) / (2 * length_scale ** 2))
    cov += 1e-6 * np.eye(len(grid))  # 数值稳定
    chol = np.linalg.cholesky(cov)
    return np.dot(rng.randn(n, len(grid)), chol.T)


def save_to_excel(
    branch: np.ndarray,
    query: np.ndarray,
    output: np.ndarray,
    path: str | Path,
    branch_sheet: str = "u_function",
) -> None:
    """保存为三个 sheet：函数值、查询点、输出值。"""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    query_columns = ["y"] if query.shape[1] == 1 else [f"q{i}" for i in range(query.shape[1])]

    with pd.ExcelWriter(path) as writer:
        pd.DataFrame(branch).to_excel(writer, sheet_name=branch_sheet, index=False)
        pd.DataFrame(query, columns=query_columns).to_excel(writer, sheet_name="y_location", index=False)
        pd.DataFrame(output, columns=["s"]).to_excel(writer, sheet_name="s_output", index=False)
    print(f"Saved: {path}")


def save_split(
    branch: np.ndarray,
    query: np.ndarray,
    output: np.ndarray,
    output_dir: str | Path,
    train_ratio: float = 0.8,
    branch_sheet: str = "u_function",
) -> None:
    """按比例拆分训练/验证并写入 train.xlsx、val.xlsx。"""
    split = int(train_ratio * len(branch))
    save_to_excel(branch[:split], query[:split], output[:split], Path(output_dir) / "train.xlsx", branch_sheet)
    save_to_excel(branch[split:], query[split:], output[split:], Path(output_dir) / "val.xlsx", branch_sheet)
