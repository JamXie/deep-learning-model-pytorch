"""从 Excel 读取 (u, y, s) 并构建 PyTorch DataLoader。

Excel 约定：
    - 分支函数 sheet 名为 "u_function"（1d/2d）或 "f_function"（3d）；
    - "y_location" 为查询点（1D 或 2D）；
    - "s_output" 为对应输出值。
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader, Dataset


class DeepONetDataset(Dataset):
    """样本格式为 ((u_i, y_i), s_i)。"""

    def __init__(self, u: np.ndarray, y: np.ndarray, s: np.ndarray) -> None:
        self.u = torch.as_tensor(u, dtype=torch.float32)
        self.y = torch.as_tensor(y, dtype=torch.float32)
        self.s = torch.as_tensor(s, dtype=torch.float32)

    def __len__(self) -> int:
        return len(self.u)

    def __getitem__(self, idx: int):
        return (self.u[idx], self.y[idx]), self.s[idx]


def load_from_excel(filepath: str | Path) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """读取 Excel，返回 (branch, query, output)，形状分别为 (N,m)、(N,1|2)、(N,1)。"""
    path = Path(filepath)
    if not path.exists():
        raise FileNotFoundError(
            f"未找到数据文件 {path}\n"
            "请先运行对应的数据构建脚本，例如：\n"
            "  python -m examples.deeponet.data_build.build_1d_integral"
        )

    xls = pd.ExcelFile(path)
    sheets = xls.sheet_names
    if "u_function" in sheets:
        branch = pd.read_excel(path, sheet_name="u_function").values
    elif "f_function" in sheets:
        branch = pd.read_excel(path, sheet_name="f_function").values
    else:
        raise ValueError(f"{path} 中缺少 u_function 或 f_function sheet")

    y = pd.read_excel(path, sheet_name="y_location").values
    s = pd.read_excel(path, sheet_name="s_output").values
    return branch, y, s


def get_dataloader(
    data_path: str | Path,
    batch_size: int = 64,
    shuffle: bool = True,
) -> DataLoader:
    u, y, s = load_from_excel(data_path)
    return DataLoader(DeepONetDataset(u, y, s), batch_size=batch_size, shuffle=shuffle)
