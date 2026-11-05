"""1CT 矩形薄壁梁斜向压溃数据集的读取、归一化与划分。

数据集为 600 条有限元仿真结果：
  输入：长 l、宽 w、高 h、壁厚 t、加载角度 LoadAngle、屈服强度 sigma
  输出：平均压溃力 MeanCrashingForce、峰值压溃力 PeakCrashingForce、
        比吸能 SpecificEnergyAbsorption
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

from common.paths import resolve_data_path

# 输入、输出列名与原始工程保持一致
INPUT_COLUMNS = ["l", "w", "h", "t", "LoadAngle", "sigma"]
OUTPUT_COLUMNS = ["MeanCrashingForce", "PeakCrashingForce", "SpecificEnergyAbsorption"]

# 固定归一化范围（取自原始实验设置），保证不同网络之间可比
INPUT_RANGES = [(30, 90), (30, 90), (150, 300), (1, 5), (0, 40), (200, 800)]
OUTPUT_RANGES = [(0, 600), (0, 1000), (0, 250)]

DATA_FILE = Path("crashworthiness") / "result_1ct_oblique_load.csv"


def load_crash_frame(path: str | Path | None = None, data_root: str | Path | None = None) -> pd.DataFrame:
    """读取 CSV；原文件为 GB2312，若失败则依次尝试 GB18030 与 UTF-8。"""
    csv_path = Path(path) if path is not None else resolve_data_path(DATA_FILE, root=data_root)
    if not csv_path.exists():
        raise FileNotFoundError(
            f"未找到数据集：{csv_path}\n"
            "请确认 --data-root 或环境变量 DLM_DATA_ROOT 指向包含 crashworthiness/ 的目录。"
        )

    last_error: Exception | None = None
    for encoding in ("gb2312", "gb18030", "utf-8"):
        try:
            return pd.read_csv(csv_path, encoding=encoding)
        except UnicodeDecodeError as exc:  # 编码不匹配时尝试下一种
            last_error = exc
    raise ValueError(f"无法解码 {csv_path}，最后错误：{last_error}")


def normalize_frame(
    frame: pd.DataFrame,
    input_ranges=INPUT_RANGES,
    output_ranges=OUTPUT_RANGES,
) -> tuple[np.ndarray, np.ndarray]:
    """按固定范围做 Min-Max 归一化，返回 (x, y)。"""
    x = frame[INPUT_COLUMNS].to_numpy(dtype=np.float32)
    y = frame[OUTPUT_COLUMNS].to_numpy(dtype=np.float32)

    x_range = np.asarray(input_ranges, dtype=np.float32)
    y_range = np.asarray(output_ranges, dtype=np.float32)
    x = (x - x_range[:, 0]) / (x_range[:, 1] - x_range[:, 0])
    y = (y - y_range[:, 0]) / (y_range[:, 1] - y_range[:, 0])
    return x, y


def split_dataset(
    x: np.ndarray,
    y: np.ndarray,
    test_size: float = 0.2,
    seed: int = 42,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """按 80/20 划分训练集与验证集。"""
    return train_test_split(x, y, test_size=test_size, random_state=seed)
