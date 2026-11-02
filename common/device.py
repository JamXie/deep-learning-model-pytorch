"""设备选择，统一处理 auto / cpu / cuda。"""

from __future__ import annotations

import torch


def get_device(choice: str = "auto") -> torch.device:
    """按 auto / cpu / cuda 返回 torch.device。

    auto 时有 CUDA 就用 CUDA，否则回退到 CPU。
    """
    choice = choice.lower()
    if choice == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if choice == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("指定了 --device cuda，但当前环境没有可用的 CUDA。")
    return torch.device(choice)
