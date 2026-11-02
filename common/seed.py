"""统一设置随机种子，保证示例可复现。"""

from __future__ import annotations

import os
import random

import numpy as np


def set_seed(seed: int = 42, deterministic: bool = False) -> None:
    """设置 random / NumPy / PyTorch 的种子。

    deterministic=True 时额外要求 cuDNN 使用确定性算法；CPU 上无影响。
    """
    random.seed(seed)
    np.random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)

    # torch 是可选依赖：某些仅做数据处理的环境可能没装
    try:
        import torch
    except ImportError:  # pragma: no cover - 依赖缺失时直接返回
        return

    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    if deterministic:
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
