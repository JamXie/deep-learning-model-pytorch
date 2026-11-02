"""跨示例共用的小工具：路径、随机种子、设备、绘图与命令行参数。"""

from .device import get_device
from .paths import (
    DATA_ROOT,
    OUTPUT_ROOT,
    REPO_ROOT,
    ensure_output_dir,
    resolve_data_path,
    resolve_output_dir,
)
from .seed import set_seed

__all__ = [
    "DATA_ROOT",
    "OUTPUT_ROOT",
    "REPO_ROOT",
    "ensure_output_dir",
    "get_device",
    "resolve_data_path",
    "resolve_output_dir",
    "set_seed",
]
