"""绘图辅助：无界面后端与保存图片。"""

from __future__ import annotations

from pathlib import Path

import matplotlib


def use_headless_backend() -> None:
    """强制使用无界面后端，避免在服务器或 CI 上弹出窗口。"""
    matplotlib.use("Agg")


def save_figure(fig, path: str | Path, dpi: int = 150, close: bool = True) -> Path:
    """保存 matplotlib 图并返回实际路径。"""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=dpi, bbox_inches="tight")
    if close:
        import matplotlib.pyplot as plt

        plt.close(fig)
    return path
