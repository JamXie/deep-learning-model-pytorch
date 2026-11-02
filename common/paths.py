"""仓库路径与环境变量，避免在示例里写死本机绝对路径。"""

from __future__ import annotations

import os
from pathlib import Path

# 仓库根目录（common/ 的上一级）
REPO_ROOT = Path(__file__).resolve().parent.parent

# 数据根目录：默认 <repo>/data，可用环境变量 DLM_DATA_ROOT 覆盖
DATA_ROOT = Path(os.environ.get("DLM_DATA_ROOT", REPO_ROOT / "data")).expanduser()

# 输出根目录：默认 <repo>/outputs，可用环境变量 DLM_OUTPUT_ROOT 覆盖
OUTPUT_ROOT = Path(os.environ.get("DLM_OUTPUT_ROOT", REPO_ROOT / "outputs")).expanduser()


def resolve_data_path(*parts: str | Path, root: str | Path | None = None) -> Path:
    """拼接数据路径；传入 root 时以该目录为基准，否则使用 DATA_ROOT。"""
    base = Path(root).expanduser() if root is not None else DATA_ROOT
    return base.joinpath(*parts)


def ensure_output_dir(*parts: str | Path, root: str | Path | None = None) -> Path:
    """创建并返回输出目录，默认位于 OUTPUT_ROOT 下。"""
    base = Path(root).expanduser() if root is not None else OUTPUT_ROOT
    target = base.joinpath(*parts)
    target.mkdir(parents=True, exist_ok=True)
    return target


def resolve_output_dir(explicit: str | Path | None, *default_parts: str | Path) -> Path:
    """用户显式给出 --output-dir 时直接使用，否则拼到 OUTPUT_ROOT 下。"""
    target = Path(explicit).expanduser() if explicit else OUTPUT_ROOT.joinpath(*default_parts)
    target.mkdir(parents=True, exist_ok=True)
    return target
