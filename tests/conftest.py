"""pytest 公共设置：把仓库根目录加入导入路径，并强制无界面绘图后端。"""

from __future__ import annotations

import sys
from pathlib import Path

import matplotlib

# 保证在任何测试导入 pyplot 之前切换到 Agg
matplotlib.use("Agg")

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))
