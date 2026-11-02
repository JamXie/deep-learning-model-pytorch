"""DeepONet 数据路径与网络结构配置。

三种任务：
    1d  积分算子      s(y) = ∫_0^y u(x) dx
    2d  受迫单摆      (u(t), t) -> s(t)
    3d  一维扩散 PDE  (f(x), (x,t)) -> s(x,t)
"""

from __future__ import annotations

from pathlib import Path

from common.paths import OUTPUT_ROOT

TASK_QUERY_DIM = {"1d": 1, "2d": 1, "3d": 2}
TASKS = tuple(TASK_QUERY_DIM)


class Config:
    """把任务相关的路径与网络结构集中在一处。"""

    def __init__(
        self,
        task: str = "1d",
        m: int = 50,
        p: int = 50,
        data_dir: str | Path | None = None,
        output_dir: str | Path | None = None,
        lr: float = 1e-3,
        epochs: int = 200,
        batch_size: int = 64,
    ) -> None:
        if task not in TASK_QUERY_DIM:
            raise ValueError(f"未知任务 {task}，可选：{TASKS}")

        self.task = task
        self.m = m
        self.p = p
        self.query_dim = TASK_QUERY_DIM[task]

        # 数据默认放在 outputs/deeponet/<task>/ 下
        self.data_dir = Path(data_dir) if data_dir is not None else OUTPUT_ROOT / "deeponet" / task
        self.train_path = self.data_dir / "train.xlsx"
        self.test_path = self.data_dir / "val.xlsx"
        self.test_path_extra = self.data_dir / "extra.xlsx"

        self.output_dir = Path(output_dir) if output_dir is not None else OUTPUT_ROOT / "deeponet" / task
        self.checkpoint_path = self.output_dir / "deeponet.pth"

        # branch 输入是 m 个离散函数值；trunk 输入是 query_dim 维查询坐标
        self.branch_layers = [m, 100, 100, p]
        self.trunk_layers = [self.query_dim, 100, 100, p]

        self.lr = lr
        self.epochs = epochs
        self.batch_size = batch_size
