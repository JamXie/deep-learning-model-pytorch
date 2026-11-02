"""基线全连接网络：直接把 u 与查询点拼接后回归，不学算子分解。"""

from __future__ import annotations

import torch
import torch.nn as nn


class FNNBaseline(nn.Module):
    """输入维度为 m + query_dim，1D/2D 查询均可复用。"""

    def __init__(self, m: int, query_dim: int = 1, hidden_dim: int = 100) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(m + query_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1),
        )

    def forward(self, u: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        return self.net(torch.cat([u, y], dim=1))
