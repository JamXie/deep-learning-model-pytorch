"""DeepONet：Branch 网络编码输入函数，Trunk 网络编码查询位置，最后做内积。"""

from __future__ import annotations

import torch
import torch.nn as nn


class FNN(nn.Module):
    """标准全连接网络；最后一层不加激活函数。"""

    def __init__(self, layer_sizes: list[int], activation=nn.ReLU) -> None:
        super().__init__()
        layers = []
        for i in range(len(layer_sizes) - 1):
            layers.append(nn.Linear(layer_sizes[i], layer_sizes[i + 1]))
            if i < len(layer_sizes) - 2:
                layers.append(activation())
        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class DeepONet(nn.Module):
    """Branch: u(x) -> ψ；Trunk: y -> φ；输出为 sum(ψ * φ)。"""

    def __init__(self, branch_layers: list[int], trunk_layers: list[int]) -> None:
        super().__init__()
        self.branch = FNN(branch_layers)
        self.trunk = FNN(trunk_layers)

    def forward(self, inputs):
        u, y = inputs
        b = self.branch(u)  # (batch, p)
        t = self.trunk(y)  # (batch, p)
        return torch.sum(b * t, dim=1, keepdim=True)
