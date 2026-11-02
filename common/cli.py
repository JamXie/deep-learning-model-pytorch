"""命令行公共参数，减少各示例入口的重复定义。"""

from __future__ import annotations

import argparse


def add_common_args(parser: argparse.ArgumentParser, epochs: bool = True) -> argparse.ArgumentParser:
    """给 parser 添加通用的随机种子、设备、路径与调试参数。"""
    parser.add_argument("--seed", type=int, default=42, help="随机种子")
    parser.add_argument("--device", default="auto", choices=["auto", "cpu", "cuda"], help="计算设备")
    parser.add_argument("--data-root", default=None, help="数据根目录（默认 <repo>/data）")
    parser.add_argument("--output-dir", default=None, help="输出目录（默认 <repo>/outputs/<example>）")
    parser.add_argument("--limit-batches", type=int, default=None, help="每个 epoch 最多运行的 batch 数，便于快速冒烟")
    if epochs:
        parser.add_argument("--epochs", type=int, default=5, help="训练轮数")
        parser.add_argument("--batch-size", type=int, default=64, help="批大小")
    return parser
