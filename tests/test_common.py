"""common/ 下共享工具的最小验证。"""

from __future__ import annotations

import random

import matplotlib.pyplot as plt
import numpy as np
import torch

from common.cli import add_common_args
from common.paths import resolve_data_path, resolve_output_dir
from common.plotting import save_figure, use_headless_backend
from common.seed import set_seed


def test_set_seed_reproducible():
    """同一随机种子应生成完全相同的随机数。"""
    set_seed(123)
    values_a = (random.random(), float(np.random.rand()), float(torch.rand(1)))

    set_seed(123)
    values_b = (random.random(), float(np.random.rand()), float(torch.rand(1)))

    assert values_a == values_b


def test_resolve_data_path(tmp_path):
    """未给 root 时以 DATA_ROOT 为基准，给 root 时以 root 为基准。"""
    default_path = resolve_data_path("crashworthiness", "sample.csv")
    assert default_path.name == "sample.csv"
    assert default_path.parent.name == "crashworthiness"

    custom = resolve_data_path("a", "b.txt", root=tmp_path)
    assert custom == tmp_path / "a" / "b.txt"


def test_resolve_output_dir_creates_directory(tmp_path):
    """显式 output-dir 应被创建并直接返回。"""
    target = resolve_output_dir(tmp_path / "run", "ignored", "parts")
    assert target == tmp_path / "run"
    assert target.is_dir()


def test_save_figure(tmp_path):
    """save_figure 应写出非空 PNG。"""
    use_headless_backend()
    fig, ax = plt.subplots()
    ax.plot([0, 1], [0, 1])
    path = save_figure(fig, tmp_path / "figure.png")

    assert path.exists()
    assert path.stat().st_size > 0


def test_add_common_args():
    """公共参数应包含种子、设备与调试开关。"""
    import argparse

    parser = add_common_args(argparse.ArgumentParser(), epochs=False)
    args = parser.parse_args([])
    assert args.seed == 42
    assert args.device == "auto"
    assert args.output_dir is None
    assert args.limit_batches is None
