"""所有入口脚本都应可导入且不产生副作用（不训练、不下载数据）。"""

from __future__ import annotations

import importlib

import pytest

MODULES = [
    # 共享工具
    "common.cli",
    "common.device",
    "common.paths",
    "common.plotting",
    "common.seed",
    # 表格回归与插值
    "examples._shared.crash_data",
    "examples.ann.train",
    "examples.grnn.run",
    "examples.rbf_network.train",
    "examples.interpolation.rbf",
    "examples.interpolation.griddata",
    # 高斯过程回归
    "examples.gpr.gpr_basic",
    "examples.gpr.gpr_hyperparameters",
    "examples.gpr.gpr_uncertain_inputs",
    # 生成模型
    "examples.diffusion.train",
    "examples.pixelcnn.train",
    "examples.vqvae.train",
    # 时序 Transformer 与 Moving MNIST
    "examples.moving_mnist.generate",
    "examples.moving_mnist.visualize",
    "examples.transformer.train",
    # DeepONet
    "examples.deeponet.config",
    "examples.deeponet.dataload",
    "examples.deeponet.train",
    "examples.deeponet.validation",
    "examples.deeponet.model.deeponet",
    "examples.deeponet.model.fnn",
    "examples.deeponet.data_build.common",
    "examples.deeponet.data_build.build_1d_integral",
    "examples.deeponet.data_build.build_2d_pendulum",
    "examples.deeponet.data_build.build_3d_diffusion",
    # 碰撞安全 SciML
    "examples.sciml_crash.pinns.models",
    "examples.sciml_crash.pinns.losses",
    "examples.sciml_crash.pinns.pinn_strong_form",
    "examples.sciml_crash.pinns.pinn_energy_form",
    "examples.sciml_crash.pinns.utils",
    "examples.sciml_crash.pinns.case_linear",
    "examples.sciml_crash.pinns.case_bilinear",
    "examples.sciml_crash.pinns.case_rate_dependent",
    "examples.sciml_crash.operator.deeponet_model",
    "examples.sciml_crash.operator.data_generator",
    "examples.sciml_crash.operator.train_operator",
    "examples.sciml_crash.operator.eval_operator",
    "examples.sciml_crash.pino.pino_pipeline",
    "examples.sciml_crash.pino.benchmark",
]


@pytest.mark.parametrize("module", MODULES)
def test_module_imports(module):
    assert importlib.import_module(module) is not None
