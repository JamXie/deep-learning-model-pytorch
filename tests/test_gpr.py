"""GPR 示例：三个脚本中的类均可拟合并在小数据上给出有限预测。"""

from __future__ import annotations

import numpy as np

from examples.gpr.gpr_basic import GPR
from examples.gpr.gpr_hyperparameters import GPR as GPRWithOpt
from examples.gpr.gpr_uncertain_inputs import GPRVariances


def _small_data():
    train_x = np.array([1.0, 2.0, 3.0, 4.0, 5.0]).reshape(-1, 1)
    train_y = np.cos(train_x).ravel().tolist()
    test_x = np.linspace(0.0, 6.0, 7).reshape(-1, 1)
    return train_x, train_y, test_x


def test_gpr_basic_predict_shapes():
    train_x, train_y, test_x = _small_data()
    model = GPR(optimize=False)
    model.fit(train_x, train_y)
    mu, cov = model.predict(test_x)

    assert mu.shape == (7,)
    assert cov.shape == (7, 7)
    assert np.isfinite(mu).all()
    assert np.isfinite(cov).all()


def test_gpr_hyperparameters_positive():
    train_x, train_y, test_x = _small_data()
    model = GPRWithOpt(optimize=True)
    model.fit(train_x, train_y)
    mu, cov = model.predict(test_x)

    assert model.params["l"] > 0
    assert model.params["sigma_f"] > 0
    assert mu.shape == (7,)
    assert np.isfinite(mu).all()


def test_gpr_uncertain_inputs_with_observation_noise():
    train_x, train_y, test_x = _small_data()
    variances = np.array([1e-3, 2e-3, 5e-3, 1e-2, 1e-4])

    model = GPRVariances()
    model.fit(train_x, train_y, obs_variances=variances)
    mu, cov = model.predict(test_x)

    assert mu.shape == (7,)
    assert cov.shape == (7, 7)
    assert np.isfinite(mu).all()
