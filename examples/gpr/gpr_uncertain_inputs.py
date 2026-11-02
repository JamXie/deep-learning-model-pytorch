"""带观测噪声方差的 GPR：把每个样本的噪声方差加到协方差对角线。

运行方式（在仓库根目录）：
    python -m examples.gpr.gpr_uncertain_inputs
"""

from __future__ import annotations

import argparse

import matplotlib.pyplot as plt
import numpy as np

from common.cli import add_common_args
from common.paths import resolve_output_dir
from common.plotting import save_figure, use_headless_backend
from common.seed import set_seed


class GPRVariances:
    """允许每个观测拥有不同噪声方差的 GPR。"""

    def __init__(self, optimize: bool = True) -> None:
        self.is_fit = False
        self.train_X: np.ndarray | None = None
        self.train_y: np.ndarray | None = None
        self.params = {"l": 0.5, "sigma_f": 0.2}
        self.optimize = optimize

    def fit(self, X, y, obs_variances=None) -> None:
        self.train_X = np.asarray(X)
        self.train_y = np.asarray(y)
        if obs_variances is None:
            self.obs_variances = np.zeros(len(X))  # 默认同方差噪声
        else:
            self.obs_variances = np.asarray(obs_variances)
        self.is_fit = True

    def predict(self, X):
        if not self.is_fit:
            raise RuntimeError("GPR 模型尚未拟合。")

        X = np.asarray(X)
        # 训练协方差包含观测噪声
        Kff = self.kernel(self.train_X, self.train_X) + np.diag(self.obs_variances)
        Kyy = self.kernel(X, X)
        Kfy = self.kernel(self.train_X, X)
        Kff_inv = np.linalg.inv(Kff + 1e-8 * np.eye(len(self.train_X)))

        mu = Kfy.T.dot(Kff_inv).dot(self.train_y)
        cov = Kyy - Kfy.T.dot(Kff_inv).dot(Kfy)
        return mu, cov

    def kernel(self, x1, x2):
        dist_matrix = np.sum(x1 ** 2, 1).reshape(-1, 1) + np.sum(x2 ** 2, 1) - 2 * np.dot(x1, x2.T)
        return self.params["sigma_f"] ** 2 * np.exp(-0.5 / self.params["l"] ** 2 * dist_matrix)


def target(x):
    x = np.asarray(x)
    return np.cos(x).tolist()


def main(argv: list[str] | None = None) -> None:
    use_headless_backend()
    parser = argparse.ArgumentParser(description="带观测方差的 GPR 示例")
    add_common_args(parser, epochs=False)
    args = parser.parse_args(argv)

    set_seed(args.seed)
    output_dir = resolve_output_dir(args.output_dir, "gpr")

    train_X = np.array([3, 1, 4, 5, 9]).reshape(-1, 1)
    train_y = target(train_X)
    test_X = np.arange(0, 10, 0.1).reshape(-1, 1)
    # 每个训练点的观测噪声方差
    train_var = np.array([0.001, 0.002, 0.005, 0.015, 0.0001])

    gpr = GPRVariances()
    gpr.fit(train_X, train_y, obs_variances=train_var)
    mu, cov = gpr.predict(test_X)

    test_y = mu.ravel()
    uncertainty = 1.96 * np.sqrt(np.clip(np.diag(cov), 0.0, None))

    fig, ax = plt.subplots(figsize=(7, 4.5))
    ax.set_title("l=%.2f sigma_f=%.2f" % (gpr.params["l"], gpr.params["sigma_f"]))
    ax.fill_between(test_X.ravel(), test_y + uncertainty, test_y - uncertainty, alpha=0.1, label="95% CI")
    ax.plot(test_X, test_y, label="predict")
    ax.scatter(train_X, train_y, label="train", c="red", marker="x")
    ax.legend()
    ax.grid(alpha=0.3)
    save_figure(fig, output_dir / "gpr_uncertain_inputs.png")
    print(f"Saved figure to {output_dir}")


if __name__ == "__main__":
    main()
