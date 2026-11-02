"""高斯过程回归（GPR）：在 cos(x) 上演示后验均值与 95% 置信带。

运行方式（在仓库根目录）：
    python -m examples.gpr.gpr_basic
"""

from __future__ import annotations

import argparse

import matplotlib.pyplot as plt
import numpy as np

from common.cli import add_common_args
from common.paths import resolve_output_dir
from common.plotting import save_figure, use_headless_backend
from common.seed import set_seed


class GPR:
    """最简 GPR：RBF 核，超参数固定，不做优化。"""

    def __init__(self, optimize: bool = True) -> None:
        self.is_fit = False
        self.train_X: np.ndarray | None = None
        self.train_y: np.ndarray | None = None
        self.params = {"l": 0.5, "sigma_f": 0.2}
        self.optimize = optimize

    def fit(self, X, y) -> None:
        self.train_X = np.asarray(X)
        self.train_y = np.asarray(y)
        self.is_fit = True

    def predict(self, X):
        if not self.is_fit:
            raise RuntimeError("GPR 模型尚未拟合。")

        X = np.asarray(X)
        Kff = self.kernel(self.train_X, self.train_X)  # (N, N)
        Kyy = self.kernel(X, X)  # (k, k)
        Kfy = self.kernel(self.train_X, X)  # (N, k)
        Kff_inv = np.linalg.inv(Kff + 1e-8 * np.eye(len(self.train_X)))

        mu = Kfy.T.dot(Kff_inv).dot(self.train_y)
        cov = Kyy - Kfy.T.dot(Kff_inv).dot(Kfy)
        return mu, cov

    def kernel(self, x1, x2):
        dist_matrix = np.sum(x1 ** 2, 1).reshape(-1, 1) + np.sum(x2 ** 2, 1) - 2 * np.dot(x1, x2.T)
        return self.params["sigma_f"] ** 2 * np.exp(-0.5 / self.params["l"] ** 2 * dist_matrix)


def target(x, noise_sigma: float = 0.0):
    x = np.asarray(x)
    return (np.cos(x) + np.random.normal(0, noise_sigma, size=x.shape)).tolist()


def plot_gpr(gpr: GPR, train_X, train_y, test_X, mu, cov, path) -> None:
    test_y = mu.ravel()
    # 协方差对角线可能因数值误差略小于 0，先截断再开方
    uncertainty = 1.96 * np.sqrt(np.clip(np.diag(cov), 0.0, None))

    fig, ax = plt.subplots(figsize=(7, 4.5))
    ax.set_title("l=%.2f sigma_f=%.2f" % (gpr.params["l"], gpr.params["sigma_f"]))
    ax.fill_between(test_X.ravel(), test_y + uncertainty, test_y - uncertainty, alpha=0.1, label="95% CI")
    ax.plot(test_X, test_y, label="predict")
    ax.scatter(train_X, train_y, label="train", c="red", marker="x")
    ax.legend()
    ax.grid(alpha=0.3)
    save_figure(fig, path)


def main(argv: list[str] | None = None) -> None:
    use_headless_backend()
    parser = argparse.ArgumentParser(description="基础 GPR 示例")
    add_common_args(parser, epochs=False)
    args = parser.parse_args(argv)

    set_seed(args.seed)
    output_dir = resolve_output_dir(args.output_dir, "gpr")

    train_X = np.array([3, 1, 4, 5, 9]).reshape(-1, 1)
    train_y = target(train_X, noise_sigma=1e-4)
    test_X = np.arange(0, 10, 0.1).reshape(-1, 1)

    gpr = GPR()
    gpr.fit(train_X, train_y)
    mu, cov = gpr.predict(test_X)

    plot_gpr(gpr, train_X, train_y, test_X, mu, cov, output_dir / "gpr_basic.png")
    print(f"Saved figure to {output_dir}")


if __name__ == "__main__":
    main()
