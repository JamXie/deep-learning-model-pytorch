"""GPR 超参数优化：用负对数边际似然最大化核长度与信号方差。

运行方式（在仓库根目录）：
    python -m examples.gpr.gpr_hyperparameters
"""

from __future__ import annotations

import argparse

import matplotlib.pyplot as plt
import numpy as np
from scipy.optimize import minimize

from common.cli import add_common_args
from common.paths import resolve_output_dir
from common.plotting import save_figure, use_headless_backend
from common.seed import set_seed


class GPR:
    """RBF 核 GPR，可选择通过 L-BFGS-B 优化 (l, sigma_f)。"""

    def __init__(self, optimize: bool = True) -> None:
        self.is_fit = False
        self.train_X: np.ndarray | None = None
        self.train_y: np.ndarray | None = None
        self.params = {"l": 0.5, "sigma_f": 0.2}
        self.optimize = optimize

    def fit(self, X, y) -> None:
        self.train_X = np.asarray(X)
        self.train_y = np.asarray(y)

        def negative_log_likelihood_loss(params):
            self.params["l"], self.params["sigma_f"] = params[0], params[1]
            Kyy = self.kernel(self.train_X, self.train_X) + 1e-8 * np.eye(len(self.train_X))
            loss = (
                0.5 * self.train_y.T.dot(np.linalg.inv(Kyy)).dot(self.train_y)
                + 0.5 * np.linalg.slogdet(Kyy)[1]
                + 0.5 * len(self.train_X) * np.log(2 * np.pi)
            )
            return loss.ravel()

        if self.optimize:
            res = minimize(
                negative_log_likelihood_loss,
                [self.params["l"], self.params["sigma_f"]],
                bounds=((1e-4, 1e4), (1e-4, 1e4)),
                method="L-BFGS-B",
            )
            self.params["l"], self.params["sigma_f"] = res.x[0], res.x[1]

        self.is_fit = True

    def predict(self, X):
        if not self.is_fit:
            raise RuntimeError("GPR 模型尚未拟合。")

        X = np.asarray(X)
        Kff = self.kernel(self.train_X, self.train_X)
        Kyy = self.kernel(X, X)
        Kfy = self.kernel(self.train_X, X)
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


def main(argv: list[str] | None = None) -> None:
    use_headless_backend()
    parser = argparse.ArgumentParser(description="GPR 超参数优化示例")
    add_common_args(parser, epochs=False)
    parser.add_argument("--no-optimize", action="store_true", help="关闭超参数优化，使用默认参数")
    args = parser.parse_args(argv)

    set_seed(args.seed)
    output_dir = resolve_output_dir(args.output_dir, "gpr")

    train_X = np.array([3, 1, 4, 5, 9]).reshape(-1, 1)
    train_y = target(train_X, noise_sigma=1e-4)
    test_X = np.arange(0, 10, 0.1).reshape(-1, 1)

    gpr = GPR(optimize=not args.no_optimize)
    gpr.fit(train_X, train_y)
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
    save_figure(fig, output_dir / "gpr_hyperparameters.png")
    print(f"Optimized l={gpr.params['l']:.4f}, sigma_f={gpr.params['sigma_f']:.4f}")
    print(f"Saved figure to {output_dir}")


if __name__ == "__main__":
    main()
