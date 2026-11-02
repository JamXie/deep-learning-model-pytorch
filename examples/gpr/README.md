# 高斯过程回归（GPR）

在 `y = cos(x)` 上演示高斯过程回归的后验均值与 95% 置信带。三个脚本共用平方指数核：

```text
k(x, x') = sigma_f^2 * exp(-(x - x')^2 / (2 * l^2))
```

## 文件

| 文件 | 内容 |
| --- | --- |
| `gpr_basic.py` | 固定超参数，展示后验均值与协方差 |
| `gpr_hyperparameters.py` | 用 SciPy 优化对数边际似然，学习 `l` 与 `sigma_f` |
| `gpr_uncertain_inputs.py` | 训练协方差中加入逐样本观测噪声方差 |

## 运行

```bash
python -m examples.gpr.gpr_basic
python -m examples.gpr.gpr_hyperparameters
python -m examples.gpr.gpr_hyperparameters --no-optimize   # 使用默认超参数
python -m examples.gpr.gpr_uncertain_inputs
```

## 输出

默认写入 `outputs/gpr/`：

- `gpr_basic.png`
- `gpr_hyperparameters.png`
- `gpr_uncertain_inputs.png`

## 说明

- 三个类都提供 `fit(train_x, train_y)` 与 `predict(test_x)`；`predict` 返回
  `(mu, cov)`，其中 `mu` 的形状为 `(n_test,)`，`cov` 为 `(n_test, n_test)`。
- 为使示例保持小巧，协方差求逆直接使用 `np.linalg.inv`；真实应用中应改用 Cholesky
  分解或线性方程求解。
- 固定 `--seed` 只影响绘图数据与初始化，不改变解析核函数。
