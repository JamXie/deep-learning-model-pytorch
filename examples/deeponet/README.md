# DeepONet 算子学习

DeepONet 用分支网络编码输入函数、主干网络编码查询坐标，最后做内积：

```text
G(u)(y) = sum_k branch_k(u) * trunk_k(y) + bias
```

本目录同时提供 FNN 基线（把 `u` 与 `y` 直接拼接后回归）。

## 三个任务

| 任务 | 算子 | 查询维度 |
| --- | --- | --- |
| `1d` | `s(y) = ∫_0^y u(x) dx` | 1 |
| `2d` | 受迫单摆 `(u(t), t) -> s(t)` | 1 |
| `3d` | 一维扩散 PDE `(f(x), (x,t)) -> s(x,t)` | 2 |

## 文件

```text
config.py                任务配置（路径、网络结构、查询维度）
dataload.py              Excel -> DataLoader；支持 u_function / f_function
train.py                 训练 DeepONet 或 FNN 基线
validation.py            在 train / val / extra 上评估并绘图
model/deeponet.py        FNN、DeepONet
model/fnn.py             FNNBaseline（m + query_dim 输入）
data_build/common.py     GRF 采样、Excel 保存
data_build/build_1d_integral.py
data_build/build_2d_pendulum.py
data_build/build_3d_diffusion.py
```

## 工作流

```bash
# 1. 生成数据（写入 outputs/deeponet/<task>/）
python -m examples.deeponet.data_build.build_1d_integral --num-train 2000 --num-val 500

# 2. 训练
python -m examples.deeponet.train --task 1d --model deeponet --epochs 200
python -m examples.deeponet.train --task 1d --model fnn --epochs 200

# 3. 验证
python -m examples.deeponet.validation --task 1d --split val
python -m examples.deeponet.validation --task 1d --split extra
```

2D / 3D 只需替换数据构建脚本和 `--task`。

## 输出

默认都在 `outputs/deeponet/<task>/`：

- `train.xlsx`、`val.xlsx`、`extra.xlsx`
- `deeponet_loss.png` / `fnn_loss.png`
- `deeponet.pth` / `fnn.pth`
- `deeponet_prediction_<split>.png`
- `deeponet_error_distribution_<split>.png`

## 说明

- Excel 的 sheet 约定见 `docs/DATA.md`；`load_from_excel` 同时支持
  `u_function` 与 `f_function`。
- 积分使用 `scipy.integrate.trapezoid`（`np.trapz` 已在 NumPy 2.x 中移除）。
- 原工程附带的 DeepXDE 源码树未收录，也不重新分发。
