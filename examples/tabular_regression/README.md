# 表格回归：压溃性能预测

用 1CT 矩形薄壁梁的 6 个结构/材料参数预测 3 项压溃性能：

- 输入：`l, w, h, t, LoadAngle, sigma`
- 输出：`MeanCrashingForce, PeakCrashingForce, SpecificEnergyAbsorption`

数据为 600 条有限元仿真结果，见仓库根目录 `data/README.md`。所有脚本使用
`load_crash_data.py` 中固定的归一化范围，并按 `random_state=42` 做 80/20 划分，
因此 5 个模型的结果可以直接横向比较。

## 文件

| 文件 | 内容 |
| --- | --- |
| `load_crash_data.py` | CSV 读取（GB2312/GB18030/UTF-8 回退）、归一化、划分 |
| `ann.py` | 三层全连接网络（BatchNorm + LeakyReLU + Dropout） |
| `grnn.py` | 广义回归神经网络；训练样本即核中心，无需反向传播 |
| `rbf_network.py` | RBF 网络；KMeans 初始化高斯核中心，输出层可训练 |
| `interpolation_rbf.py` | `scipy.interpolate.RBFInterpolator`（multiquadric 核） |
| `interpolation_griddata.py` | `scipy.interpolate.griddata`（nearest）基线 |

## 运行

```bash
python -m examples.tabular_regression.ann --epochs 300
python -m examples.tabular_regression.grnn --sigma 0.16
python -m examples.tabular_regression.rbf_network --epochs 500 --hidden-dim 100
python -m examples.tabular_regression.interpolation_rbf
python -m examples.tabular_regression.interpolation_griddata
```

公共参数（`--seed`、`--device`、`--data-root`、`--output-dir`、`--limit-batches`）
见根目录 README。训练类脚本另有 `--epochs`、`--batch-size`。

## 输出

默认写入 `outputs/tabular_regression/`：

- `ann_regression.png`
- `grnn_regression.png`
- `rbf_regression.png`
- `interpolation_rbf.png`
- `interpolation_griddata.png`

每张图包含三个输出的“预测值-真实值”散点图与 `x = y` 参考线。终端同时打印训练/验证
R²。

## 说明

- 归一化使用固定范围而非按数据集重新统计，目的是与原始工程保持一致。
- `griddata` 使用 nearest；其它插值方式在凸包外会产生 NaN，需额外清理。
- 冒烟运行（`--epochs 1 --limit-batches 1`）只为验证流程，指标无参考意义。
