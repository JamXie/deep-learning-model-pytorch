# 插值基线：RBF 与 griddata

两个非参数方法，用来和 ANN、GRNN、RBF 网络对比。输入为 6 个结构/材料参数，输出为
3 项压溃性能；数据见 [data/README.md](../../data/README.md)。

| 文件 | 方法 |
| --- | --- |
| `rbf.py` | `scipy.interpolate.RBFInterpolator`，multiquadric 核，`epsilon=1.8` |
| `griddata.py` | `scipy.interpolate.griddata`，默认 `nearest` |

## 运行

```bash
python -m examples.interpolation.rbf
python -m examples.interpolation.griddata

# 对比不同 griddata 方法
python -m examples.interpolation.griddata --method linear
```

公共参数见仓库根目录 README。

## 输出

默认写入 `outputs/interpolation/`：

- `interpolation_rbf.png`
- `interpolation_griddata.png`

## 说明

- 6 维散点无法构成规则网格，`griddata` 的 `linear` / `cubic` 在查询点落在凸包外时
  会返回 NaN。脚本会先统计 NaN 数量并 `nan_to_num`，再计算 R²；`nearest` 基线不会
  产生 NaN。
- RBF 插值同样对 NaN 做了清理。
- 与 ANN、GRNN、RBF 网络共用
  [`examples/_shared/crash_data.py`](../_shared/crash_data.py) 的归一化与划分。
