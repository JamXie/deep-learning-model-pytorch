# 数据说明

本仓库只提交一个数据文件；其余数据由脚本生成或自动下载，不进入版本库。

## 1. `crashworthiness/result_1ct_oblique_load.csv`

1CT 矩形薄壁梁斜向压溃的有限元仿真结果，**600 行 × 11 列**，原始编码为
GB2312。读取函数位于 `examples/tabular_regression/load_crash_data.py`，会依次
尝试 `gb2312 → gb18030 → utf-8`。

| 列名 | 角色 | 含义 |
| --- | --- | --- |
| `Unnamed: 0` | 忽略 | 导出时附带的序号列 |
| `l` | 输入 | 长度方向尺寸 |
| `w` | 输入 | 宽度方向尺寸 |
| `h` | 输入 | 高度方向尺寸 |
| `t` | 输入 | 壁厚 |
| `LoadAngle` | 输入 | 加载角度 |
| `sigma` | 输入 | 材料屈服强度 |
| `label` | 忽略 | 原始分类标签，本仓库的回归示例不使用 |
| `MeanCrashingForce` | 输出 | 平均压溃力 |
| `PeakCrashingForce` | 输出 | 峰值压溃力 |
| `SpecificEnergyAbsorption` | 输出 | 比吸能 |

数值单位沿用原始仿真导出，本仓库不做单位换算。表格回归示例按
`load_crash_data.py` 中固定的 `INPUT_RANGES` / `OUTPUT_RANGES` 做 Min-Max
归一化，并用 `random_state=42` 按 80/20 划分训练集与验证集。数据来源与许可见根目录
`NOTICE`。

## 2. MNIST（自动下载，不入库）

`examples/mnist_generative/*` 通过 `torchvision.datasets.MNIST(download=True)`
把数据下载到 `data/mnist/`。该目录已被 `.gitignore` 排除。图像张量取值
`[0, 1]`，生成模型在 `[-1, 1]` 与 `[0, 1]` 之间按脚本约定转换（见各脚本注释）。

## 3. Moving MNIST（由脚本生成，不入库）

```bash
python -m examples.temporal_transformer.generate_moving_mnist --num-samples 600
```

输出默认路径：`data/moving_mnist/moving_mnist_sequences.pt`。固定 schema：

```python
{
    "sequences": FloatTensor[N, L, 1, 64, 64],   # 取值 [0, 1]
    "conditions": [                              # 每个序列一条
        {"initial_position": [x0, y0], "velocity": [vx, vy]},
        ...
    ],
    "meta": {
        "num_samples": N,
        "seq_len": L,
        "canvas_size": 64,
        "image_size": 28,
        "seed": 42,
        "source": "MNIST train",
    },
}
```

数字在 64×64 画布上匀速移动，碰边反弹。生成过程使用显式
`numpy.random.RandomState(seed)`，同一种子可复现。

## 4. DeepONet Excel（由脚本生成，不入库）

```bash
python -m examples.deeponet.data_build.build_1d_integral
python -m examples.deeponet.data_build.build_2d_pendulum
python -m examples.deeponet.data_build.build_3d_diffusion
```

默认写入 `outputs/deeponet/{1d,2d,3d}/`，包含 `train.xlsx`、`val.xlsx`，
1d/2d 任务另有外推集 `extra.xlsx`。每个工作簿有三个 sheet：

| Sheet | 内容 | 形状 |
| --- | --- | --- |
| `u_function`（1d/2d）或 `f_function`（3d） | 分支函数离散值 | `(N, m)` |
| `y_location` | 查询点；1-D 查询列名为 `y`，多维查询列为 `q0, q1, ...` | `(N, query_dim)` |
| `s_output` | 目标输出 | `(N, 1)` |

`examples/deeponet/dataload.py::load_from_excel` 同时支持
`u_function` 与 `f_function` 两种分支 sheet。

## 5. 碰撞算子数据集（由脚本生成，不入库）

```bash
python -m examples.sciml_crash.operator.data_generator --num-samples 500
```

默认输出 `outputs/sciml_crash/crash_dataset.pkl`，内容为：

```python
{
    "full":  {"params": (N, 6), "param_names": [...], "t_grid": (T,),
              "t_span": (0.0, 0.12), "x_data": (N, T),
              "v_data": (N, T), "a_data": (N, T)},
    "train": {...}, "val": {...}, "test": {...},   # 80/10/10 划分
}
```

参数顺序为 `[m, c, k1, k2, x_yield, v0]`。参数采样范围：

| 参数 | 范围 |
| --- | --- |
| `m` | 800–2000 kg |
| `c` | 300–1000 N·s/m |
| `k1` | 150–500 kN/m |
| `k2` | `k1 × U(0.05, 0.20)` |
| `x_yield` | 0.02–0.06 m |
| `v0` | −20 – −5 m/s |

小样本时会保证训练集与验证集非空（至少 1 条），因此冒烟运行（例如
`--num-samples 8`）也能完成划分。

## 6. 路径与输出

| 环境变量 | 作用 | 默认值 |
| --- | --- | --- |
| `DLM_DATA_ROOT` | 数据根目录 | `<repo>/data` |
| `DLM_OUTPUT_ROOT` | 输出根目录 | `<repo>/outputs` |

命令行可用 `--data-root` / `--output-dir` 覆盖。所有生成文件都在被忽略的目录中。
