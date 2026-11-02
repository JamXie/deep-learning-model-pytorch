# 整理过程中修复的问题

本仓库来自作者早期的算例工作区（`ProjectOfPython3810/DeepLearningModel`）。整理时
遵循两条原则：

1. **只收录能运行的示例**，修复会阻塞运行的 bug，不改变模型的算法意图；
2. **不重新训练、不追求复现旧图**，原工作区只读，不修改任何源文件。

下表记录主要问题与处理方式。原工程中的权重、LS-DYNA 输出、生成的
xlsx/csv/pptx/png/gif、DeepXDE 残缺副本等均未纳入本仓库。

## 1. 表格回归（ANN / GRNN / RBF / 插值）

| 位置（原工程） | 问题 | 处理 |
| --- | --- | --- |
| `ANN/ANN.py` | 验证阶段只把模型搬到 `device`，验证张量仍在 CPU，CUDA 下会报设备不一致 | 训练与验证张量都显式 `.to(device)`，预测结果再 `.cpu()` |
| `ANN/ANN.py` | 小数据集时 `drop_last=True` 可能产生空 batch | `drop_last` 仅在样本数不少于 batch size 时启用 |
| `RBF/RBFNet.py` | KMeans 在 CPU 上求中心，而网络在 GPU 上 | `initialize_centers(..., device=...)` 显式把中心放到目标设备 |
| `RBF/RBFNet.py` | 验证阶段张量未随模型迁移 | 同 ANN |
| `FP(griddata).py` | 查询点落在凸包外时 `griddata` 返回 NaN，导致 R² 为 NaN | 计算指标前 `np.nan_to_num`，并在文档中说明 nearest 基线的含义 |
| `FP(RBF).py` | 脚本级代码，无法复用 | 包进 `main()` + argparse，导入无副作用 |
| 数据读取 | 原工程散落在多处、依赖本机路径与硬编码编码 | 统一到 `load_crash_data.py`：编码回退、固定列、`random_state=42` |

## 2. 高斯过程回归

| 位置（原工程） | 问题 | 处理 |
| --- | --- | --- |
| `GPR/*.py` | 脚本顶层直接绘图、硬编码输出路径 | 三个脚本均改为 `main()` + 公共 CLI；固定种子；图片写入 `outputs/gpr/` |
| `GPR/*.py` | 使用本机 matplotlib 后端与中文字体设置 | 统一 `Agg` 后端，标题与坐标轴使用英文，避免字体缺失警告 |

算法本身（平方指数核、超参数优化、观测噪声方差）保持不变。

## 3. 生成模型（扩散 / PixelCNN / VQ-VAE）

| 位置（原工程） | 问题 | 处理 |
| --- | --- | --- |
| `DiffusionModel/DiffusionModel.py` | 导入了未使用的 `diffusers`；使用 Qt 后端；导出 PPTX | 删除无用依赖与 PPTX 代码；首次运行 `download=True`；新增采样网格图 |
| `PixcelCNN/PixcelCNN.py` | 数据只做训练/测试二分，验证集缺失或不合理 | 改为 80/10/10 的训练/验证/测试划分 |
| `PixcelCNN/PixcelCNN.py` | 输出层 256 通道，但损失是 MSE，通道数不匹配 | 输出改为 1 通道，与重建目标一致 |
| `PixcelCNN/PixcelCNN.py` | 掩码通过原地修改 `weight` 实现，可能破坏参数与自动求导 | 每次前向使用 `self.weight * self.mask`，不修改原参数 |
| `PixcelCNN/PixcelCNN.py` | 文档未说明模型其实是无条件重建式 PixelCNN | README 中明确其局限：不是逐像素自回归采样，也没有标签条件 |
| `VQ-VAE/VQ-VAE.py` | 验证损失累加后除以训练循环变量 `i`，而非验证计数 `v_i` | 使用 `v_i + 1` 作为验证批次数；删除临时文件与 PPTX |

## 4. 时序 Transformer（重写）

| 位置（原工程） | 问题 | 处理 |
| --- | --- | --- |
| `Transformer/TemporalTransformer.py` | 直接把 64×64=4096 维展平结果当作 `d_model`，注意力层参数与显存开销过大 | 增加 `Linear(64*64, d_model)` 帧编码器，默认 `d_model=128` |
| 同上 | 训练与推理的输入构造不一致，推理路径不可靠 | 训练用 `[SOS, target[:-1]]` 做 teacher forcing；推理用 5 帧历史逐帧自回归 |
| 同上 | 权重加载未指定 `map_location`，跨设备加载失败 | `torch.load(..., map_location=device)` |
| `Transformer/GPU/` | 训练循环被注释、存在维度错误、附带 3.8 GB 权重 | 不纳入本仓库 |

这是本仓库唯一“重写”而非“修补”的示例。重写目标是得到一个**可运行、结构合理**的
教学版本；它不与原脚本逐行对应，也不声称复现原文结果。

## 5. DeepONet

| 位置（原工程） | 问题 | 处理 |
| --- | --- | --- |
| `DeepONets/deeponet/*` | 使用 `np.trapz`，在 NumPy 2.x 中已移除 | 改用 `scipy.integrate.trapezoid` |
| `DeepONets/deeponet/validation.py` | 验证集误用训练数据路径 | 使用 `config.test_path`，并支持 `--split {train,val,extra}` |
| 训练/验证/评估脚本 | `torch.load` 未指定 `map_location` | 统一传入目标设备 |
| `DeepONets/fnn/model/fnn.py` | 基线 FNN 的输入维度写死，无法支持 2D 查询 | 输入维度按 `m + query_dim` 计算，1D/2D 通用 |
| `data_build_4.*` 等 | 三个数据构建脚本重复实现保存逻辑，随机性来源不明确 | 抽出 `data_build/common.py`，使用显式 `RandomState`，输出到 `outputs/deeponet/<task>/` |
| 原工程内嵌的 DeepXDE | 残缺副本、LGPL-2.1 | **删除且不再分发**；NOTICE 中说明来源 |

## 6. 碰撞安全 SciML（PINN / 算子 / PINO）

| 位置（原工程） | 问题 | 处理 |
| --- | --- | --- |
| `pinns_crash/*`、`operator_crash/*`、`pino_crash/*` | 通过 `sys.path` 注入原目录，无法作为包运行 | 改为 `examples.sciml_crash` 包内相对导入，用 `python -m` 运行 |
| 各脚本 | 直接 `plt.show()`，无界面环境会阻塞 | 改用 `common.plotting.save_figure` 保存 PNG |
| `pinn_energy_form.py` | `predict` 中一阶导数用 `create_graph=False`，之后无法再求二阶导 | 一阶导改为 `create_graph=True`，二阶导再 `create_graph=False` |
| 两个 PINN solver 的 `predict` | 只接受 NumPy 数组，传入 Python list 时报 `'list' object has no attribute 'reshape'` | 同时支持 tensor / ndarray / list |
| 数据划分 | 小样本时 `int(8*0.1)=0`，验证集为空，验证 RMSE 为 NaN | `split_dataset` 在比率 > 0 时保证至少 1 条训练/验证样本，并限制验证集不超过剩余样本 |
| `train_operator.py` | checkpoint 中的归一化参数为 NumPy 数组，torch 2.x 的 `weights_only=True` 默认值下加载失败 | 保存为 Python list，加载时构造张量 |
| PINN solver | 未显式支持设备，CUDA 不可用 | 增加 `device` 参数并迁移模型/配点 |
| 控制台输出 | 使用 en dash（`–`）在 GBK 控制台显示为乱码 | 改为 ASCII 连字符 |
| `PINN/PINN`、`#SyncVersion/{PINN,MLP,PINN_MPDB}` | 重复目录、断链、混入 LS-DYNA 结果 | 不纳入本仓库 |

## 7. 路径、后端与编码

| 问题 | 处理 |
| --- | --- |
| 硬编码的本地绝对路径（Windows 盘符） | 统一经 `common.paths` 解析；支持 `DLM_DATA_ROOT` / `DLM_OUTPUT_ROOT` 与 `--data-root` / `--output-dir` |
| `matplotlib.use("Qt5Agg")` | 统一 `Agg`，测试在 `conftest.py` 中提前设置 |
| `SimHei` 中文标题导致缺字警告 | 图内文字改为英文 |
| `KMP_DUPLICATE_LIB_OK=TRUE` 规避 OpenMP 冲突 | 删除该 hack；依赖版本在 `requirements.txt` 中固定下限 |
| 到处导出 PPTX | 移除 `python-pptx` 依赖与所有导出代码 |
| 输出文件散落、覆盖源目录 | 所有产物写入 `outputs/`（或显式 `--output-dir`），`.gitignore` 排除 |

## 8. 关于 `docs/assets/` 的图片

三张图片（`pinns_bilinear.png`、`deeponet_predictions.png`、
`pino_benchmark.png`）来自作者早期使用同一批模型得到的实验结果，经缩放后用于
展示图表形态。它们**不是**当前代码的精确复现输出，仅作示意；当前代码的冒烟输出
请见 `docs/REPRODUCIBILITY.md` 中的命令。
