# 碰撞安全科学机器学习（PINN → DeepONet → PINO）

以弹簧-质量系统为对象，演示三条技术路线：

1. **PINNs**：直接求解 `m*x'' + c*x' + F_spring(x) = F(t)`，分为强形式与能量形式；
2. **算子学习**：用 DeepONet 学习 `(m, c, k1, k2, x_yield, v0) -> x(t)`；
3. **PINO**：在 DeepONet 预测上加入物理残差微调，提升一致性。

系统示意：弹簧力采用双线性弹塑性模型，初速度 `v0 < 0` 表示碰撞；外力取 0，
由初始动能驱动衰减振动。

## 目录

```text
pinns/
  models.py               FNN、DerivativeNetwork、HardConstraintWrapper
  losses.py               强形式损失、能量形式损失、NTK 自适应权重
  utils.py                弹簧模型、RK45 参考解、配点采样、误差指标
  pinn_strong_form.py     强形式 PINN 求解器
  pinn_energy_form.py     能量形式 PINN 求解器
  case_linear.py          线性弹簧案例
  case_bilinear.py        双线性弹塑性弹簧案例
  case_rate_dependent.py  率相关弹簧案例

operator/
  deeponet_model.py       BranchNet / TrunkNet / DeepONet / ParameterNormalizer
  data_generator.py       生成碰撞场景数据集（RK45）
  train_operator.py       训练算子网络
  eval_operator.py        精度与速度评估

pino/
  pino_pipeline.py        物理残差计算与微调器
  benchmark.py            DeepONet 与 PINO 对比
```

## 推荐运行顺序

```bash
# 1. PINN 单案例
python -m examples.sciml_crash.pinns.case_linear --iters 5000
python -m examples.sciml_crash.pinns.case_bilinear --iters 8000
python -m examples.sciml_crash.pinns.case_rate_dependent --iters 6000

# 2. 生成算子数据集并训练
python -m examples.sciml_crash.operator.data_generator --num-samples 500
python -m examples.sciml_crash.operator.train_operator --epochs 300

# 3. 评估与 PINO 微调
python -m examples.sciml_crash.operator.eval_operator
python -m examples.sciml_crash.pino.benchmark --iters 200
```

所有脚本都支持 `--output-dir`、`--seed`、`--device` 等公共参数；评估脚本支持
`--checkpoint` 指定权重。

## 默认参数

| 案例 | 质量 m | 阻尼 c | 刚度 | 初速度 v0 | 迭代 |
| --- | --- | --- | --- | --- | --- |
| 线性 | 1200 kg | 500 N·s/m | k=200 kN/m | −15 m/s | 5000 |
| 双线性 | 1200 kg | 500 N·s/m | k1=300, k2=30 kN/m, x_yield=40 mm | −15 m/s | 8000 |
| 率相关 | 1200 kg | 500 N·s/m | k0=200 kN/m, `beta`、`n` 可调 | −15 m/s | 6000 |

时间区间默认 `[0, 0.12] s`。RK45 解作为参考。

## 输出

| 命令 | 输出（默认在 `outputs/sciml_crash/`） |
| --- | --- |
| 三个 PINN 案例 | `pinns/case_*_results.png` |
| 数据生成 | `crash_dataset.pkl` |
| 算子训练 | `deeponet_crash.pt` |
| 评估 | `deeponet_predictions.png` |
| PINO 对比 | `pino_benchmark_results.png` |

## 说明

- 数据划分默认 80/10/10；小样本时会保证训练/验证集非空。
- PINN 的强形式与能量形式共享 `utils.py`，但训练流程独立。
- 修复记录（能量形式二阶导、`predict` 输入类型、空验证集、checkpoint 兼容性等）
  见 `docs/FIXED_ISSUES.md` 第 6 节。
- 冒烟运行请用 `--iters 20`、`--num-samples 8 --num-time 50` 等小规模参数；
  指标没有参考意义。
