"""用于碰撞响应预测的物理信息神经算子（Physics-Informed Neural Operator, PINO）。

实现 PINO 框架，参考：
  Li Z et al. (2021)，综述论文第 2.3 节，图 6

PINO 流程：
  离线阶段：历史碰撞数据 -> 训练 DeepONet
  在线阶段：新参数 -> DeepONet -> x0(t) [约 1 毫秒]
           物理残差 -> 微调 -> x(t) [高精度]
"""

from __future__ import annotations

import torch

from examples.sciml_crash.operator.deeponet_model import DeepONet, ParameterNormalizer


def compute_physics_residual(
    model, normalizer, params_raw, t_grid_raw, device="cpu", normalize=True
):
    """使用无量纲归一化计算 ODE 残差。

    关键说明：DeepONet 内部使用归一化后的参数和时间，
    但 ODE 系数（m, c, k1, k2, xy）必须是原始值。

    ODE 残差 m*a + c*v + F_spring 的单位为牛顿（N）。
    通过特征惯性力 F_char = m * |v0| / T_max 进行归一化，
    使物理项与数据项都达到 O(1) 量级。

    参数：
        model: 使用归一化输入的 DeepONet
        normalizer: 参数归一化器
        params_raw: (batch, 6) 原始碰撞参数 [m, c, k1, k2, xy, v0]
        t_grid_raw: (n_time,) 以秒为单位的原始时间网格
        device: torch 设备
        normalize: 如果为 True，返回无量纲残差（用于 PINO 训练）

    返回：
        residual: (batch, n_time) 物理残差（normalize 时为无量纲）
        x, v, a: 物理单位下的预测值
    """
    batch_size = params_raw.shape[0]
    n_time = len(t_grid_raw)
    T_max = t_grid_raw[-1]

    # 为 DeepONet 归一化输入
    params_norm = normalizer.normalize_tensor(params_raw)
    t_norm = t_grid_raw / T_max
    t_norm_t = t_norm.reshape(-1, 1)

    # 扩展用于批量预测
    params_exp = params_norm.unsqueeze(1).expand(-1, n_time, -1)
    t_exp = t_norm_t.unsqueeze(0).expand(batch_size, -1, -1)

    params_flat = params_exp.reshape(-1, params_raw.shape[1])
    t_flat = t_exp.reshape(-1, 1).requires_grad_(True)

    # 前向传播
    x_flat = model(params_flat, t_flat)
    x = x_flat.reshape(batch_size, n_time)

    # 相对于归一化时间的速度与加速度
    v_norm = torch.autograd.grad(
        x_flat, t_flat, grad_outputs=torch.ones_like(x_flat), create_graph=True, retain_graph=True
    )[0].reshape(batch_size, n_time)

    v_norm_flat = v_norm.reshape(-1, 1)
    a_norm_flat = torch.autograd.grad(
        v_norm_flat, t_flat, grad_outputs=torch.ones_like(v_norm_flat),
        create_graph=True, retain_graph=True,
    )[0]
    a_norm = a_norm_flat.reshape(batch_size, n_time)

    # 缩放到真实时间导数
    v = v_norm / T_max              # dx/dt = (dx/dt_norm) / T_max
    a = a_norm / (T_max**2)         # d^2x/dt^2 = (d^2x/dt_norm^2) / T_max^2

    # 提取原始物理参数
    m = params_raw[:, 0:1]
    c = params_raw[:, 1:2]
    k1 = params_raw[:, 2:3]
    k2 = params_raw[:, 3:4]
    xy = params_raw[:, 4:5]

    # 双线性弹簧力（使用原始刚度值）
    F_spring = torch.where(
        torch.abs(x) <= xy,
        k1 * x,
        (k1 * xy + k2 * (torch.abs(x) - xy)) * torch.sign(x),
    )

    # 外力 = 0；残差：m*a + c*v + F_spring - F_ext [牛顿]
    residual = m * a + c * v + F_spring

    if normalize:
        # 使用特征惯性力归一化为无量纲 O(1)
        v0_abs = torch.abs(params_raw[:, 5:6])
        F_char = m * v0_abs / T_max
        residual = residual / (F_char + 1e-8)

    return residual, x, v, a


class PINOFineTuner:
    """使用组合的物理损失和数据保真度损失微调 DeepONet。

    组合损失：L = L_physics + lambda_data * L_data
      其中 L_data = MSE(x_current, x_deeponet_initial)

    这种方法将解“锚定”在数据驱动预测附近，
    同时允许物理信息对其进行细化以获得一致性。

    策略：
    1. 获取 DeepONet 初始预测 x0(t)
    2. 使用 L = L_physics + lambda_data * L_data 进行微调
    3. 默认仅微调主干网络（参数更少，速度更快）
    """

    def __init__(self, model, normalizer, device="cpu"):
        self.model = model
        self.normalizer = normalizer
        self.device = device

    def fine_tune(
        self,
        params_raw_np,
        t_grid_raw,
        n_iterations=200,
        lr=5e-5,
        lambda_data=5.0,
        freeze_branch=True,
        verbose=True,
    ):
        """使用组合的物理损失和数据保真度损失进行微调。

        物理残差通过 m*|v0|/T_max 归一化，使物理损失和数据损失均为 O(1) 量级。
        lambda_data 越大，解越靠近 DeepONet 的初始预测。

        参数：
            params_raw_np: (6,) 原始碰撞参数
            t_grid_raw: (n_time,) 以秒为单位的时间网格
            n_iterations: 微调迭代次数
            lr: 学习率
            lambda_data: 数据保真度项的权重
            freeze_branch: 是否冻结分支网络

        返回：
            x_finetuned, loss_history 字典
        """
        self.model.train()

        if freeze_branch:
            for p in self.model.branch.parameters():
                p.requires_grad = False
            for p in self.model.trunk.parameters():
                p.requires_grad = True
            self.model.bias.requires_grad = True

        params_raw = torch.tensor(
            params_raw_np.reshape(1, -1).astype("float32"), device=self.device
        )
        T_max = float(t_grid_raw[-1])
        t_np = t_grid_raw.astype("float32")
        t_tensor = torch.tensor(t_np, device=self.device)

        # 获取 DeepONet 初始预测作为锚定
        with torch.no_grad():
            params_norm = self.normalizer.normalize_tensor(params_raw)
            t_norm_t = torch.tensor((t_np / T_max).reshape(-1, 1), device=self.device)
            x_anchor = self.model.predict_trajectory(params_norm, t_norm_t).detach()

        optimizer = torch.optim.Adam(
            filter(lambda p: p.requires_grad, self.model.parameters()), lr=lr
        )

        loss_history = {"physics": [], "data": [], "total": []}
        for it in range(n_iterations):
            optimizer.zero_grad()

            # 物理损失（无量纲，对于好的解为 O(1)）
            residual, x_pred, _, _ = compute_physics_residual(
                self.model, self.normalizer, params_raw, t_tensor, self.device, normalize=True
            )
            loss_physics = torch.mean(residual**2)

            # 数据保真度：保持接近 DeepONet 初始值（x 以米为单位）
            loss_data = torch.mean((x_pred - x_anchor) ** 2)

            loss = loss_physics + lambda_data * loss_data
            loss.backward()
            optimizer.step()

            loss_history["physics"].append(loss_physics.item())
            loss_history["data"].append(loss_data.item())
            loss_history["total"].append(loss.item())
            if verbose and it % max(1, n_iterations // 4) == 0:
                print(
                    f"  Iter {it:4d} | Phys: {loss_physics.item():.4f} | "
                    f"Data: {loss_data.item():.2e} | Total: {loss.item():.4f}"
                )

        self.model.eval()

        # 预测最终轨迹
        with torch.no_grad():
            x_final = self.model.predict_trajectory(params_norm, t_norm_t)
        return x_final.cpu().numpy().flatten(), loss_history

    def reset_branch(self):
        """恢复分支网络的可训练状态。"""
        for p in self.model.branch.parameters():
            p.requires_grad = True
