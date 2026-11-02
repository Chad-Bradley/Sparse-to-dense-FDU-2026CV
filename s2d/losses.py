"""带无效像素掩码的回归损失。

公式见 FORMULAS.md 第 1 节。核心约定：

    mask = target > 0
    loss = mean( |pred - target| [mask] )     # L1
    loss = mean( (pred - target)^2 [mask] )   # MSE

深度 0 表示该像素没有观测（Kinect 没测到 / 旋转补边），
不能当成「真实深度是 0 米」参与监督，否则网络会学会往空洞里输出 0。
"""

from __future__ import annotations

import torch
import torch.nn as nn


def _masked_mean(diff, mask):
    """对 mask 选中的元素求均值。

    注意三个细节：

    1. 分母是**有效像素数**，不是全图像素数，也不是 batch 大小。
       这样每个有效像素权重相同，不受图片空洞比例影响。
    2. 用布尔索引 diff[mask] 得到一维张量，再 .mean()。
    3. 有效像素为 0 时返回 `diff.sum() * 0` 而不是 Python 的 0.0：
       前者仍然连着计算图，backward 不会报 "does not require grad"；
       后者会让训练在整个 batch 都无有效深度时崩溃。
    """
    if not mask.any():
        return diff.sum() * 0.0
    return diff[mask].mean()


class MaskedL1Loss(nn.Module):
    """只在 target > 0 的像素上计算 L1。"""

    def forward(self, pred, target):
        assert pred.shape == target.shape, \
            "pred {} 与 target {} 形状不一致".format(pred.shape, target.shape)
        # target 不需要梯度，mask 是纯布尔张量
        mask = target > 0
        diff = (pred - target).abs()
        return _masked_mean(diff, mask)


class MaskedMSELoss(nn.Module):
    """只在 target > 0 的像素上计算 MSE。"""

    def forward(self, pred, target):
        assert pred.shape == target.shape, \
            "pred {} 与 target {} 形状不一致".format(pred.shape, target.shape)
        mask = target > 0
        diff = (pred - target) ** 2
        return _masked_mean(diff, mask)


def build_criterion(name):
    """按名字构建损失，未知名字直接报错，避免训练时静默用错损失。"""
    name = name.lower()
    if name == "l1":
        return MaskedL1Loss()
    if name in ("l2", "mse"):
        return MaskedMSELoss()
    raise ValueError("未知 criterion: {}（可选 l1 / l2）".format(name))
