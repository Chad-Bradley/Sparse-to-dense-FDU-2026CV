"""带无效像素掩码的回归损失。"""

"""说明"""
"""新增了 BerHu、梯度、法向、对数、尺度不变损失，并扩展了 build_criterion。"""
"""对应地，需要在scripts/train.py中把 
p.add_argument("--criterion", default=d.criterion, choices=["l1", "l2", "mse"])
这一行改为
p.add_argument("--criterion", default=d.criterion, 
               choices=["l1", "l2", "mse", "berhu", "gradient", "normal", "log", "scale_inv"])"""


from __future__ import annotations
import torch
import torch.nn as nn
import torch.nn.functional as F

def _masked_mean(diff, mask):
    if not mask.any():
        return diff.sum() * 0.0
    return diff[mask].mean()

class MaskedL1Loss(nn.Module):
    def forward(self, pred, target):
        assert pred.shape == target.shape
        mask = target > 0
        diff = (pred - target).abs()
        return _masked_mean(diff, mask)

class MaskedMSELoss(nn.Module):
    def forward(self, pred, target):
        assert pred.shape == target.shape
        mask = target > 0
        diff = (pred - target) ** 2
        return _masked_mean(diff, mask)

class MaskedBerHuLoss(nn.Module):
    """BerHu Loss (Reverse Huber) 参考论文 Eq.(2)"""
    def forward(self, pred, target):
        assert pred.shape == target.shape
        mask = target > 0
        if not mask.any():
            return pred.sum() * 0.0
        
        diff = (pred - target).abs()
        # 计算 batch 内的最大误差，作为阈值 c
        c = 0.2 * diff[mask].max().detach()
        
        # 避免 c=0 导致除零错误
        c = torch.clamp(c, min=1e-6)
        
        # BerHu 逻辑
        berhu = torch.where(diff <= c, diff, (diff**2 + c**2) / (2 * c))
        return _masked_mean(berhu, mask)

class MaskedGradientLoss(nn.Module):
    """梯度损失：约束深度图的局部平滑性"""
    def forward(self, pred, target):
        mask = target > 0
        if not mask.any():
            return pred.sum() * 0.0

        # 计算 X 和 Y 方向的梯度
        pred_dx = torch.abs(pred[:, :, :, :-1] - pred[:, :, :, 1:])
        pred_dy = torch.abs(pred[:, :, :-1, :] - pred[:, :, 1:, :])
        
        target_dx = torch.abs(target[:, :, :, :-1] - target[:, :, :, 1:])
        target_dy = torch.abs(target[:, :, :-1, :] - target[:, :, 1:, :])
        
        # 对应的 mask 也要缩小尺寸
        mask_dx = mask[:, :, :, :-1] & mask[:, :, :, 1:]
        mask_dy = mask[:, :, :-1, :] & mask[:, :, 1:, :]

        loss_dx = _masked_mean((pred_dx - target_dx).abs(), mask_dx)
        loss_dy = _masked_mean((pred_dy - target_dy).abs(), mask_dy)
        
        return loss_dx + loss_dy

class MaskedNormalLoss(nn.Module):
    """法向损失：约束表面的朝向"""
    def forward(self, pred, target):
        mask = target > 0
        if not mask.any():
            return pred.sum() * 0.0

        # 计算梯度 (未取绝对值)
        pred_dx = pred[:, :, :, :-1] - pred[:, :, :, 1:]
        pred_dy = pred[:, :, :-1, :] - pred[:, :, 1:, :]
        
        target_dx = target[:, :, :, :-1] - target[:, :, :, 1:]
        target_dy = target[:, :, :-1, :] - target[:, :, 1:, :]
        
        # 构建法向量 (N, C, H, W, 3) -> [-dx, -dy, 1]
        # 注意：这里简化了处理，取内部的 H-1, W-1 区域
        pred_norm = F.normalize(torch.stack([-pred_dx[:,:,:,:-1], -pred_dy[:,:,:-1,:], torch.ones_like(pred_dx[:,:,:,:-1])], dim=-1), p=2, dim=-1)
        target_norm = F.normalize(torch.stack([-target_dx[:,:,:,:-1], -target_dy[:,:,:-1,:], torch.ones_like(target_dx[:,:,:,:-1])], dim=-1), p=2, dim=-1)
        
        # 计算余弦相似度
        cos_sim = (pred_norm * target_norm).sum(dim=-1)
        loss = 1 - cos_sim
        
        mask_inner = mask[:, :, :-1, :-1]
        return _masked_mean(loss, mask_inner)

class MaskedLogLoss(nn.Module):
    """对数深度损失"""
    def forward(self, pred, target):
        mask = target > 0
        if not mask.any():
            return pred.sum() * 0.0
        # 加 eps 避免 log(0)
        eps = 1e-6
        diff = torch.abs(torch.log(pred + eps) - torch.log(target + eps))
        return _masked_mean(diff, mask)

class MaskedScaleInvariantLoss(nn.Module):
    """尺度不变损失 (Scale-Invariant)"""
    def forward(self, pred, target):
        mask = target > 0
        if not mask.any():
            return pred.sum() * 0.0
        eps = 1e-6
        # 只对有效像素计算 log 差值
        d = torch.log(pred[mask] + eps) - torch.log(target[mask] + eps)
        n = d.numel()
        loss = (d ** 2).mean() - 0.5 * (d.mean() ** 2)
        return loss

def build_criterion(name):
    name = name.lower()
    if name == "l1":
        return MaskedL1Loss()
    if name in ("l2", "mse"):
        return MaskedMSELoss()
    if name == "berhu":
        return MaskedBerHuLoss()
    if name == "gradient":
        return MaskedGradientLoss()
    if name == "normal":
        return MaskedNormalLoss()
    if name == "log":
        return MaskedLogLoss()
    if name == "scale_inv":
        return MaskedScaleInvariantLoss()
    raise ValueError(f"未知 criterion: {name}")