"""Optional depth-only input adapters. Dataset inputs remain zero-filled meters."""

import math

import torch
from torch import nn
from torch.nn import functional as F


class SparseConv2d(nn.Module):
    """Masked convolution normalized by valid spatial count, with mask propagation.

    This is a sparsity-invariant front-end, evaluated with dense PyTorch kernels.
    It does not claim sparse-kernel compute acceleration.
    """

    def __init__(self, in_channels, out_channels, kernel_size=3):
        super().__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size,
                              padding=kernel_size // 2, bias=False)
        self.bias = nn.Parameter(torch.zeros(out_channels))
        self.register_buffer("count_kernel", torch.ones(1, 1, kernel_size, kernel_size))

    def forward(self, features, mask):
        mask = mask.to(features.dtype)
        values = torch.where(mask.bool(), features, torch.zeros_like(features))
        count = F.conv2d(mask, self.count_kernel, padding=self.conv.padding)
        valid = count > 0
        result = self.conv(values) / count.clamp_min(1)
        result = (result + self.bias[None, :, None, None]) * valid
        return result, valid.to(features.dtype)


class InvalidPointAdapter(nn.Module):
    """zero / negative sentinel / explicit mask / masked depth front-end."""

    def __init__(self, in_channels, mode="zero", invalid_value=-1.0):
        super().__init__()
        if mode not in ("zero", "sentinel", "mask", "sparse"):
            raise ValueError("unknown invalid-point mode: " + mode)
        if mode != "zero" and in_channels not in (1, 4):
            raise ValueError("invalid-point modes require raw d(1) or rgbd(4) input")
        if not math.isfinite(invalid_value) or invalid_value >= 0:
            raise ValueError("invalid_value must be finite and negative")
        self.mode = mode
        self.invalid_value = invalid_value
        self.in_channels = in_channels
        self.out_channels = in_channels + (mode in ("mask", "sparse"))
        if mode == "sparse":
            self.sparse1 = SparseConv2d(1, 16)
            self.sparse2 = SparseConv2d(16, 1)

    def forward(self, x):
        if x.shape[1] != self.in_channels:
            raise ValueError("adapter expects raw dataset input channels")
        if self.mode == "zero":
            return x
        depth = x[:, -1:]
        valid = torch.isfinite(depth) & (depth > 0)
        clean = torch.where(valid, depth, torch.zeros_like(depth))
        if self.mode == "sentinel":
            encoded = torch.where(valid, clean, torch.full_like(depth, self.invalid_value))
            return torch.cat((x[:, :-1], encoded), dim=1)
        if self.mode == "sparse":
            features, mask = self.sparse1(clean, valid)
            features, mask = self.sparse2(F.relu(features), mask)
            return torch.cat((x[:, :-1], features, mask), dim=1)
        return torch.cat((x[:, :-1], clean, valid.to(x.dtype)), dim=1)
