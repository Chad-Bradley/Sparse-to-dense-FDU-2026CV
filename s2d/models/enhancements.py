"""Optional context, U-Net fusion and lightweight CSPN-style propagation.

These modules are deliberately not a reproduction of the full CSPN++ paper.
"""
from __future__ import annotations

import torch
from torch import nn
from torch.nn import functional as F


class MultiScaleContext(nn.Module):
    def __init__(self, channels):
        super().__init__()
        width = channels // 4
        self.branches = nn.ModuleList([
            nn.Sequential(nn.Conv2d(channels, width, 1, bias=False),
                          nn.BatchNorm2d(width), nn.ReLU(inplace=True)),
            *[nn.Sequential(
                nn.Conv2d(channels, width, 3, padding=rate, dilation=rate, bias=False),
                nn.BatchNorm2d(width), nn.ReLU(inplace=True)) for rate in (1, 2, 3)],
        ])
        self.project = nn.Sequential(nn.Conv2d(width * 4, channels, 1, bias=False),
                                     nn.BatchNorm2d(channels))

    def forward(self, x):
        return x + self.project(torch.cat([branch(x) for branch in self.branches], dim=1))


class SkipFusion(nn.Module):
    def __init__(self, skip_channels, channels):
        super().__init__()
        self.project = nn.Conv2d(skip_channels, channels, 1, bias=False)
        self.fuse = nn.Sequential(nn.Conv2d(2 * channels, channels, 3, padding=1, bias=False),
                                  nn.BatchNorm2d(channels), nn.ReLU(inplace=True))

    def forward(self, decoded, skip):
        skip = F.interpolate(self.project(skip), size=decoded.shape[-2:],
                             mode="bilinear", align_corners=True)
        return self.fuse(torch.cat((decoded, skip), dim=1))


class CSPNLite(nn.Module):
    """Signed normalized affinities, fixed coarse center, optional sparse anchors."""
    def __init__(self, feature_channels, kernel_size=3, steps=3, preserve_sparse=True):
        super().__init__()
        if kernel_size < 3 or kernel_size % 2 != 1:
            raise ValueError("prop_kernel must be an odd integer >= 3")
        if steps < 1:
            raise ValueError("prop_steps must be positive")
        self.kernel_size = kernel_size
        self.steps = steps
        self.preserve_sparse = preserve_sparse
        self.feature_project = nn.Conv2d(feature_channels, 16, 1)
        self.guidance = nn.Sequential(nn.Conv2d(19, 32, 3, padding=1), nn.ReLU(),
                                      nn.Conv2d(32, kernel_size ** 2 - 1, 3, padding=1))
        self.reset_affinity()

    def reset_affinity(self):
        # Identity propagation initially (except for explicitly enabled anchors).
        nn.init.zeros_(self.guidance[-1].weight)
        nn.init.zeros_(self.guidance[-1].bias)

    def forward(self, coarse, features, sparse=None):
        if sparse is None:
            sparse = torch.zeros_like(coarse)
        if sparse.shape != coarse.shape:
            raise ValueError("Sparse anchors and output must have identical spatial dimensions")
        mask = torch.isfinite(sparse) & (sparse > 0)
        sparse = torch.where(mask, sparse, torch.zeros_like(sparse))
        features = F.interpolate(self.feature_project(features), size=coarse.shape[-2:],
                                 mode="bilinear", align_corners=True)
        affinity = torch.tanh(self.guidance(torch.cat(
            (features, coarse, sparse, mask.to(coarse.dtype)), dim=1)))
        affinity = affinity / affinity.abs().sum(dim=1, keepdim=True).clamp_min(1.0)
        center = 1 - affinity.sum(dim=1, keepdim=True)
        depth = coarse
        k = self.kernel_size
        middle = k * k // 2
        for _ in range(self.steps):
            patches = F.unfold(F.pad(depth, (k // 2,) * 4, mode="replicate"), k)
            patches = patches.reshape(depth.shape[0], k * k, *depth.shape[-2:])
            neighbors = torch.cat((patches[:, :middle], patches[:, middle + 1:]), dim=1)
            depth = center * coarse + (affinity * neighbors).sum(dim=1, keepdim=True)
            if self.preserve_sparse:
                depth = torch.where(mask, sparse, depth)
        return depth
