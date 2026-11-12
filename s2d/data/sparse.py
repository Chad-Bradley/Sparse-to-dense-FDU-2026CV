"""稀疏深度采样。"""

from __future__ import annotations

import numpy as np


class UniformSampling:
    """严格无放回抽取恰好 num_samples 个点（原有代码保持不变）"""
    name = "uniform"

    def __init__(self, num_samples=500, max_depth=np.inf):
        self.num_samples = int(num_samples)
        self.max_depth = float(max_depth)

    def __repr__(self):
        return "UniformSampling(n={}, max_depth={})".format(self.num_samples, self.max_depth)

    def __call__(self, depth):
        depth = np.asarray(depth)
        valid = depth > 0
        if np.isfinite(self.max_depth):
            valid &= depth <= self.max_depth

        idx = np.flatnonzero(valid)
        mask = np.zeros(depth.shape, dtype=bool)
        if idx.size == 0:
            return mask

        k = min(self.num_samples, idx.size)
        chosen = np.random.choice(idx, size=k, replace=False)
        mask.flat[chosen] = True
        return mask


class DynamicUniformSampling:
    """动态随机数量采样：每次随机取 [min_samples, max_samples] 之间的点数。"""
    name = "dynamic_uniform"

    def __init__(self, min_samples=100, max_samples=1000, max_depth=np.inf):
        self.min_samples = int(min_samples)
        self.max_samples = int(max_samples)
        self.max_depth = float(max_depth)

    def __repr__(self):
        return "DynamicUniformSampling(min={}, max={})".format(self.min_samples, self.max_samples)

    def __call__(self, depth):
        depth = np.asarray(depth)
        valid = depth > 0
        if np.isfinite(self.max_depth):
            valid &= depth <= self.max_depth

        idx = np.flatnonzero(valid)
        mask = np.zeros(depth.shape, dtype=bool)
        if idx.size == 0:
            return mask

        k = np.random.randint(self.min_samples, self.max_samples)
        k = min(k, idx.size)
        
        chosen = np.random.choice(idx, size=k, replace=False)
        mask.flat[chosen] = True
        return mask


class GridSampling:
    """网格采样：每隔 step 个像素采一个点。"""
    name = "grid"

    def __init__(self, step=10):
        self.step = step

    def __repr__(self):
        return "GridSampling(step={})".format(self.step)

    def __call__(self, depth):
        depth = np.asarray(depth)
        mask = np.zeros(depth.shape, dtype=bool)
        mask[::self.step, ::self.step] = depth[::self.step, ::self.step] > 0
        return mask