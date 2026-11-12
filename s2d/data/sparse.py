"""稀疏深度采样。"""

from __future__ import annotations

import numpy as np


class UniformSampling:
    """从有效深度像素里无放回抽取恰好 num_samples 个点。

    为什么不用上游的概率采样：
        上游对每个有效像素以 p = N / n_valid 独立抛硬币，点数只是**期望**为 N，
        实际每次在 N ± sqrt(N) 附近波动（500 点约 ±22）。实验要求写的是
        「生成包含 500 个有效点的稀疏深度」，所以这里固定为无放回抽样，
        每张图恰好 500 个点（有效像素不足 500 时才取全部）。

    参数：
        num_samples: 采样点数，默认 500。
        max_depth:   只在该深度范围内采样；np.inf 表示不限制。
    """

    name = "uniform"

    def __init__(self, num_samples=500, max_depth=np.inf):
        self.num_samples = int(num_samples)
        self.max_depth = float(max_depth)

    def __repr__(self):
        return "UniformSampling(n={}, max_depth={})".format(
            self.num_samples, self.max_depth)

    def __call__(self, depth):
        """depth: H×W float32。返回 H×W bool 掩码，True 表示保留该像素。"""
        depth = np.asarray(depth)

        # 只用有观测的像素：深度 0 表示该像素无效（Kinect 没测到），
        # 让它们参与采样会把「0 深度」当成有效观测，直接污染输入。
        valid = depth > 0

        # max_depth 是可选的上限过滤；当前用 np.inf（不过滤）。
        if np.isfinite(self.max_depth):
            valid &= depth <= self.max_depth

        idx = np.flatnonzero(valid)          # 有效像素的一维索引
        mask = np.zeros(depth.shape, dtype=bool)
        if idx.size == 0:
            # 极端情况（全图无效）：返回全 False，交给上层决定怎么处理。
            return mask

        k = min(self.num_samples, idx.size)
        # replace=False 保证同一个像素不会重复计数，点数严格为 k。
        # idx 是预先筛过的有效像素，所以选出来的点一定满足 depth > 0。
        chosen = np.random.choice(idx, size=k, replace=False)
        mask.flat[chosen] = True
        return mask
