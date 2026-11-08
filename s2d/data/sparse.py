"""Uniform sampling interface specified by the baseline tests."""

import numpy as np


class UniformSampling:
    def __init__(self, num_samples=500, max_depth=np.inf):
        if int(num_samples) != num_samples or num_samples < 0:
            raise ValueError("num_samples must be a nonnegative integer")
        self.num_samples = int(num_samples)
        self.max_depth = float(max_depth)

    def __call__(self, depth):
        depth = np.asarray(depth)
        valid = np.isfinite(depth) & (depth > 0) & (depth <= self.max_depth)
        candidates = np.flatnonzero(valid)
        selected = np.zeros(depth.shape, dtype=bool)
        count = min(self.num_samples, candidates.size)
        if count:
            selected.flat[np.random.choice(candidates, count, replace=False)] = True
        return selected
