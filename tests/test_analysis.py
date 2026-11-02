"""边界 / 无纹理分析工具的单测（不需要 GPU / 真实数据）。"""

from __future__ import annotations

import unittest

import numpy as np

from s2d.analysis import (
    gradient_magnitude,
    local_variance,
    quantile_edges,
    rgb_to_gray,
    stratified_metrics,
    summarize_bins,
)


class TestScores(unittest.TestCase):
    def test_gradient_zero_on_constant(self):
        depth = np.ones((10, 10), dtype=np.float32)
        g = gradient_magnitude(depth)
        self.assertEqual(g.shape, (10, 10))
        self.assertTrue(np.allclose(g, 0.0))

    def test_gradient_positive_at_step_edge(self):
        depth = np.zeros((10, 10), dtype=np.float32)
        depth[:, 5:] = 1.0                 # 第 5 列是垂直边界
        g = gradient_magnitude(depth)
        self.assertGreater(float(g[5, 4]), 0.0)
        self.assertAlmostEqual(float(g[5, 0]), 0.0, places=6)

    def test_local_variance_constant_is_zero(self):
        gray = np.full((20, 20), 0.5, dtype=np.float32)
        v = local_variance(gray)
        self.assertTrue(np.allclose(v, 0.0, atol=1e-6))

    def test_local_variance_texture_is_positive(self):
        rng = np.random.RandomState(0)
        gray = rng.rand(20, 20).astype(np.float32)
        v = local_variance(gray)
        self.assertGreater(float(v.mean()), 0.0)

    def test_rgb_to_gray_red(self):
        rgb = np.zeros((2, 2, 3), dtype=np.float32)
        rgb[..., 0] = 1.0
        self.assertAlmostEqual(float(rgb_to_gray(rgb).mean()), 0.2989, places=4)

    def test_rgb_to_gray_accepts_uint8(self):
        rgb = np.zeros((2, 2, 3), dtype=np.uint8)
        rgb[..., 1] = 255
        self.assertAlmostEqual(float(rgb_to_gray(rgb).mean()), 0.587, places=4)


class TestBinning(unittest.TestCase):
    def test_quantile_edges(self):
        values = np.arange(100, dtype=np.float64)
        edges = quantile_edges(values, bins=4)
        self.assertEqual(len(edges), 5)
        self.assertAlmostEqual(float(edges[0]), 0.0)
        self.assertAlmostEqual(float(edges[-1]), 99.0)

    def test_constant_score_falls_into_one_bin(self):
        score = np.zeros(10, dtype=np.float64)
        rows, edges = summarize_bins(score, score, score, bins=4)
        self.assertEqual(sum(r["n"] for r in rows), 10)
        self.assertEqual(len(rows), len(edges) - 1)

    def test_summarize_bins_counts_and_trend(self):
        score = np.arange(6, dtype=np.float64)
        sq = np.array([0.0, 1.0, 4.0, 9.0, 16.0, 25.0])
        rel = np.ones(6)
        rows, edges = summarize_bins(score, sq, rel, bins=2)
        self.assertEqual(sum(r["n"] for r in rows), 6)
        self.assertEqual(len(rows), len(edges) - 1)
        # 低分桶误差应小于高分桶
        self.assertLess(rows[0]["rmse"], rows[-1]["rmse"])

    def test_stratified_metrics_ignores_invalid(self):
        target = np.ones((4, 4), dtype=np.float32)
        target[0, 0] = 0.0
        pred = target + 1.0
        score = np.ones((4, 4), dtype=np.float32)
        rows, _ = stratified_metrics(pred, target, score, bins=2)
        # 16 个像素里有 1 个无效
        self.assertEqual(sum(r["n"] for r in rows), 15)


if __name__ == "__main__":
    unittest.main()
