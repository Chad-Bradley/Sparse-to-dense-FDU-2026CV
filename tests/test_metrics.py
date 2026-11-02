"""指标单元测试（不需要 GPU / 真实数据）。

手算样例：
    target = [[1, 2],
              [0, 4]]
    pred   = [[2, 1],
              [9, 4]]
    有效像素 3 个；diff = [1, -1, 0]

    mse    = (1 + 1 + 0) / 3          = 2/3
    rmse   = sqrt(2/3)                ≈ 0.816497
    mae    = (1 + 1 + 0) / 3          = 2/3
    absrel = (1/1 + 1/2 + 0/4) / 3    = 0.5
    lg10   = (|log10 2 - log10 1| + |log10 1 - log10 2| + 0) / 3
           = 2*log10(2)/3             ≈ 0.200686
    ratios = [2, 2, 1]
    1.25^1 = 1.25  -> 只有 1 个通过 -> delta1 = 1/3
    1.25^2 = 1.5625 -> 1/3
    1.25^3 = 1.953125 -> 2.0 > 1.953 -> 1/3
"""

from __future__ import annotations

import math
import unittest

import torch

from s2d.metrics import MetricAccumulator, compute_metrics


class TestMetrics(unittest.TestCase):
    def setUp(self):
        self.target = torch.tensor([[[[1.0, 2.0],
                                      [0.0, 4.0]]]])
        self.pred = torch.tensor([[[[2.0, 1.0],
                                    [9.0, 4.0]]]])

    def test_hand_computed_values(self):
        m = compute_metrics(self.pred, self.target)

        self.assertEqual(m["n"], 3)
        self.assertAlmostEqual(m["mse"], 2.0 / 3.0, places=6)
        self.assertAlmostEqual(m["rmse"], math.sqrt(2.0 / 3.0), places=6)
        self.assertAlmostEqual(m["mae"], 2.0 / 3.0, places=6)
        self.assertAlmostEqual(m["absrel"], 0.5, places=6)
        self.assertAlmostEqual(m["lg10"], 2.0 * math.log10(2.0) / 3.0,
                               places=6)
        self.assertAlmostEqual(m["delta1"], 1.0 / 3.0, places=6)
        self.assertAlmostEqual(m["delta2"], 1.0 / 3.0, places=6)
        self.assertAlmostEqual(m["delta3"], 1.0 / 3.0, places=6)

    def test_perfect_prediction(self):
        m = compute_metrics(self.target, self.target)
        self.assertAlmostEqual(m["rmse"], 0.0, places=6)
        self.assertAlmostEqual(m["absrel"], 0.0, places=6)
        self.assertAlmostEqual(m["mae"], 0.0, places=6)
        self.assertAlmostEqual(m["delta1"], 1.0, places=6)
        self.assertAlmostEqual(m["delta3"], 1.0, places=6)

    def test_invalid_pixels_ignored(self):
        # 只改 target=0 的位置，所有指标必须不变
        pred2 = self.pred.clone()
        pred2[0, 0, 1, 0] = 1e6
        m1 = compute_metrics(self.pred, self.target)
        m2 = compute_metrics(pred2, self.target)
        for key in ("n", "mse", "rmse", "mae", "absrel", "lg10",
                    "delta1", "delta2", "delta3"):
            self.assertAlmostEqual(m1[key], m2[key], places=6, msg=key)

    def test_empty_target(self):
        m = compute_metrics(self.pred, torch.zeros_like(self.target))
        self.assertEqual(m["n"], 0)
        self.assertTrue(math.isnan(m["rmse"]))
        self.assertTrue(math.isnan(m["absrel"]))

    def test_negative_prediction_is_nan_safe(self):
        m = compute_metrics(torch.full_like(self.pred, -1.0), self.target)
        self.assertFalse(math.isnan(m["lg10"]))
        self.assertFalse(math.isnan(m["delta1"]))
        self.assertFalse(math.isnan(m["rmse"]))

    def test_accumulation_matches_single_call(self):
        """分两次 update 与一次算全部，结果应完全一致（pixel-wise 口径）。"""
        acc = MetricAccumulator()
        acc.update(self.pred[:, :, :1, :], self.target[:, :, :1, :])
        acc.update(self.pred[:, :, 1:, :], self.target[:, :, 1:, :])
        split = acc.compute()

        full = compute_metrics(self.pred, self.target)
        for key in ("n", "mse", "rmse", "mae", "absrel", "lg10",
                    "delta1", "delta2", "delta3"):
            self.assertAlmostEqual(split[key], full[key], places=6, msg=key)

    def test_pixelwise_differs_from_per_image_average(self):
        """确认实现确实是 pixel-wise，而不是上游的按图平均。

        两张图：图 A 误差大但有效像素多，图 B 误差小但有效像素少。
        pixel-wise 会被 A 主导；按图平均则两图等权。
        """
        target_a = torch.ones(1, 1, 1, 100) * 1.0
        pred_a = torch.ones(1, 1, 1, 100) * 5.0          # 误差 4，100 个像素
        target_b = torch.ones(1, 1, 1, 1) * 1.0
        pred_b = torch.ones(1, 1, 1, 1) * 2.0            # 误差 1，1 个像素

        acc = MetricAccumulator()
        acc.update(pred_a, target_a)
        acc.update(pred_b, target_b)
        pixelwise = acc.compute()["rmse"]

        per_image = (4.0 + 1.0) / 2.0
        expected = math.sqrt((100 * 16.0 + 1 * 1.0) / 101.0)
        self.assertAlmostEqual(pixelwise, expected, places=6)
        self.assertNotAlmostEqual(pixelwise, per_image, places=3)

    def test_image_average_matches_original_protocol(self):
        """image 口径 = 每张图 RMSE 的算术平均（原版 Lua / PyTorch 仓库）。"""
        # 图 A：100 个像素，误差 4；图 B：1 个像素，误差 1
        target_a = torch.ones(1, 1, 1, 100)
        pred_a = torch.full((1, 1, 1, 100), 5.0)
        target_b = torch.ones(1, 1, 1, 1)
        pred_b = torch.full((1, 1, 1, 1), 2.0)

        acc = MetricAccumulator("image")
        acc.update(pred_a, target_a)
        acc.update(pred_b, target_b)
        m = acc.compute()

        self.assertEqual(m["n"], 2)                     # 2 张图
        self.assertAlmostEqual(m["rmse"], (4.0 + 1.0) / 2.0, places=6)
        # mse 也是按图平均，不是 sqrt 关系
        self.assertAlmostEqual(m["mse"], (16.0 + 1.0) / 2.0, places=6)

    def test_pixel_average_is_higher_in_this_case(self):
        target_a = torch.ones(1, 1, 1, 100)
        pred_a = torch.full((1, 1, 1, 100), 5.0)
        target_b = torch.ones(1, 1, 1, 1)
        pred_b = torch.full((1, 1, 1, 1), 2.0)

        acc = MetricAccumulator("pixel")
        acc.update(pred_a, target_a)
        acc.update(pred_b, target_b)
        m = acc.compute()

        self.assertEqual(m["n"], 101)                   # 101 个有效像素
        self.assertAlmostEqual(m["rmse"], math.sqrt(1601.0 / 101.0), places=6)
        self.assertGreater(m["rmse"], 2.5)              # 比按图平均大

    def test_image_mode_batch_matches_single_updates(self):
        target = torch.ones(2, 1, 1, 4)
        pred = torch.ones(2, 1, 1, 4) * 2.0
        pred[1] = 3.0                                   # 第二张图误差更大

        batch_acc = MetricAccumulator("image")
        batch_acc.update(pred, target)

        single_acc = MetricAccumulator("image")
        single_acc.update(pred[:1], target[:1])
        single_acc.update(pred[1:], target[1:])

        a = batch_acc.compute()
        b = single_acc.compute()
        for key in ("n", "mse", "rmse", "mae", "absrel", "lg10",
                    "delta1", "delta2", "delta3"):
            self.assertAlmostEqual(a[key], b[key], places=6, msg=key)

    def test_image_mode_skips_all_invalid_image(self):
        target = torch.ones(1, 1, 2, 2)
        pred = torch.full_like(target, 2.0)
        acc = MetricAccumulator("image")
        acc.update(pred, target)
        acc.update(pred, torch.zeros_like(target))      # 全无效，应被跳过
        self.assertEqual(acc.compute()["n"], 1)


if __name__ == "__main__":
    unittest.main()
