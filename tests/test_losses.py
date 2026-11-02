"""损失函数单元测试（不需要 GPU / 真实数据）。"""

from __future__ import annotations

import unittest

import torch

from s2d.losses import MaskedL1Loss, MaskedMSELoss, build_criterion


class TestMaskedLosses(unittest.TestCase):
    def setUp(self):
        # 有效像素：左上 (1) 和右下 (3)；另外两个位置 target=0，应被忽略
        self.target = torch.tensor([[[[1.0, 0.0],
                                      [0.0, 3.0]]]])
        # 有效像素上的 diff：+1 和 -2
        self.pred_a = torch.tensor([[[[2.0, 100.0],
                                      [100.0, 1.0]]]])
        # 无效像素处换一组完全不同的值
        self.pred_b = torch.tensor([[[[2.0, -100.0],
                                      [50.0, 1.0]]]])

    def test_invalid_pixels_ignored(self):
        l1 = MaskedL1Loss()
        mse = MaskedMSELoss()
        self.assertTrue(torch.allclose(l1(self.pred_a, self.target),
                                       l1(self.pred_b, self.target)))
        self.assertTrue(torch.allclose(mse(self.pred_a, self.target),
                                       mse(self.pred_b, self.target)))

    def test_l1_value(self):
        # (|1| + |-2|) / 2 = 1.5
        loss = MaskedL1Loss()(self.pred_a, self.target)
        self.assertAlmostEqual(float(loss), 1.5, places=6)

    def test_mse_value(self):
        # (1^2 + 2^2) / 2 = 2.5
        loss = MaskedMSELoss()(self.pred_a, self.target)
        self.assertAlmostEqual(float(loss), 2.5, places=6)

    def test_perfect_prediction(self):
        self.assertAlmostEqual(
            float(MaskedL1Loss()(self.target, self.target)), 0.0, places=6)
        self.assertAlmostEqual(
            float(MaskedMSELoss()(self.target, self.target)), 0.0, places=6)

    def test_backward_masks_invalid_pixels(self):
        pred = self.pred_a.clone().requires_grad_(True)
        loss = MaskedL1Loss()(pred, self.target)
        loss.backward()

        grad = pred.grad
        # 有效像素：diff 分别为 +1 和 -2，所以梯度是 +1/N 和 -1/N，N=2
        self.assertAlmostEqual(float(grad[0, 0, 0, 0]), 0.5, places=6)
        self.assertAlmostEqual(float(grad[0, 0, 1, 1]), -0.5, places=6)
        # 无效像素：梯度必须严格为 0
        self.assertEqual(float(grad[0, 0, 0, 1]), 0.0)
        self.assertEqual(float(grad[0, 0, 1, 0]), 0.0)

    def test_all_invalid_is_backward_safe(self):
        target = torch.zeros(1, 1, 2, 2)
        pred = torch.ones(1, 1, 2, 2, requires_grad=True)
        loss = MaskedL1Loss()(pred, target)
        self.assertEqual(float(loss), 0.0)
        loss.backward()          # 不应抛 "does not require grad"
        self.assertIsNotNone(pred.grad)
        self.assertTrue(torch.all(pred.grad == 0))

    def test_build_criterion(self):
        self.assertIsInstance(build_criterion("l1"), MaskedL1Loss)
        self.assertIsInstance(build_criterion("l2"), MaskedMSELoss)
        self.assertIsInstance(build_criterion("mse"), MaskedMSELoss)
        with self.assertRaises(ValueError):
            build_criterion("huber")


if __name__ == "__main__":
    unittest.main()
