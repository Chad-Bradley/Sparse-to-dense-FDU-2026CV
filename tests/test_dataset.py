"""数据层单元测试。

运行（从 s2d_baseline/ 目录）：

    python -m unittest discover -s tests -v

不需要真实数据的测试（稀疏采样、几何变换）在本地也能跑；
需要 h5 的测试在数据目录不存在时自动 SkipTest。
"""

from __future__ import annotations

import os
import unittest

import numpy as np

from s2d.config import Config
from s2d.data.dataset import NYUDataset
from s2d.data.sparse import UniformSampling
from s2d.data.transforms import (
    IMAGENET_MEAN,
    IMAGENET_STD,
    CenterCrop,
    HorizontalFlip,
    Lighting,
    Normalize,
    Resize,
    Rotate,
)


class TestUniformSampling(unittest.TestCase):
    def setUp(self):
        # 构造一块 40×50 的有效区域，深度 0.5~9 m，周围全是 0（无效）
        rng = np.random.RandomState(0)
        self.depth = np.zeros((50, 60), dtype=np.float32)
        self.depth[5:45, 5:55] = rng.uniform(
            0.5, 9.0, size=(40, 50)).astype(np.float32)

    def test_exact_count(self):
        np.random.seed(0)
        mask = UniformSampling(500)(self.depth)
        self.assertEqual(int(mask.sum()), 500)

    def test_exact_points_are_valid(self):
        np.random.seed(0)
        mask = UniformSampling(500)(self.depth)
        # 所有被选中的像素都必须有观测（depth > 0）
        self.assertTrue(np.all(self.depth[mask] > 0))
        # 且不会重复选中同一个像素（mask 天然去重，这里再确认一下）
        self.assertEqual(int(mask.sum()), int(np.count_nonzero(mask)))

    def test_exact_never_exceeds_valid(self):
        np.random.seed(0)
        mask = UniformSampling(10 ** 6)(self.depth)
        self.assertEqual(int(mask.sum()), 40 * 50)

    def test_max_depth_filter(self):
        np.random.seed(0)
        mask = UniformSampling(100, max_depth=2.0)(self.depth)
        self.assertTrue(np.all(self.depth[mask] <= 2.0))
        self.assertTrue(np.all(self.depth[mask] > 0))

    def test_empty_depth(self):
        mask = UniformSampling(500)(np.zeros((10, 10), np.float32))
        self.assertEqual(int(mask.sum()), 0)


class TestTransforms(unittest.TestCase):
    def test_resize_factor(self):
        rgb = np.zeros((480, 640, 3), dtype=np.uint8)
        depth = np.zeros((480, 640), dtype=np.float32)
        self.assertEqual(Resize(0.5)(rgb).shape, (240, 320, 3))
        self.assertEqual(Resize(0.5)(depth).shape, (240, 320))
        self.assertEqual(Resize(250.0 / 480)(depth).shape, (250, 333))

    def test_resize_depth_keeps_original_values(self):
        # 最近邻缩放不应产生原图里没有的深度值
        depth = np.zeros((40, 60), dtype=np.float32)
        values = np.arange(1, 10, dtype=np.float32)
        depth[10:30, 10:50] = np.random.RandomState(0).choice(
            values, size=(20, 40))
        out = Resize(0.75)(depth)
        nonzero = np.unique(out[out > 0])
        self.assertTrue(np.all(np.isin(nonzero, values)))

    def test_center_crop_shape(self):
        depth = np.zeros((250, 333), dtype=np.float32)
        self.assertEqual(CenterCrop((228, 304))(depth).shape, (228, 304))

    def test_rotate_zero_angle_is_identity(self):
        rng = np.random.RandomState(0)
        depth = rng.uniform(0.5, 8.0, size=(50, 60)).astype(np.float32)
        out = Rotate(0.0)(depth)
        np.testing.assert_allclose(out, depth)

    def test_flip(self):
        depth = np.arange(12, dtype=np.float32).reshape(3, 4)
        np.testing.assert_array_equal(HorizontalFlip(True)(depth),
                                      np.fliplr(depth))
        np.testing.assert_array_equal(HorizontalFlip(False)(depth), depth)


class TestNormalizeLighting(unittest.TestCase):
    def test_normalize(self):
        img = np.full((4, 4, 3), 0.5, dtype=np.float32)
        out = Normalize()(img)
        expected = (0.5 - IMAGENET_MEAN) / IMAGENET_STD
        np.testing.assert_allclose(out[0, 0], expected, rtol=1e-5)

    def test_lighting_zero_std_is_identity(self):
        img = np.random.rand(4, 4, 3).astype(np.float32)
        np.testing.assert_array_equal(Lighting(0.0)(img), img)

    def test_lighting_offset_is_spatially_constant(self):
        img = np.zeros((4, 4, 3), dtype=np.float32)
        out = Lighting(0.1)(img)
        # 光照噪声是整张图的常数偏移，不同像素应完全相同
        np.testing.assert_allclose(out[0, 0], out[3, 3], atol=1e-6)
        self.assertGreater(float(np.abs(out).max()), 0.0)


class TestNYUDataset(unittest.TestCase):
    """需要真实数据；数据不存在时跳过。"""

    @classmethod
    def setUpClass(cls):
        cls.root = os.path.join(Config().data_root, "val")
        if not os.path.isdir(cls.root):
            raise unittest.SkipTest(
                "找不到 {}，跳过需要真实数据的测试".format(cls.root))

    def test_length(self):
        ds = NYUDataset(self.root, split="val", modality="rgbd")
        self.assertEqual(len(ds), 654)

    def test_val_sample_rgbd(self):
        ds = NYUDataset(self.root, split="val", modality="rgbd",
                        num_samples=500)
        inp, tgt = ds[0]
        self.assertEqual(tuple(inp.shape), (4, 228, 304))
        self.assertEqual(tuple(tgt.shape), (1, 228, 304))

        # RGB 已做 ImageNet 归一化：数值不再在 [0,1]，只要求有限且范围合理
        self.assertTrue(bool((inp[:3] > -8).all() and (inp[:3] < 8).all()))

        # 严格 500 个稀疏点，且值与 GT 一致
        n_sparse = int((inp[3] > 0).sum())
        self.assertEqual(n_sparse, 500)
        self.assertTrue(np.allclose(
            inp[3][inp[3] > 0].numpy(), tgt[0][inp[3] > 0].numpy()))

        # 深度量纲应为米
        valid = tgt[tgt > 0]
        self.assertGreater(float(valid.median()), 0.1)
        self.assertLess(float(valid.median()), 30.0)

    def test_modalities(self):
        for modality, channels in (("rgb", 3), ("d", 1)):
            ds = NYUDataset(self.root, split="val", modality=modality)
            inp, _ = ds[0]
            self.assertEqual(tuple(inp.shape), (channels, 228, 304))

    def test_deterministic_with_seed(self):
        ds = NYUDataset(self.root, split="val", modality="rgbd",
                        num_samples=500)
        np.random.seed(123)
        a_in, a_tgt = ds[0]
        np.random.seed(123)
        b_in, b_tgt = ds[0]
        self.assertTrue(bool((a_in == b_in).all()))
        self.assertTrue(bool((a_tgt == b_tgt).all()))


if __name__ == "__main__":
    unittest.main()
