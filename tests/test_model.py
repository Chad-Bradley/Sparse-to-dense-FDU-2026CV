"""模型单元测试。

全部用 pretrained=False + resnet18，避免联网下载权重，也能在 CPU 上快速跑完。
"""

from __future__ import annotations

import unittest

import torch

from s2d.models.decoder import Unpool, choose_decoder
from s2d.models.encoder import ResNetEncoder
from s2d.models.sparse2dense import SparseToDense


class TestUnpool(unittest.TestCase):
    def test_zero_padding_unpool(self):
        x = torch.arange(1, 5, dtype=torch.float32).reshape(1, 1, 2, 2)
        y = Unpool(1)(x)
        expected = torch.tensor([[
            [[1, 0, 2, 0],
             [0, 0, 0, 0],
             [3, 0, 4, 0],
             [0, 0, 0, 0]]]], dtype=torch.float32)
        self.assertTrue(torch.equal(y, expected))


class TestDecoders(unittest.TestCase):
    def test_all_decoder_shapes(self):
        # 用较小的 in_channels 测试形状，避免在 CPU 上构建几十 M 参数的 UpProj。
        # 4 级后：in_channels -> in_channels/16，分辨率 ×16。
        x = torch.zeros(1, 64, 2, 2)
        for name in ("deconv2", "deconv3", "upconv", "upproj"):
            with self.subTest(decoder=name):
                decoder = choose_decoder(name, 64)
                y = decoder(x)
                self.assertEqual(tuple(y.shape), (1, 4, 32, 40),
                                 "{} 输出形状不对".format(name))

    def test_unknown_decoder(self):
        with self.assertRaises(ValueError):
            choose_decoder("bogus", 64)


class TestEncoder(unittest.TestCase):
    def test_4ch_conv1_and_bottleneck_shape(self):
        enc = ResNetEncoder(arch="resnet18", in_channels=4, pretrained=False)
        self.assertEqual(enc.conv1.in_channels, 4)
        x = torch.zeros(1, 4, 128, 160)
        y = enc(x)      # 128/32=4, 160/32=5
        self.assertEqual(tuple(y.shape), (1, 512, 4, 5))
        self.assertEqual(enc.out_channels, 512)

    def test_3ch_keeps_original_conv1(self):
        enc = ResNetEncoder(arch="resnet18", in_channels=3, pretrained=False)
        self.assertEqual(enc.conv1.in_channels, 3)


class TestSparseToDense(unittest.TestCase):
    def test_forward_shape(self):
        model = SparseToDense(
            in_channels=4, arch="resnet18", decoder="deconv3",
            output_size=(228, 304), pretrained=False)
        model.eval()
        with torch.no_grad():
            y = model(torch.zeros(1, 4, 128, 160))
        self.assertEqual(tuple(y.shape), (1, 1, 228, 304))

    def test_all_modalities_forward(self):
        for in_channels in (3, 1, 4):
            with self.subTest(in_channels=in_channels):
                model = SparseToDense(
                    in_channels=in_channels, arch="resnet18",
                    decoder="deconv3", output_size=(64, 80),
                    pretrained=False)
                model.eval()
                with torch.no_grad():
                    y = model(torch.zeros(1, in_channels, 64, 80))
                self.assertEqual(tuple(y.shape), (1, 1, 64, 80))

    def test_parameter_count_resnet50(self):
        # ResNet50 + deconv3 应在 30M~35M 量级；明显偏离说明接错了层
        model = SparseToDense(
            in_channels=4, arch="resnet50", decoder="deconv3",
            output_size=(228, 304), pretrained=False)
        n = sum(p.numel() for p in model.parameters())
        self.assertGreater(n, 25e6)
        self.assertLess(n, 40e6)

    def test_bottleneck_shape(self):
        # 确认 conv2 之后是 1024 通道、空间尺寸不变
        model = SparseToDense(
            in_channels=4, arch="resnet18", decoder="deconv3",
            output_size=(228, 304), pretrained=False)
        x = torch.zeros(1, 4, 64, 80)
        f = model.encoder(x)
        self.assertEqual(tuple(f.shape), (1, 512, 2, 3))
        f = model.bn2(model.conv2(f))
        self.assertEqual(tuple(f.shape), (1, 256, 2, 3))


if __name__ == "__main__":
    unittest.main()
