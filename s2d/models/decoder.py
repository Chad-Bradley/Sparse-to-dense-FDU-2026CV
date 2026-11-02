"""解码器：deconv2 / deconv3 / upconv / upproj（上游仓库版本）。

四种解码器的共同点（也是与上游 `models.py` 对齐的地方）：
    - 4 级串联，每级分辨率 ×2、通道 ÷2；
    - 输入通道数 = 编码器瓶颈卷积后的通道数（ResNet50 时 1024）；
    - 输出通道数 = in_channels / 16（ResNet50 时 64）。
"""

from __future__ import annotations

import collections

import torch
import torch.nn as nn
import torch.nn.functional as F


class Decoder(nn.Module):
    """统一用 Sequential 串联 4 级，避免像上游那样手写 layer1..4 + forward。"""

    names = ["deconv2", "deconv3", "upconv", "upproj"]

    def __init__(self, blocks):
        super().__init__()
        self.blocks = nn.Sequential(*blocks)

    def forward(self, x):
        return self.blocks(x)


class Unpool(nn.Module):
    """2×2 零填充反池化（UpConv / UpProj 共用）。

    每个输入值放到输出 2×2 邻域的左上角，其余三个位置补 0：
        [[1, 2],        [[1, 0, 2, 0],
         [3, 4]]   ->    [0, 0, 0, 0],
                         [3, 0, 4, 0],
                         [0, 0, 0, 0]]

    实现为固定权重的分组转置卷积：核 [[1, 0], [0, 0]]，不可学习。
    上游把这份权重写成了 `.cuda()` 的 Variable（`models.py:17`），
    这里用 register_buffer，让它跟着模型的 device 走。
    """

    def __init__(self, num_channels, stride=2):
        super().__init__()
        self.stride = stride
        weight = torch.zeros(num_channels, 1, stride, stride)
        weight[:, :, 0, 0] = 1
        self.register_buffer("weight", weight)

    def forward(self, x):
        return F.conv_transpose2d(x, self.weight, stride=self.stride,
                                  groups=x.shape[1])


class DeConv(Decoder):
    """转置卷积解码器。

    kernel_size=2 -> deconv2：padding=0, output_padding=0
    kernel_size=3 -> deconv3：padding=1, output_padding=1
    两种情况都严格满足 H_out = 2 * H_in。
    """

    def __init__(self, in_channels, kernel_size):
        assert kernel_size >= 2, "kernel_size 必须 >= 2"

        def block(channels):
            stride = 2
            padding = (kernel_size - 1) // 2
            output_padding = kernel_size % 2
            # 转置卷积输出尺寸：H_out = (H-1)*s - 2*p + k + op
            # 要恰好 2H，就必须满足下面这个等式；上游也用 assert 守着。
            assert -2 - 2 * padding + kernel_size + output_padding == 0

            return nn.Sequential(collections.OrderedDict([
                ("deconv", nn.ConvTranspose2d(
                    channels, channels // 2, kernel_size,
                    stride, padding, output_padding, bias=False)),
                ("batchnorm", nn.BatchNorm2d(channels // 2)),
                ("relu", nn.ReLU(inplace=True)),
            ]))

        super().__init__([
            block(in_channels),
            block(in_channels // 2),
            block(in_channels // 4),
            block(in_channels // 8),
        ])
        self.in_channels = in_channels


class UpConv(Decoder):
    """Unpool + 5×5 卷积：先零填充上采样，再用卷积补出细节。"""

    @staticmethod
    def _block(in_channels):
        out_channels = in_channels // 2
        return nn.Sequential(collections.OrderedDict([
            ("unpool", Unpool(in_channels)),
            ("conv", nn.Conv2d(in_channels, out_channels, kernel_size=5,
                               stride=1, padding=2, bias=False)),
            ("batchnorm", nn.BatchNorm2d(out_channels)),
            ("relu", nn.ReLU(inplace=True)),
        ]))

    def __init__(self, in_channels):
        super().__init__([
            self._block(in_channels),
            self._block(in_channels // 2),
            self._block(in_channels // 4),
            self._block(in_channels // 8),
        ])
        self.in_channels = in_channels


class UpProjModule(nn.Module):
    """上投影模块（Laina et al.）：

        unpool → ┬ 5×5 conv → BN → ReLU → 3×3 conv → BN ┐
                 └ 5×5 conv → BN ────────────────────────┴ → 相加 → ReLU

    两条支路输出通道数相同，所以可以直接相加；残差结构让上采样
    不必只靠一个转置卷积恢复细节。
    """

    def __init__(self, in_channels):
        super().__init__()
        out_channels = in_channels // 2
        self.unpool = Unpool(in_channels)
        self.upper_branch = nn.Sequential(collections.OrderedDict([
            ("conv1", nn.Conv2d(in_channels, out_channels, kernel_size=5,
                                stride=1, padding=2, bias=False)),
            ("batchnorm1", nn.BatchNorm2d(out_channels)),
            ("relu", nn.ReLU(inplace=True)),
            ("conv2", nn.Conv2d(out_channels, out_channels, kernel_size=3,
                                stride=1, padding=1, bias=False)),
            ("batchnorm2", nn.BatchNorm2d(out_channels)),
        ]))
        self.bottom_branch = nn.Sequential(collections.OrderedDict([
            ("conv", nn.Conv2d(in_channels, out_channels, kernel_size=5,
                               stride=1, padding=2, bias=False)),
            ("batchnorm", nn.BatchNorm2d(out_channels)),
        ]))
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        x = self.unpool(x)
        return self.relu(self.upper_branch(x) + self.bottom_branch(x))


class UpProj(Decoder):
    """4 个 UpProjModule 串联。"""

    def __init__(self, in_channels):
        super().__init__([
            UpProjModule(in_channels),
            UpProjModule(in_channels // 2),
            UpProjModule(in_channels // 4),
            UpProjModule(in_channels // 8),
        ])
        self.in_channels = in_channels


def choose_decoder(name, in_channels):
    """按名字构建解码器（对应上游 `models.py:142-153`）。"""
    if name in ("deconv2", "deconv3"):
        return DeConv(in_channels, kernel_size=int(name[-1]))
    if name == "upconv":
        return UpConv(in_channels)
    if name == "upproj":
        return UpProj(in_channels)
    raise ValueError("未知 decoder: {}（可选 {}）".format(
        name, ", ".join(Decoder.names)))
