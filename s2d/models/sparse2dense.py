"""完整网络组装：ResNet 编码器 + 瓶颈卷积 + 解码器 + 输出头。

本文件对应上游仓库 `models.py:156-230` 的 `ResNet` 类。阶段 A 力求逐层对齐：

    x = encoder(x)        # 2048×8×10（ResNet50）
    x = conv2(x)          # 1×1, 2048→1024, 无 ReLU
    x = bn2(x)
    x = decoder(x)        # 4 级上采样，1024→512→256→128→64
    x = conv3(x)          # 3×3, 64→1
    x = bilinear(x)       # 插值到 228×304

注意顺序：上游是**先卷积再插值**（与本仓库早期笔记里「先插值再卷积」的描述相反），
阶段 A 以代码为准。
"""

from __future__ import annotations

import torch.nn as nn
import torch.nn.functional as F

from s2d.models.decoder import choose_decoder
from s2d.models.encoder import ResNetEncoder, weights_init
from s2d.models.invalid_points import InvalidPointAdapter


class SparseToDense(nn.Module):
    """输入 RGB+稀疏深度，输出同尺寸稠密深度图。

    参数：
        in_channels:       4（rgbd）/ 3（rgb）/ 1（d）
        arch:              "resnet50" / "resnet18"
        decoder:           "deconv2" / "deconv3" / "upconv" / "upproj"
                           默认 upproj（Lua 原版）；deconv3 对应 PyTorch baseline
        output_size:       (H, W) = (228, 304)
        pretrained:        是否加载 ImageNet 权重
        conv1_init:        "random"（阶段 A，与上游一致）/ "copy"（阶段 B）
        bottleneck_kernel: 上游用 1；保留参数是为了做阶段 B 的消融
    """

    def __init__(self, in_channels=4, arch="resnet50", decoder="upproj",
                 output_size=(228, 304), pretrained=True,
                 conv1_init="random", bottleneck_kernel=1,
                 learned_upsample=False, invalid_mode="zero", invalid_value=-1.0):
        super().__init__()
        self.output_size = tuple(output_size)
        self.input_adapter = InvalidPointAdapter(in_channels, invalid_mode, invalid_value)
        self.learned_upsample = learned_upsample
        self.encoder = ResNetEncoder(
            arch=arch, in_channels=self.input_adapter.out_channels,
            pretrained=pretrained, conv1_init=conv1_init)

        # 上游 conv2：把主干通道压一半。ResNet50: 2048→1024，ResNet18: 512→256。
        mid_channels = self.encoder.out_channels // 2
        self.conv2 = nn.Conv2d(
            self.encoder.out_channels, mid_channels,
            kernel_size=bottleneck_kernel,
            padding=bottleneck_kernel // 2,
            bias=False)
        self.bn2 = nn.BatchNorm2d(mid_channels)
        # 注意：上游的 conv2 之后**不加 ReLU**，这里保持一致。

        self.decoder = choose_decoder(decoder, mid_channels)

        # 上游 conv3：3×3，输出单通道。
        # in_channels = 主干通道数 // 32 = 解码器输出通道数
        # （ResNet50: 2048//32 = 64；ResNet18: 512//32 = 16）。
        self.conv3 = nn.Conv2d(
            self.encoder.out_channels // 32, 1,
            kernel_size=3, stride=1, padding=1, bias=False)

        # 最后一步才插值到目标尺寸；align_corners=True 与上游一致。
        self.bilinear = nn.Upsample(size=self.output_size, mode="bilinear",
                                    align_corners=True)

        # 只初始化新增的层。encoder 里除被替换的 conv1/bn1 外都保留预训练权重，
        # 不要再 apply 一次，否则会把 ImageNet 权重覆盖掉。
        self.conv2.apply(weights_init)
        self.bn2.apply(weights_init)
        self.decoder.apply(weights_init)
        self.conv3.apply(weights_init)
        if learned_upsample:
            channels = self.encoder.out_channels // 32
            self.learned_head = nn.Sequential(
                nn.Conv2d(channels, channels // 2, 3, padding=1, bias=False),
                nn.ReLU(inplace=True),
                nn.Conv2d(channels // 2, 1, 3, padding=1, bias=False),
            )
            self.learned_head.apply(weights_init)
            self.conv3 = None

    def forward(self, x):
        x = self.input_adapter(x)
        x = self.encoder(x)       # B×2048×8×10
        x = self.conv2(x)         # B×1024×8×10
        x = self.bn2(x)
        x = self.decoder(x)       # B×64×128×160
        if self.learned_upsample:
            return self.learned_head(F.interpolate(
                x, size=self.output_size, mode="bilinear", align_corners=True))
        x = self.conv3(x)         # B×1×128×160
        x = self.bilinear(x)      # B×1×228×304
        return x
