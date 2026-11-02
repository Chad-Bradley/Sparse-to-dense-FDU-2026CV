"""ResNet 编码器（上游仓库版本）。"""

from __future__ import annotations

import math

import torch.nn as nn
import torchvision
from torchvision.models import ResNet18_Weights, ResNet50_Weights


def weights_init(m):
    """上游仓库的初始化方式：高斯 N(0, sqrt(2/n))，BN 为恒等。

    注意 n 的定义：Conv2d 用 k*k*out_channels，ConvTranspose2d 用 k*k*in_channels。
    这不是 PyTorch 默认的 kaiming，而是上游 `models.py:23-37` 的做法；
    阶段 A 为对齐 baseline，原样保留。
    """
    if isinstance(m, nn.Conv2d):
        n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
        m.weight.data.normal_(0, math.sqrt(2.0 / n))
        if m.bias is not None:
            m.bias.data.zero_()
    elif isinstance(m, nn.ConvTranspose2d):
        n = m.kernel_size[0] * m.kernel_size[1] * m.in_channels
        m.weight.data.normal_(0, math.sqrt(2.0 / n))
        if m.bias is not None:
            m.bias.data.zero_()
    elif isinstance(m, nn.BatchNorm2d):
        m.weight.data.fill_(1)
        m.bias.data.zero_()


class ResNetEncoder(nn.Module):
    """torchvision ResNet → layer4 输出，无 avgpool/fc，无 skip connection。

    参数：
        arch:        "resnet50"（正式实验）/ "resnet18"（单元测试用）
        in_channels: 4（rgbd）/ 3（rgb）/ 1（d）
        pretrained:  是否加载 ImageNet 权重
        conv1_init:  "random"（阶段 A，与上游一致）
                     "copy"（阶段 B 的改进：前 3 通道拷 ImageNet 权重，
                             其余通道置 0，BN 参数也一起拷）
    """

    def __init__(self, arch="resnet50", in_channels=4,
                 pretrained=True, conv1_init="random"):
        super().__init__()
        assert arch in ("resnet18", "resnet50"), "只支持 resnet18/50"
        assert conv1_init in ("random", "copy"), "conv1_init 只能是 random/copy"

        if arch == "resnet50":
            weights = ResNet50_Weights.IMAGENET1K_V1 if pretrained else None
        else:
            weights = ResNet18_Weights.IMAGENET1K_V1 if pretrained else None
        net = getattr(torchvision.models, arch)(weights=weights)

        # ------------------------------------------------------------------
        # 第一层卷积改成 in_channels 输入。
        #
        # 上游的做法（`models.py:170-177`）：只要 in_channels != 3，就同时重建
        # conv1 和 bn1 并随机初始化，**不拷贝** ImageNet 的 RGB 滤波器。
        # 为什么不能直接改原 conv1 的 weight 形状：预训练权重是 3 通道，
        # shape 不匹配；而且 4 通道输入的统计分布和第 4 通道（稀疏深度）不同，
        # BN 的 running stats 也不该直接沿用。
        #
        # 阶段 B 的 copy 模式会把前 min(3, in_channels) 个通道拷过来，
        # 剩余通道置 0，让网络从 ImageNet 的 RGB 先验出发。
        # ------------------------------------------------------------------
        if in_channels != 3:
            conv1 = nn.Conv2d(in_channels, 64, kernel_size=7, stride=2,
                              padding=3, bias=False)
            bn1 = nn.BatchNorm2d(64)
            if conv1_init == "copy" and pretrained:
                with torch.no_grad():
                    k = min(3, in_channels)
                    conv1.weight[:, :k] = net.conv1.weight[:, :k]
                    conv1.weight[:, k:] = 0
                    bn1.weight.copy_(net.bn1.weight)
                    bn1.bias.copy_(net.bn1.bias)
                    bn1.running_mean.copy_(net.bn1.running_mean)
                    bn1.running_var.copy_(net.bn1.running_var)
            else:
                weights_init(conv1)
                weights_init(bn1)
            net.conv1 = conv1
            net.bn1 = bn1

        # 只保留到 layer4：avgpool 和 fc 直接丢弃（上游没有用它们）
        self.conv1 = net.conv1
        self.bn1 = net.bn1
        self.relu = net.relu
        self.maxpool = net.maxpool
        self.layer1 = net.layer1
        self.layer2 = net.layer2
        self.layer3 = net.layer3
        self.layer4 = net.layer4

        self.out_channels = 2048 if arch == "resnet50" else 512

    def forward(self, x):
        # 尺寸（输入 228×304）：
        #   4×228×304 → conv1 s2 → 64×114×152 → maxpool s2 → 64×57×76
        #   → layer1 256×57×76 → layer2 512×29×38
        #   → layer3 1024×15×19 → layer4 2048×8×10
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        return x
