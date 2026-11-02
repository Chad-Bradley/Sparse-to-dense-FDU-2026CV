"""模型层。

对外接口：

    from s2d.models.sparse2dense import SparseToDense
    from s2d.models.decoder import choose_decoder, Decoder

    model = SparseToDense(
        in_channels=4,          # rgb=3 / d=1 / rgbd=4
        arch="resnet50",
        decoder="upproj",       # 默认（Lua 原版）；deconv3 对应 PyTorch baseline
        output_size=(228, 304),
        pretrained=True,
        conv1_init="random",    # 阶段 A；copy 留给阶段 B
        bottleneck_kernel=1,    # 上游的 1×1 瓶颈卷积
    )

文件划分：
    encoder.py       ResNet 主干（4 通道 conv1、去掉 avgpool/fc）
    decoder.py       deconv2 / deconv3 / upconv / upproj
    sparse2dense.py  组装完整网络
"""

from s2d.models.decoder import Decoder, choose_decoder
from s2d.models.encoder import ResNetEncoder, weights_init
from s2d.models.sparse2dense import SparseToDense

__all__ = [
    "Decoder",
    "ResNetEncoder",
    "SparseToDense",
    "choose_decoder",
    "weights_init",
]
