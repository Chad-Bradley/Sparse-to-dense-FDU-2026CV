"""s2d_baseline：Sparse-to-Dense 的独立复现（仅 NYU Depth V2）。

与仓库根目录的上游实现互不依赖。包内分层：

    data/      数据集、增广、稀疏化
    models/    编码器、UpProj 解码器、完整网络
    losses.py  带无效像素掩码的回归损失
    metrics.py 按像素累计的评测指标
    engine.py  训练 / 评估循环
    viz.py     深度可视化
    analysis.py 边界 / 无纹理分桶统计
    config.py  配置项
    utils.py   种子、checkpoint、日志
"""

__version__ = "0.1.0"
