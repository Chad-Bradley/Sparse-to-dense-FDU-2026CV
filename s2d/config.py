from __future__ import annotations

import os
from dataclasses import asdict, dataclass
from pathlib import Path

# ---------------------------------------------------------------------------
# 路径约定
#
# 本目录现在是一个**独立仓库**，结构：
#   <repo>/s2d/config.py                          （本文件）
#   <repo>/data/nyudepthv2/{train,val}/<scene>/*.h5
#
# parents[1] 从 s2d/config.py 往上数：s2d -> 仓库根目录。
#
# 数据不在本仓库里时，用环境变量 S2D_DATA_ROOT 指定，例如：
#   export S2D_DATA_ROOT=/root/DZQ/CV/sparse-to-dense.pytorch-master/data/nyudepthv2
# 或者在仓库根目录创建 data 软链接：
#   ln -s /root/Test/data data
# ---------------------------------------------------------------------------
REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_DATA_ROOT = Path(
    os.environ.get("S2D_DATA_ROOT", str(REPO_ROOT / "data" / "nyudepthv2"))
)

# 训练输出默认放在 <repo>/runs；服务器根盘常年 99% 满，
# 因此允许用环境变量 S2D_RUNS 指到 /root/Test 的大盘上。
DEFAULT_RUNS_DIR = Path(
    os.environ.get("S2D_RUNS", str(REPO_ROOT / "runs"))
)


@dataclass
class Config:
    """所有超参数的单一来源，避免上游那种模块级全局 args。"""

    # --- 数据 ---
    data_root: str = str(DEFAULT_DATA_ROOT)
    modality: str = "rgbd"          # rgb | d | rgbd
    # 稀疏深度点数：每张图**严格** 500 个（无放回采样）。
    # 这里不跟上游的概率采样保持一致——实验要求是「包含 500 个有效点」，
    # 概率采样的点数会波动（期望 500，实际约 500±22）。
    num_samples: int = 500
    max_depth: float = -1.0         # 负数表示不限制最大深度
    sparsifier: str = "uniform"
    # 新增：深度噪声强度
    depth_noise_std: float = 0.0    # 稀疏深度的高斯噪声标准差，0 = 关闭
    output_size: tuple = (228, 304)  # (H, W)，NYU 原始 480×640 经增广后的尺寸
    # RGB 预处理（与 Lua 原版对齐）：
    #   rgb_normalize  -> ImageNet mean/std 标准化（训练/验证都做）
    #   lighting_std   -> AlexNet PCA 光照噪声标准差，仅训练；0 = 关闭
    rgb_normalize: bool = True
    lighting_std: float = 0.1

    # --- 模型（阶段 A 一律用上游仓库设置，不做改进）---
    arch: str = "resnet50"
    # 默认用 Lua 原版的 upproj；想复现 PyTorch baseline 可 --decoder deconv3
    decoder: str = "upproj"
    # 上游对 4 通道 conv1 是整体随机初始化；copy 属于阶段 B 的改进，先不用。
    conv1_init: str = "random"
    # 上游的瓶颈卷积是 1×1、2048→1024、后接 BN 且不加 ReLU。
    bottleneck_kernel: int = 1
    pretrained: bool = True

    # --- 优化 ---
    epochs: int = 15
    batch_size: int = 8
    eval_batch_size: int = 1
    lr: float = 0.01
    momentum: float = 0.9
    weight_decay: float = 1e-4
    lr_step: int = 5
    lr_gamma: float = 0.1
    criterion: str = "l1"
    # 指标聚合口径：
    #   image = 每张图先算 RMSE 再平均（原版 Lua / PyTorch 仓库，验证 batch=1）
    #   pixel = 所有有效像素累计后开方（常见论文口径）
    # 与 baseline 的 0.220 对比必须用 image。
    metric_reduction: str = "image"
    workers: int = 10
    seed: int = 0

    # --- 输出 ---
    out_dir: str = str(DEFAULT_RUNS_DIR)
    run_name: str = ""
    print_freq: int = 50
    save_samples: int = 8

    def __post_init__(self):
        # JSON 反序列化会把 tuple 变成 list；统一转回 tuple，便于比较和传给模型。
        if not isinstance(self.output_size, tuple):
            self.output_size = tuple(self.output_size)

    def to_dict(self) -> dict:
        """写进 checkpoint / config.json 的快照。"""
        return asdict(self)

    @classmethod
    def from_dict(cls, d: dict) -> "Config":
        return cls(**d)
