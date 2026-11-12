"""NYU Depth V2 数据集。"""

from __future__ import annotations

import os

import h5py
import numpy as np
import torch
from torch.utils.data import Dataset

# 引入新的采样器
from s2d.data.sparse import UniformSampling, DynamicUniformSampling, GridSampling
# 引入平移变换
from s2d.data.transforms import (
    CenterCrop, ColorJitter, Compose, HorizontalFlip,
    Lighting, Normalize, Resize, Rotate, RandomTranslate
)

def find_h5_files(root):
    """返回 <root>/<scene>/*.h5 的排序列表。

    为什么按 scene 再按文件名排序：
    - 训练时 shuffle=True，顺序本身不重要；但验证集 shuffle=False，
      稳定的顺序保证每次评估用的是同一批图。
    - 与上游 dataloader 的 find_classes/make_dataset 遍历方式一致。
    """
    if not os.path.isdir(root):
        raise FileNotFoundError(
            "数据目录不存在: {}\n"
            "检查 data 软链接是否指向 /root/Test/data".format(root))

    files = []
    for scene in sorted(os.listdir(root)):
        scene_dir = os.path.join(root, scene)
        if not os.path.isdir(scene_dir):
            continue
        for name in sorted(os.listdir(scene_dir)):
            if name.endswith(".h5"):
                files.append(os.path.join(scene_dir, name))
    return files


def load_h5(path):
    """读一个 .h5，返回 (rgb H×W×3 uint8, depth H×W float32)。

    每个样本临时打开文件，读完后由 with 关闭：
    - 多进程 DataLoader 下，长生命周期句柄在 fork 后共享会出问题；
    - h5 文件很小（几十 KB），打开开销可以忽略。
    """
    with h5py.File(path, "r") as f:
        rgb = np.asarray(f["rgb"])          # (3, H, W), uint8
        depth = np.asarray(f["depth"])      # (H, W), float32

    # h5 里 RGB 是 CHW（与上游 h5_loader 的 transpose(1,2,0) 对应）
    rgb = np.transpose(rgb, (1, 2, 0))
    depth = depth.astype(np.float32, copy=False)

    # 防御性清理：转换脚本理论上已经保证 0 = 无效，
    # 但训练前把 NaN/Inf/负值归 0 成本很低，可以避免整轮 loss 变 NaN。
    depth = np.nan_to_num(depth, nan=0.0, posinf=0.0, neginf=0.0)
    depth[depth < 0] = 0.0
    return rgb, depth


class NYUDataset(Dataset):
    """NYU Depth V2。

    返回：
        input : C×228×304 FloatTensor，C = 3(rgb) / 1(d) / 4(rgbd)
        target: 1×228×304 FloatTensor，单位米，0 = 无效
    """

    def __init__(self, root, split, modality="rgbd", num_samples=500,
                 max_depth=-1.0, sparsifier="uniform",
                 output_size=(228, 304), augment=None,
                 rgb_normalize=True, lighting_std=0.1, 
                 # 新增参数：噪声强度
                 depth_noise_std=0.0):
        assert split in ("train", "val"), "split 只能是 train/val"
        assert modality in ("rgb", "d", "rgbd"), \
            "modality 只能是 rgb/d/rgbd"

        self.root = root
        self.split = split
        self.modality = modality
        self.output_size = (int(output_size[0]), int(output_size[1]))
        self.files = find_h5_files(root)
        if not self.files:
            raise RuntimeError("{} 下没有找到任何 .h5".format(root))

        # augment=None 时按 split 自动决定；显式传 bool 可强制验证集也做增广
        # （调试增广本身时有用）。
        self.augment = (split == "train") if augment is None else bool(augment)

        self.color_jitter = ColorJitter(brightness=0.4, contrast=0.4,
                                        saturation=0.4)
        # 与 Lua 原版一致：RGB 先 /255，再做 Lighting（仅训练），
        # 最后做 ImageNet 归一化（训练/验证都做）。
        self.normalize = Normalize() if rgb_normalize else None
        self.lighting = (Lighting(lighting_std)
                         if lighting_std and lighting_std > 0 else None)

        self.depth_noise_std = float(depth_noise_std)

        # rgb 模态不需要稀疏深度（上游也会把 num_samples 强制成 0）。
        self.sparsifier = None
        if modality in ("d", "rgbd") and num_samples > 0:
            max_depth = np.inf if max_depth < 0 else float(max_depth)
            # 根据传入的 sparsifier 名称选择采样器
            if sparsifier == "uniform":
                self.sparsifier = UniformSampling(num_samples, max_depth=max_depth)
            elif sparsifier == "dynamic_uniform":
                self.sparsifier = DynamicUniformSampling(100, 1000, max_depth=max_depth)
            elif sparsifier == "grid":
                self.sparsifier = GridSampling(step=10)
            else:
                raise ValueError("未知的 sparsifier: {}".format(sparsifier))

    def __len__(self):
        return len(self.files)

    # ------------------------------------------------------------------
    # 增广
    # ------------------------------------------------------------------
    def _geometry(self, rgb, depth):
        """对 RGB 和深度施加同一组几何变换。

        随机量（缩放 s、角度、是否翻转）在这里采样一次，
        然后构造确定性的 Compose，分别调用两次。这是保证两幅图对齐的关键。
        """
        if self.augment:
            # 随机缩放 + 旋转 + 翻转，参数范围与上游 nyu_dataloader 一致。
            s = np.random.uniform(1.0, 1.5)
            angle = np.random.uniform(-5.0, 5.0)
            do_flip = np.random.rand() < 0.5
            # 新增：随机平移参数（在旋转、裁剪之前平移）
            tx = np.random.randint(-20, 20)
            ty = np.random.randint(-20, 20)

            # 图像放大 s 倍相当于相机拉近，深度应该除以 s 才能保持物理一致；
            # 否则同一物体在增广后深度和尺寸对不上，网络学到的是错误监督。
            depth = depth / s

            transform = Compose([
                # 先缩到 250 高再旋转：旋转在原尺寸（480×640）上很慢，
                # 这是上游为效率做的预处理。
                Resize(250.0 / 480),
                RandomTranslate(tx, ty), # <--- 插入平移
                Rotate(angle),
                Resize(s),
                CenterCrop(self.output_size),
                HorizontalFlip(do_flip),
            ])
        else:
            # 验证集：只缩放 + 中心裁剪，确定性变换。
            transform = Compose([
                Resize(240.0 / 480),
                CenterCrop(self.output_size),
            ])

        rgb = transform(rgb)
        depth = transform(depth)
        return rgb, depth

    # ------------------------------------------------------------------
    def __getitem__(self, index):
        rgb, depth = load_h5(self.files[index])

        # 1) 几何增广：RGB 与深度共用参数
        rgb, depth = self._geometry(rgb, depth)

        # 2) 颜色抖动只作用于 RGB，且发生在 /255 之前
        if self.augment:
            rgb = self.color_jitter(rgb)

        # 3) RGB 到 [0, 1]；深度保持米制，不做归一化
        rgb = rgb.astype(np.float32) / 255.0
        depth = depth.astype(np.float32, copy=False)

        # 3.1) Lighting + ImageNet 归一化（Lua 原版顺序：jitter -> lighting -> normalize）
        if self.augment and self.lighting is not None:
            rgb = self.lighting(rgb)
        if self.normalize is not None:
            rgb = self.normalize(rgb)

        # 4) 稀疏化：在增广后的 GT 上采样，保证输入和标签严格对应
        if self.sparsifier is not None:
            mask = self.sparsifier(depth)
            sparse = np.zeros_like(depth)
            sparse[mask] = depth[mask]

            # ---- 新增：给稀疏深度点加高斯噪声 ----
            if self.augment and self.depth_noise_std > 0:
                # 只对 mask 为 True 的点加噪声
                noise = np.random.normal(0, self.depth_noise_std, size=sparse.shape)
                sparse[mask] += noise[mask]
                # 噪声可能导致深度变成负数，强制截断为 0（无效）或保留极小正数
                sparse = np.clip(sparse, 0, None)
            # ------------------------------------
            
        else:
            sparse = None

        # 5) 按模态拼通道
        if self.modality == "rgb":
            inp = rgb
        elif self.modality == "d":
            inp = sparse[..., None]
        else:  # rgbd
            inp = np.concatenate([rgb, sparse[..., None]], axis=2)

        # 6) HWC -> CHW，并保证内存连续（np.fliplr 会留下负步长视图）
        inp = np.ascontiguousarray(inp.transpose(2, 0, 1), dtype=np.float32)
        tgt = np.ascontiguousarray(depth[None, ...], dtype=np.float32)

        return torch.from_numpy(inp), torch.from_numpy(tgt)
