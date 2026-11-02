"""数据管线自检：读一条样本，打印统计并保存可视化。

运行（从 s2d_baseline/ 目录）：

    python -m scripts.debug_sample
    python -m scripts.debug_sample --split train          # 看训练增广
    python -m scripts.debug_sample --modality rgb --index 3

它检查的就是实验要求第 1 条：
    「正确读取 RGB 和深度图，并生成包含 500 个有效点的稀疏深度」
"""

from __future__ import annotations

import argparse
import os

import torch

from s2d.config import Config
from s2d.data.dataset import NYUDataset
from s2d.data.transforms import IMAGENET_MEAN, IMAGENET_STD
from s2d.utils import set_seed
from s2d.viz import make_panel, save_panel


def parse_args():
    default = Config()
    parser = argparse.ArgumentParser(description="NYU 数据管线自检")
    parser.add_argument("--data-root", default=default.data_root)
    parser.add_argument("--split", default="val", choices=["train", "val"])
    parser.add_argument("--index", type=int, default=0)
    parser.add_argument("--modality", default=default.modality,
                        choices=["rgb", "d", "rgbd"])
    parser.add_argument("--num-samples", type=int, default=default.num_samples,
                        help="每张图的稀疏点数（严格，无放回），默认 500")
    parser.add_argument("--output", default="debug_sample.png")
    parser.add_argument("--seed", type=int, default=default.seed)
    return parser.parse_args()


def main():
    args = parse_args()
    root = os.path.join(args.data_root, args.split)

    print("=> 构建数据集: {}".format(root))
    dataset = NYUDataset(
        root,
        split=args.split,
        modality=args.modality,
        num_samples=args.num_samples,
        max_depth=-1.0,
        output_size=(228, 304),
    )

    set_seed(args.seed)
    inp, tgt = dataset[args.index]
    path = dataset.files[args.index]

    # ---- 形状与数值检查 --------------------------------------------------
    c = inp.shape[0]
    assert tuple(inp.shape) == (c, 228, 304), inp.shape
    assert tuple(tgt.shape) == (1, 228, 304), tgt.shape
    assert torch.isfinite(inp).all(), "输入里有 NaN/Inf"
    assert torch.isfinite(tgt).all(), "目标里有 NaN/Inf"

    if args.modality != "d":
        # rgbd 时前 3 通道是 RGB；rgb 模态整张输入都是 RGB
        rgb = inp[:3] if args.modality == "rgbd" else inp
        # 已做 ImageNet 归一化（减均值除标准差），数值大约在 [-2.5, 2.5]，
        # 不再要求 [0, 1]；这里只检查有限且范围合理。
        assert torch.isfinite(rgb).all(), "RGB 里有 NaN/Inf"
        assert -8.0 < float(rgb.min()) and float(rgb.max()) < 8.0, \
            "RGB 归一化后的范围异常: [{:.3f}, {:.3f}]".format(
                float(rgb.min()), float(rgb.max()))

    valid = tgt[tgt > 0]
    assert valid.numel() > 0, "整张图没有有效深度"
    median = float(valid.median())
    assert 0.1 < median < 30.0, (
        "深度中位数 {:.2f} 不在米制范围，检查数据量纲".format(median))

    print("dataset size : {}".format(len(dataset)))
    print("file         : {}".format(path))
    print("input shape  : {}  {}".format(tuple(inp.shape), inp.dtype))
    print("target shape : {}  {}".format(tuple(tgt.shape), tgt.dtype))
    if args.modality != "d":
        print("rgb range    : [{:.3f}, {:.3f}]  (ImageNet 归一化后)".format(
            float(rgb.min()), float(rgb.max())))
    print("depth range  : [{:.2f}, {:.2f}] m, median {:.2f}, 有效 {:.1%}".format(
        float(valid.min()), float(valid.max()), median,
        float(valid.numel()) / tgt.numel()))

    # ---- 稀疏点检查 ------------------------------------------------------
    if args.modality in ("d", "rgbd"):
        sparse = inp[3] if args.modality == "rgbd" else inp[0]
        n_sparse = int((sparse > 0).sum())
        # 稀疏采样固定为无放回恰好 N 个点（N 默认 500），这里必须严格相等
        assert n_sparse == min(args.num_samples, int(valid.numel())), (
            "期望 {} 个稀疏点，实际 {}".format(args.num_samples, n_sparse))
        # 稀疏点的值必须与同一位置的 GT 完全一致（不是插值出来的）
        assert torch.equal(sparse[sparse > 0], tgt[0][sparse > 0]), \
            "稀疏深度与 GT 在同一位置不一致"
        print("sparse points: {}  (严格无放回)".format(n_sparse))

    # ---- 确定性检查 ------------------------------------------------------
    # 固定种子后重新取同一条样本，几何增广与稀疏采样应完全一致。
    # 如果不是，说明有随机数在 dataset 之外被消耗，会导致训练不可复现。
    set_seed(args.seed)
    inp2, tgt2 = dataset[args.index]
    assert torch.equal(inp, inp2) and torch.equal(tgt, tgt2), \
        "固定种子后两次取数结果不一致"

    # ---- 保存可视化 ------------------------------------------------------
    gt_np = tgt[0].numpy()
    if args.modality == "rgbd":
        rgb_np = inp[:3].permute(1, 2, 0).numpy()
        sparse_np = inp[3].numpy()
    elif args.modality == "rgb":
        rgb_np = inp.permute(1, 2, 0).numpy()
        sparse_np = None
    else:
        rgb_np = None
        sparse_np = inp[0].numpy()

    # 模型输入已做 ImageNet 归一化，显示前还原到 [0,1]
    if rgb_np is not None and dataset.normalize is not None:
        rgb_np = rgb_np * IMAGENET_STD + IMAGENET_MEAN

    # pred 先不传：数据阶段没有预测，三联图更直观
    panel = make_panel(rgb_np, sparse_np, gt_np)
    out = save_panel(args.output, panel)
    print("saved        : {}".format(out))
    print("\n数据管线检查通过 ✓")


if __name__ == "__main__":
    main()
