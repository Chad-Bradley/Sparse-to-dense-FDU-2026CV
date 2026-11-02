"""边界 / 无纹理区域分析入口。

运行（从 s2d_baseline/ 目录）：

    python -m scripts.analyze --checkpoint runs/<run>/best.pth.tar
    python -m scripts.analyze --checkpoint ... --bins 5 --stride 4

输出：
    终端两张表 + <checkpoint 目录>/analysis/
        boundary.csv   按 GT 深度梯度分桶
        texture.csv    按 RGB 局部方差分桶

解读：
    boundary 桶从低到高 = 从平坦区域到物体轮廓；
    texture  桶从低到高 = 从无纹理区域（白墙/桌面）到纹理丰富区域。
    如果最后几桶的 RMSE 明显更高，就说明模型在边界处更差。
"""

from __future__ import annotations

import argparse
import csv
import os
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

from s2d.analysis import (
    gradient_magnitude,
    local_variance,
    rgb_to_gray,
    summarize_bins,
)
from s2d.config import Config
from s2d.data.dataset import NYUDataset
from s2d.data.transforms import IMAGENET_MEAN, IMAGENET_STD
from s2d.models.sparse2dense import SparseToDense

IN_CHANNELS = {"rgb": 3, "d": 1, "rgbd": 4}


def parse_args():
    p = argparse.ArgumentParser(description="边界 / 无纹理分桶分析")
    p.add_argument("--checkpoint", required=True)
    p.add_argument("--split", default="val", choices=["train", "val"])
    p.add_argument("--data-root", default="")
    p.add_argument("--output-dir", default="",
                   help="默认 <checkpoint 目录>/analysis")
    p.add_argument("--bins", type=int, default=4)
    p.add_argument("--stride", type=int, default=8,
                   help="每 N 个有效像素取 1 个，控制内存；1 = 全量")
    p.add_argument("--workers", type=int, default=4)
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    return p.parse_args()


def print_table(name, rows):
    print("\n[{}]".format(name))
    print("  {:>3s}  {:>10s}  {:>10s}  {:>9s}  {:>8s}  {:>8s}".format(
        "bin", "lo", "hi", "pixels", "rmse", "absrel"))
    for r in rows:
        print("  {:>3d}  {:>10.4f}  {:>10.4f}  {:>9d}  {:>8.4f}  {:>8.4f}"
              .format(r["bin"], r["lo"], r["hi"], r["n"],
                      r["rmse"], r["absrel"]))


def save_csv(path, rows):
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f, fieldnames=["bin", "lo", "hi", "n", "rmse", "absrel"])
        writer.writeheader()
        writer.writerows(rows)


def main():
    args = parse_args()

    ckpt = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    cfg = Config.from_dict(ckpt["config"])
    data_root = args.data_root or cfg.data_root
    out_dir = (Path(args.output_dir) if args.output_dir
               else Path(args.checkpoint).resolve().parent / "analysis")
    out_dir.mkdir(parents=True, exist_ok=True)

    device = torch.device(args.device)
    dataset = NYUDataset(
        os.path.join(data_root, args.split), split=args.split,
        modality=cfg.modality, num_samples=cfg.num_samples,
        max_depth=cfg.max_depth, sparsifier=cfg.sparsifier,
        output_size=cfg.output_size,
        rgb_normalize=cfg.rgb_normalize, lighting_std=cfg.lighting_std)
    loader = DataLoader(dataset, batch_size=1, shuffle=False,
                        num_workers=args.workers, pin_memory=True)

    model = SparseToDense(
        in_channels=IN_CHANNELS[cfg.modality],
        arch=cfg.arch, decoder=cfg.decoder,
        output_size=cfg.output_size, pretrained=False,
        conv1_init=cfg.conv1_init,
        bottleneck_kernel=cfg.bottleneck_kernel).to(device)
    model.load_state_dict(ckpt["model"])
    model.eval()

    print("=> checkpoint: {}".format(args.checkpoint))
    print("   {} 张 {} 图 | bins={} stride={}".format(
        len(dataset), args.split, args.bins, args.stride))

    boundary_scores, texture_scores = [], []
    sq_errs, abs_rel_errs = [], []

    with torch.no_grad():
        for inp, tgt in loader:
            inp = inp.to(device, non_blocking=True)
            pred = model(inp)
            pred = pred[0, 0].cpu().numpy()
            gt = tgt[0, 0].cpu().numpy()

            mask = gt > 0
            flat = np.flatnonzero(mask)
            if args.stride > 1:
                flat = flat[::args.stride]

            diff = pred - gt
            sq_errs.append((diff[mask] ** 2)[::args.stride])
            abs_rel_errs.append((np.abs(diff[mask]) / gt[mask])[::args.stride])
            boundary_scores.append(gradient_magnitude(gt)[mask][::args.stride])

            if cfg.modality != "d":
                rgb = inp[0, :3].permute(1, 2, 0).cpu().numpy()
                if cfg.rgb_normalize:
                    # 还原到 [0,1] 再算局部方差，否则每通道的 std 会改变纹理尺度
                    rgb = rgb * IMAGENET_STD + IMAGENET_MEAN
                texture_scores.append(
                    local_variance(rgb_to_gray(rgb))[mask][::args.stride])

    sq = np.concatenate(sq_errs)
    rel = np.concatenate(abs_rel_errs)
    n_pix = sq.size
    print("\n=> 全局（抽样后，pixel-wise）: RMSE {:.4f}  AbsRel {:.4f}  像素 {}"
          .format(float(np.sqrt(sq.mean())), float(rel.mean()), n_pix))

    boundary = np.concatenate(boundary_scores)
    rows, _ = summarize_bins(boundary, sq, rel, bins=args.bins)
    print_table("boundary（GT 深度梯度，越大越靠近物体轮廓）", rows)
    save_csv(out_dir / "boundary.csv", rows)

    if texture_scores:
        texture = np.concatenate(texture_scores)
        rows, _ = summarize_bins(texture, sq, rel, bins=args.bins)
        print_table("texture（RGB 局部方差，越小越无纹理）", rows)
        save_csv(out_dir / "texture.csv", rows)
    else:
        print("\n[texture] 跳过：modality=d 时没有 RGB，无法计算纹理")

    print("\n=> 结果写入 {}".format(out_dir))


if __name__ == "__main__":
    main()
