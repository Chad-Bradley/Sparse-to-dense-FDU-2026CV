"""评估入口：指标 + 四联图 + 误差图。

运行（从 s2d_baseline/ 目录）：

    python -m scripts.evaluate --checkpoint runs/<run>/best.pth.tar
    python -m scripts.evaluate --checkpoint runs/<run>/best.pth.tar \
        --save-samples 16 --split val

输出：
    <checkpoint 所在目录>/eval/
    ├── metrics.json
    ├── sample_000.png     RGB | SPARSE | GT | PRED
    ├── error_000.png      |PRED - GT| 误差图
    └── ...

稀疏输入、预测深度、真实深度都画在同一张图里，方便直接放进报告。
"""

from __future__ import annotations

import argparse
import os
from pathlib import Path

import torch
from torch.utils.data import DataLoader

from s2d.config import Config
from s2d.data.dataset import NYUDataset
from s2d.data.transforms import IMAGENET_MEAN, IMAGENET_STD
from s2d.metrics import MetricAccumulator
from s2d.models.factory import build_model
from s2d.utils import save_json
from s2d.viz import make_error_map, make_panel, save_panel

IN_CHANNELS = {"rgb": 3, "d": 1, "rgbd": 4}


def parse_args():
    p = argparse.ArgumentParser(description="评估并保存深度可视化")
    p.add_argument("--checkpoint", required=True)
    p.add_argument("--split", default="val", choices=["train", "val"])
    p.add_argument("--data-root", default="",
                   help="默认使用 checkpoint 里保存的 data_root")
    p.add_argument("--output-dir", default="",
                   help="默认 <checkpoint 目录>/eval")
    p.add_argument("--save-samples", type=int, default=8,
                   help="保存多少张四联图，0 = 不保存")
    p.add_argument("--workers", type=int, default=4)
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--prediction", choices=["refined", "coarse"], default="refined",
                   help="Evaluate refinement or the same checkpoint's coarse output")
    return p.parse_args()


def extract_views(inp, cfg):
    """从模型输入里拆出 RGB 与稀疏深度（batch size 必须为 1）。"""
    rgb = None
    sparse = None
    if cfg.modality in ("rgb", "rgbd"):
        rgb = inp[0, :3].permute(1, 2, 0).cpu().numpy()
        if cfg.rgb_normalize:
            # 模型输入是 ImageNet 归一化后的；显示前还原到 [0,1]
            rgb = rgb * IMAGENET_STD + IMAGENET_MEAN
    if cfg.modality == "rgbd":
        sparse = inp[0, 3].cpu().numpy()
    elif cfg.modality == "d":
        sparse = inp[0, 0].cpu().numpy()
    return rgb, sparse


def main():
    args = parse_args()

    ckpt = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    cfg = Config.from_dict(ckpt["config"])
    data_root = args.data_root or cfg.data_root

    out_dir = (Path(args.output_dir) if args.output_dir
               else Path(args.checkpoint).resolve().parent / (
                   "eval" if args.prediction == "refined" else "eval_coarse"))
    out_dir.mkdir(parents=True, exist_ok=True)

    device = torch.device(args.device)
    dataset = NYUDataset(
        os.path.join(data_root, args.split), split=args.split,
        modality=cfg.modality, num_samples=cfg.num_samples,
        max_depth=cfg.max_depth, sparsifier=cfg.sparsifier,
        output_size=cfg.output_size,
        rgb_normalize=cfg.rgb_normalize, lighting_std=cfg.lighting_std,
        val_sampling_seed=cfg.val_sampling_seed)
    # batch 固定 1：方便逐张取 RGB/稀疏图，和上游验证设置也一致
    loader = DataLoader(dataset, batch_size=1, shuffle=False,
                        num_workers=args.workers, pin_memory=True)

    model = build_model(cfg).to(device)
    model.load_state_dict(ckpt["model"])
    model.eval()

    print("=> checkpoint: {}".format(args.checkpoint))
    print("   epoch {} | {} + {} | {} 张 {} 图".format(
        int(ckpt["epoch"]) + 1, cfg.arch, cfg.decoder,
        len(dataset), args.split))

    # 两套口径同时累计：
    #   image = 原版 Lua / PyTorch 仓库（验证 batch=1，按图平均）
    #   pixel = 常见论文口径（按像素累计）
    acc_img = MetricAccumulator("image")
    acc_pix = MetricAccumulator("pixel")
    saved = 0
    with torch.no_grad():
        for inp, tgt in loader:
            inp = inp.to(device, non_blocking=True)
            tgt = tgt.to(device, non_blocking=True)
            pred = model.forward_outputs(inp)[args.prediction]
            acc_img.update(pred, tgt)
            acc_pix.update(pred, tgt)

            if saved < args.save_samples:
                rgb, sparse = extract_views(inp, cfg)
                gt = tgt[0, 0].cpu().numpy()
                pr = pred[0, 0].cpu().numpy()
                save_panel(out_dir / "sample_{:03d}.png".format(saved),
                           make_panel(rgb, sparse, gt, pr))
                save_panel(out_dir / "error_{:03d}.png".format(saved),
                           make_error_map(pr, gt))
                saved += 1

    metrics = {
        "image": acc_img.compute(),
        "pixel": acc_pix.compute(),
    }
    save_json(metrics, out_dir / "metrics.json")

    for name, desc in (("image", "按图平均（原版口径，可与 0.220 对比）"),
                       ("pixel", "按像素累计（标准论文口径）")):
        m = metrics[name]
        print("\n=> 指标 [{}] {}".format(name, desc))
        print("   N       = {}".format(m["n"]))
        print("   RMSE    = {:.4f}".format(m["rmse"]))
        print("   AbsRel  = {:.4f}".format(m["absrel"]))
        print("   MAE     = {:.4f}".format(m["mae"]))
        print("   lg10    = {:.4f}".format(m["lg10"]))
        print("   delta1  = {:.4f}".format(m["delta1"]))
        print("   delta2  = {:.4f}".format(m["delta2"]))
        print("   delta3  = {:.4f}".format(m["delta3"]))

    print("\n   metrics : {}".format(out_dir / "metrics.json"))
    if saved:
        print("   可视化  : {}".format(out_dir / "sample_*.png"))
        print("   误差图  : {}".format(out_dir / "error_*.png"))


if __name__ == "__main__":
    main()
