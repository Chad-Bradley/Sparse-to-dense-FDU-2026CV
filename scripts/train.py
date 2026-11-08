"""训练入口。

运行（从 s2d_baseline/ 目录）：

    python -m scripts.train                      # 默认 upproj（Lua 原版）
    python -m scripts.train --decoder deconv3    # PyTorch baseline 那套
    python -m scripts.train --num-samples 500

    # 端到端快速验证（64 张训练图、2 个 epoch，几十秒）
    python -m scripts.train --limit-train 64 --limit-val 64 \
        --epochs 2 --workers 0

输出目录：

    <out_dir>/<run_name>/
    ├── config.json        配置快照
    ├── metrics.csv        每 epoch 一行
    ├── last.pth.tar       最新 epoch
    └── best.pth.tar       验证 RMSE 最小

恢复训练：

    python -m scripts.train --resume runs/<run>/last.pth.tar

注意：恢复时以 checkpoint 里保存的 config 为准，命令行参数会被忽略
（这是刻意的——避免出现上游那种「改了 --epochs 但被 checkpoint 覆盖」的困惑）。
"""

from __future__ import annotations

import argparse
import csv
import os
import sys
import time
from pathlib import Path

import torch
from torch.utils.data import DataLoader, Subset

from s2d.config import Config
from s2d.data.dataset import NYUDataset
from s2d.engine import evaluate, train_one_epoch
from s2d.losses import build_criterion
from s2d.models.sparse2dense import SparseToDense
from s2d.utils import save_json, set_seed, worker_init_fn

IN_CHANNELS = {"rgb": 3, "d": 1, "rgbd": 4}

CSV_FIELDS = [
    "epoch", "lr",
    "train_loss", "train_rmse", "train_absrel",
    "val_rmse", "val_absrel", "val_mae", "val_lg10",
    "val_delta1", "val_delta2", "val_delta3",
]


def parse_args():
    d = Config()
    p = argparse.ArgumentParser(description="Sparse-to-Dense 独立复现训练")
    # 数据
    p.add_argument("--data-root", default=d.data_root)
    p.add_argument("--modality", "-m", default=d.modality,
                   choices=["rgb", "d", "rgbd"])
    p.add_argument("--num-samples", "-s", type=int, default=d.num_samples)
    p.add_argument("--max-depth", type=float, default=d.max_depth)
    p.add_argument("--sparsifier", default=d.sparsifier, choices=["uniform"])
    p.add_argument("--no-rgb-normalize", dest="rgb_normalize",
                   action="store_false", default=d.rgb_normalize,
                   help="关闭 ImageNet RGB 归一化（默认开启）")
    p.add_argument("--lighting-std", type=float, default=d.lighting_std,
                   help="AlexNet PCA 光照噪声强度，0 = 关闭（默认 0.1）")
    # 模型
    p.add_argument("--arch", "-a", default=d.arch, choices=["resnet18", "resnet50"])
    p.add_argument("--decoder", "-d", default=d.decoder,
                   choices=["deconv2", "deconv3", "upconv", "upproj"])
    p.add_argument("--conv1-init", default=d.conv1_init,
                   choices=["random", "copy"])
    p.add_argument("--learned-upsample", action="store_true", default=d.learned_upsample)
    p.add_argument("--invalid-mode", default=d.invalid_mode,
                   choices=["zero", "sentinel", "mask", "sparse"])
    p.add_argument("--invalid-value", type=float, default=d.invalid_value)
    p.add_argument("--bottleneck-kernel", type=int,
                   default=d.bottleneck_kernel)
    p.add_argument("--no-pretrain", dest="pretrained", action="store_false",
                   default=d.pretrained, help="不加载 ImageNet 权重")
    # 优化
    p.add_argument("--epochs", type=int, default=d.epochs)
    p.add_argument("--batch-size", type=int, default=d.batch_size)
    p.add_argument("--eval-batch-size", type=int, default=d.eval_batch_size)
    p.add_argument("--lr", type=float, default=d.lr)
    p.add_argument("--momentum", type=float, default=d.momentum)
    p.add_argument("--weight-decay", type=float, default=d.weight_decay)
    p.add_argument("--lr-step", type=int, default=d.lr_step)
    p.add_argument("--lr-gamma", type=float, default=d.lr_gamma)
    p.add_argument("--criterion", default=d.criterion,
                   choices=["l1", "l2", "mse"])
    p.add_argument("--metric-reduction", default=d.metric_reduction,
                   choices=["image", "pixel"],
                   help="image = 原版按图平均（默认）；pixel = 按像素累计")
    p.add_argument("--workers", type=int, default=d.workers)
    p.add_argument("--seed", type=int, default=d.seed)
    # 输出
    p.add_argument("--out-dir", default=d.out_dir)
    p.add_argument("--run-name", default=d.run_name)
    p.add_argument("--print-freq", type=int, default=d.print_freq)
    p.add_argument("--save-samples", type=int, default=d.save_samples)
    # 仅用于快速验证的临时截断（0 = 全量），不会写进 config
    p.add_argument("--limit-train", type=int, default=0,
                   help="只用前 N 张训练图（0 = 全部，试跑用）")
    p.add_argument("--limit-val", type=int, default=0,
                   help="只用前 N 张验证图（0 = 全部，试跑用）")
    # 恢复
    p.add_argument("--resume", default="", metavar="PATH")
    return p.parse_args()


def adjust_lr(optimizer, epoch, base_lr, step, gamma):
    """复刻上游 utils.py:80-84 的阶梯学习率。

        lr(e) = base_lr * gamma ** (e // step)

    刻意不用 StepLR：它的 step 时机容易和「epoch 从 0 开始」错开一格，
    这里显式按 epoch 计算，和 baseline 完全一致（epoch 5、10 各降一次）。
    """
    lr = base_lr * (gamma ** (epoch // step))
    for group in optimizer.param_groups:
        group["lr"] = lr
    return lr


def build_loaders(cfg, limit_train=0, limit_val=0):
    common = dict(
        modality=cfg.modality,
        num_samples=cfg.num_samples,
        max_depth=cfg.max_depth,
        sparsifier=cfg.sparsifier,
        output_size=cfg.output_size,
        rgb_normalize=cfg.rgb_normalize,
        lighting_std=cfg.lighting_std,
    )
    train_ds = NYUDataset(os.path.join(cfg.data_root, "train"),
                          split="train", **common)
    val_ds = NYUDataset(os.path.join(cfg.data_root, "val"),
                        split="val", **common)

    # 只对「试跑」有意义：截断数据集，避免为了一次形状检查跑满 5948 个 iter
    if limit_train:
        train_ds = Subset(train_ds, range(min(limit_train, len(train_ds))))
    if limit_val:
        val_ds = Subset(val_ds, range(min(limit_val, len(val_ds))))

    train_loader = DataLoader(
        train_ds, batch_size=cfg.batch_size, shuffle=True,
        num_workers=cfg.workers, pin_memory=True,
        worker_init_fn=worker_init_fn, drop_last=False)
    val_loader = DataLoader(
        val_ds, batch_size=cfg.eval_batch_size, shuffle=False,
        num_workers=cfg.workers, pin_memory=True,
        worker_init_fn=worker_init_fn)
    return train_ds, train_loader, val_ds, val_loader


def save_checkpoint(path, model, optimizer, epoch, best_rmse, cfg):
    torch.save({
        "model": model.state_dict(),
        "optimizer": optimizer.state_dict(),
        "epoch": epoch,                 # 0-based
        "best_rmse": best_rmse,
        "config": cfg.to_dict(),        # dict，不是 argparse Namespace
    }, path)


def append_csv(path, row, fields):
    write_header = not path.exists()
    with open(path, "a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        if write_header:
            writer.writeheader()
        writer.writerow(row)


def main():
    args = parse_args()
    resume = args.resume
    limit_train = args.limit_train
    limit_val = args.limit_val

    # ------------------------------------------------------------------
    # 配置与输出目录
    # ------------------------------------------------------------------
    if resume:
        if not os.path.isfile(resume):
            raise FileNotFoundError("checkpoint 不存在: {}".format(resume))
        # torch 2.5 默认 weights_only=False，这里显式写出来
        ckpt = torch.load(resume, map_location="cpu", weights_only=False)
        cfg = Config.from_dict(ckpt["config"])
        run_dir = Path(resume).resolve().parent
        start_epoch = int(ckpt["epoch"]) + 1
        best_rmse = float(ckpt["best_rmse"])
        print("=> 从 {} 恢复（已完成 epoch {}）".format(
            resume, int(ckpt["epoch"]) + 1))
        print("   恢复时使用 checkpoint 内的配置，命令行参数被忽略")
    else:
        values = vars(args).copy()
        values.pop("resume")
        values.pop("limit_train")   # 试跑参数不进入正式 config
        values.pop("limit_val")
        cfg = Config(**values)

        run_name = cfg.run_name or "{}_{}_{}_s{}_{}".format(
            cfg.arch, cfg.decoder, cfg.modality,
            cfg.num_samples, cfg.criterion)
        if not cfg.run_name:
            run_name += "_{}{}".format(
                cfg.invalid_mode, "_learned" if cfg.learned_upsample else "")
        run_dir = Path(cfg.out_dir) / run_name
        if run_dir.exists():
            # 不覆盖旧实验：同名目录存在时追加时间戳
            run_dir = Path(cfg.out_dir) / (
                run_name + "_" + time.strftime("%Y%m%d-%H%M%S"))
        run_dir.mkdir(parents=True, exist_ok=True)
        save_json(cfg.to_dict(), run_dir / "config.json")
        start_epoch = 0
        best_rmse = float("inf")

    print("=> 输出目录: {}".format(run_dir))
    print("=> 配置: {}".format(cfg.to_dict()))

    set_seed(cfg.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("=> device: {}".format(device))
    print("=> 指标口径: {}（image = 原版按图平均，pixel = 按像素累计）"
          .format(cfg.metric_reduction))

    # ------------------------------------------------------------------
    # 数据
    # ------------------------------------------------------------------
    train_ds, train_loader, val_ds, val_loader = build_loaders(
        cfg, limit_train, limit_val)
    print("=> train {} 张 / val {} 张".format(len(train_ds), len(val_ds)))
    if limit_train or limit_val:
        print("   ! 使用了 --limit-train/--limit-val，这是试跑，不是正式实验")

    # ------------------------------------------------------------------
    # 模型 / 损失 / 优化器
    # ------------------------------------------------------------------
    # 恢复时不加载 ImageNet 权重：反正马上会被 checkpoint 覆盖，
    # 还能避免构建阶段联网下载。
    model = SparseToDense(**cfg.model_kwargs(
        pretrained=(cfg.pretrained and not resume))).to(device)

    criterion = build_criterion(cfg.criterion).to(device)
    optimizer = torch.optim.SGD(
        model.parameters(), lr=cfg.lr,
        momentum=cfg.momentum, weight_decay=cfg.weight_decay)

    if resume:
        model.load_state_dict(ckpt["model"])
        optimizer.load_state_dict(ckpt["optimizer"])

    n_params = sum(p.numel() for p in model.parameters()) / 1e6
    print("=> 模型: {} + {}  {:.1f}M 参数".format(
        cfg.arch, cfg.decoder, n_params))

    # ------------------------------------------------------------------
    # 训练
    # ------------------------------------------------------------------
    csv_path = run_dir / "metrics.csv"
    for epoch in range(start_epoch, cfg.epochs):
        lr = adjust_lr(optimizer, epoch, cfg.lr, cfg.lr_step, cfg.lr_gamma)
        print("\n=> Epoch {}/{}  lr {:.2e}".format(
            epoch + 1, cfg.epochs, lr))

        t0 = time.time()
        train_m = train_one_epoch(
            model, train_loader, criterion, optimizer, device,
            epoch=epoch, print_freq=cfg.print_freq,
            reduction=cfg.metric_reduction)
        # 验证 batch 固定为 1 时，image 口径与原始 Lua 完全一致
        val_m = evaluate(model, val_loader, device,
                         reduction=cfg.metric_reduction)
        elapsed = time.time() - t0

        is_best = val_m["rmse"] < best_rmse
        if is_best:
            best_rmse = val_m["rmse"]

        append_csv(csv_path, {
            "epoch": epoch + 1,
            "lr": lr,
            "train_loss": train_m["loss"],
            "train_rmse": train_m["rmse"],
            "train_absrel": train_m["absrel"],
            "val_rmse": val_m["rmse"],
            "val_absrel": val_m["absrel"],
            "val_mae": val_m["mae"],
            "val_lg10": val_m["lg10"],
            "val_delta1": val_m["delta1"],
            "val_delta2": val_m["delta2"],
            "val_delta3": val_m["delta3"],
        }, CSV_FIELDS)

        save_checkpoint(run_dir / "last.pth.tar", model, optimizer,
                        epoch, best_rmse, cfg)
        if is_best:
            save_checkpoint(run_dir / "best.pth.tar", model, optimizer,
                            epoch, best_rmse, cfg)

        print("   train loss {:.4f}  rmse {:.4f}  absrel {:.4f}".format(
            train_m["loss"], train_m["rmse"], train_m["absrel"]))
        print("   val   rmse {:.4f}  absrel {:.4f}  mae {:.4f}  "
              "delta1 {:.4f}  ({:.0f}s){}".format(
                  val_m["rmse"], val_m["absrel"], val_m["mae"],
                  val_m["delta1"], elapsed, "  *best" if is_best else ""))

    print("\n=> 训练结束")
    print("   输出目录 : {}".format(run_dir))
    print("   最佳 RMSE: {:.4f}".format(best_rmse))
    print("   best     : {}".format(run_dir / "best.pth.tar"))


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n中断：当前 epoch 未完成，last.pth.tar 是上一个 epoch 的检查点。")
        sys.exit(130)
