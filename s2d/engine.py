"""训练与评估循环。

与模型、数据、损失解耦：只依赖 loss 可调用对象和 MetricAccumulator。
checkpoint / 日志 / LR 调度属于 scripts/train.py 的职责。
"""

from __future__ import annotations

import time

import torch

from s2d.metrics import MetricAccumulator


def _sync(device):
    """计时前同步 CUDA；CPU 上是空操作。"""
    if device.type == "cuda":
        torch.cuda.synchronize()


def train_one_epoch(model, loader, criterion, optimizer, device,
                    epoch, print_freq=50, reduction="image"):
    """训练一个 epoch，返回该 epoch 的指标字典。

    注意 loss 的累计方式：按**样本数**加权（loss * batch_size 再除以总样本数）。
    loss 本身已经按有效像素取均值，这里只是跨 batch 汇总。
    指标口径由 reduction 决定（image = 原版按图平均，pixel = 按像素累计）。
    """
    model.train()
    acc = MetricAccumulator(reduction)
    total_loss = 0.0
    total_samples = 0

    t_end = time.time()
    for i, (inp, tgt) in enumerate(loader):
        inp = inp.to(device, non_blocking=True)
        tgt = tgt.to(device, non_blocking=True)
        data_time = time.time() - t_end

        _sync(device)
        t0 = time.time()
        pred = model(inp)
        loss = criterion(pred, tgt)
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()
        _sync(device)
        step_time = time.time() - t0

        acc.update(pred.detach(), tgt)
        batch_size = inp.size(0)
        total_loss += float(loss.detach()) * batch_size
        total_samples += batch_size

        if print_freq and (i + 1) % print_freq == 0:
            metrics = acc.compute()
            print("  epoch {} [{}/{}]  loss {:.4f}  rmse {:.3f}  "
                  "absrel {:.3f}  data {:.3f}s  step {:.3f}s".format(
                      epoch, i + 1, len(loader),
                      total_loss / total_samples,
                      metrics["rmse"], metrics["absrel"],
                      data_time, step_time))

        t_end = time.time()

    metrics = acc.compute()
    metrics["loss"] = total_loss / max(total_samples, 1)
    return metrics


@torch.no_grad()
def evaluate(model, loader, device, reduction="image"):
    """在验证/测试集上计算指标。"""
    model.eval()
    acc = MetricAccumulator(reduction)
    for inp, tgt in loader:
        inp = inp.to(device, non_blocking=True)
        tgt = tgt.to(device, non_blocking=True)
        pred = model(inp)
        acc.update(pred, tgt)
    return acc.compute()
