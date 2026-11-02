"""评测指标：支持两种聚合口径。

reduction="image"（默认，与原始 Lua / PyTorch 仓库一致）
    每个 batch 内先算一次 RMSE，再按 batchSize 加权平均。
    NYU 验证时 batch=1（原版 `main.lua:32-35` 强制这么做），
    所以最终 = 每张图 RMSE 的算术平均。

reduction="pixel"（常见的论文口径）
    累加所有有效像素的平方误差，最后统一开方：
        RMSE = sqrt(Σ(pred-gt)² / N)
    对每张图的权重正比于其有效像素数。

两者数值不同（见 FORMULAS.md 第 3 节），与 baseline 0.220 对比时
必须用 image 口径。scripts/evaluate.py 会同时报告两套。
"""

from __future__ import annotations

import math

import torch

# 预测值 clamp 的下界。只用于 lg10 和 δ 的比值，避免训练早期
# pred <= 0 时 log10 产生 NaN。
_EPS = 1e-6


def _per_image_stats(pred, target):
    """单张图的指标；没有有效像素时返回 None。"""
    mask = target > 0
    if not mask.any():
        return None

    p = pred[mask]
    t = target[mask]
    diff = p - t

    mse = float((diff ** 2).mean().item())
    mae = float(diff.abs().mean().item())
    absrel = float((diff.abs() / t).mean().item())

    p_safe = p.clamp_min(_EPS)
    lg10 = float(
        (torch.log10(p_safe) - torch.log10(t)).abs().mean().item())
    ratio = torch.maximum(p_safe / t, t / p_safe)
    deltas = [float((ratio < 1.25 ** (k + 1)).float().mean().item())
              for k in range(3)]

    return {
        "mse": mse,
        "rmse": math.sqrt(mse),
        "mae": mae,
        "absrel": absrel,
        "lg10": lg10,
        "delta1": deltas[0],
        "delta2": deltas[1],
        "delta3": deltas[2],
    }


class MetricAccumulator:
    """跨 batch 累加指标。

    n 的含义随 reduction 变化：
        image -> 参与统计的图片数
        pixel -> 有效像素总数
    """

    def __init__(self, reduction="pixel"):
        assert reduction in ("image", "pixel"), \
            "reduction 只能是 image / pixel"
        self.reduction = reduction
        self.reset()

    def reset(self):
        self.n = 0

        # ---- pixel 模式累计 ----
        self.px_sq = 0.0
        self.px_abs = 0.0
        self.px_absrel = 0.0
        self.px_lg10 = 0.0
        self.px_delta = [0, 0, 0]

        # ---- image 模式累计（每张图指标值之和）----
        self.img_mse = 0.0
        self.img_rmse = 0.0
        self.img_mae = 0.0
        self.img_absrel = 0.0
        self.img_lg10 = 0.0
        self.img_delta = [0.0, 0.0, 0.0]

    @torch.no_grad()
    def update(self, pred, target):
        """pred, target: 任意前导维度相同的张量，通常为 B×1×H×W。"""
        assert pred.shape == target.shape, \
            "pred {} 与 target {} 形状不一致".format(pred.shape, target.shape)
        if self.reduction == "pixel":
            self._update_pixel(pred, target)
        else:
            self._update_image(pred, target)

    # ------------------------------------------------------------------
    def _update_pixel(self, pred, target):
        mask = target > 0
        n = int(mask.sum().item())
        if n == 0:
            return

        p = pred[mask]
        t = target[mask]
        diff = p - t

        self.n += n
        self.px_sq += float((diff ** 2).sum().item())
        self.px_abs += float(diff.abs().sum().item())
        self.px_absrel += float((diff.abs() / t).sum().item())

        p_safe = p.clamp_min(_EPS)
        self.px_lg10 += float(
            (torch.log10(p_safe) - torch.log10(t)).abs().sum().item())

        ratio = torch.maximum(p_safe / t, t / p_safe)
        for k in range(3):
            self.px_delta[k] += int((ratio < 1.25 ** (k + 1)).sum().item())

    def _update_image(self, pred, target):
        """逐张图计算指标再累加——对应原版 Lua 在 batch=1 下的行为。

        注意：原版在 batch>1 时是一个 batch 算一次 RMSE 再按 batchSize 加权，
        与真正的逐图平均并不严格等价（原版注释也承认 batch>1 会 overestimate）。
        这里实现的是严格逐图平均；要保持和原版完全一致，验证时用 batch=1。
        """
        for b in range(pred.shape[0]):
            stats = _per_image_stats(pred[b], target[b])
            if stats is None:
                # 原版会把这种图算成 NaN 并污染整轮平均；这里选择跳过，
                # 但实际上 NYU 不会出现整张图无有效像素的情况。
                continue
            self.n += 1
            self.img_mse += stats["mse"]
            self.img_rmse += stats["rmse"]
            self.img_mae += stats["mae"]
            self.img_absrel += stats["absrel"]
            self.img_lg10 += stats["lg10"]
            for k in range(3):
                self.img_delta[k] += stats["delta{}".format(k + 1)]

    # ------------------------------------------------------------------
    def compute(self):
        if self.n == 0:
            return {
                "reduction": self.reduction, "n": 0,
                "mse": float("nan"), "rmse": float("nan"),
                "mae": float("nan"), "absrel": float("nan"),
                "lg10": float("nan"),
                "delta1": float("nan"), "delta2": float("nan"),
                "delta3": float("nan"),
            }

        n = self.n
        if self.reduction == "pixel":
            mse = self.px_sq / n
            return {
                "reduction": "pixel",
                "n": n,                       # 有效像素数
                "mse": mse,
                "rmse": math.sqrt(mse),
                "mae": self.px_abs / n,
                "absrel": self.px_absrel / n,
                "lg10": self.px_lg10 / n,
                "delta1": self.px_delta[0] / n,
                "delta2": self.px_delta[1] / n,
                "delta3": self.px_delta[2] / n,
            }

        return {
            "reduction": "image",
            "n": n,                           # 图片数
            "mse": self.img_mse / n,
            "rmse": self.img_rmse / n,        # 注意：不是 sqrt(mean mse)
            "mae": self.img_mae / n,
            "absrel": self.img_absrel / n,
            "lg10": self.img_lg10 / n,
            "delta1": self.img_delta[0] / n,
            "delta2": self.img_delta[1] / n,
            "delta3": self.img_delta[2] / n,
        }


@torch.no_grad()
def compute_metrics(pred, target, reduction="pixel"):
    """单 batch 的便捷入口（测试和调试用）。"""
    acc = MetricAccumulator(reduction)
    acc.update(pred, target)
    return acc.compute()
