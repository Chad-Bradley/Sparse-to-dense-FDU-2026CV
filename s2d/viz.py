"""深度可视化：四联图与误差图。

用途：
    - scripts/debug_sample.py：检查数据管线的对齐/稀疏点
    - scripts/evaluate.py：训练后展示 RGB | 稀疏 | GT | 预测
    - scripts/analyze.py：配合误差图挑典型失败案例

色阶约定：
    同一张面板里的 GT / 预测共用一组 vmin/vmax，否则两张图颜色不可比。
    vmin/vmax 默认取 GT 有效像素的 1% / 99% 分位数，避免个别离群值把色阶拉爆。
"""

from __future__ import annotations

from pathlib import Path

import cv2
import matplotlib
import numpy as np
from PIL import Image

# matplotlib 3.9+ 用 matplotlib.colormaps；旧版本回退到 cm.get_cmap。
# 不 import pyplot：服务器没有显示环境，而且我们只需要 colormap 函数。
try:
    _CMAP = matplotlib.colormaps["viridis"]
except AttributeError:  # pragma: no cover - 仅为旧版 matplotlib 兼容
    from matplotlib import cm

    _CMAP = cm.get_cmap("viridis")


def _apply_cmap(values, vmin, vmax, mask):
    """把标量图映射成 RGB，mask=False 的位置涂黑。"""
    denom = max(float(vmax) - float(vmin), 1e-6)
    norm = np.clip((values - vmin) / denom, 0.0, 1.0)
    rgb = (_CMAP(norm)[..., :3] * 255.0).astype(np.uint8)
    rgb[~mask] = 0
    return rgb


def colorize_depth(depth, vmin=None, vmax=None):
    """H×W 深度 -> H×W×3 uint8。深度 <= 0 视为无效，涂黑。"""
    depth = np.asarray(depth, dtype=np.float32)
    mask = depth > 0
    if vmin is None or vmax is None:
        if mask.any():
            lo = float(np.percentile(depth[mask], 1))
            hi = float(np.percentile(depth[mask], 99))
        else:
            lo, hi = 0.0, 1.0
        vmin = lo if vmin is None else vmin
        vmax = hi if vmax is None else vmax
    return _apply_cmap(depth, vmin, vmax, mask)


def rgb_float_to_uint8(rgb):
    """H×W×3、[0,1] 浮点 RGB -> uint8。"""
    rgb = np.asarray(rgb, dtype=np.float32)
    return (np.clip(rgb, 0.0, 1.0) * 255.0).astype(np.uint8)


def _put_label(img, text):
    """在左上角写英文标签（中文字体在服务器上不可靠，这里用英文）。"""
    out = np.ascontiguousarray(img.copy())
    pos = (8, 24)
    # 先画黑色粗描边，再画白字，深浅背景上都看得清
    cv2.putText(out, text, pos, cv2.FONT_HERSHEY_SIMPLEX, 0.7,
                (0, 0, 0), 3, cv2.LINE_AA)
    cv2.putText(out, text, pos, cv2.FONT_HERSHEY_SIMPLEX, 0.7,
                (255, 255, 255), 1, cv2.LINE_AA)
    return out


def make_panel(rgb, sparse, gt, pred=None, vmin=None, vmax=None):
    """拼图。

    参数：
        rgb:    H×W×3 float [0,1]，None 时用深灰占位（d 模态没有 RGB）
        sparse: H×W float，None 时用黑图占位
        gt:     H×W float，必须提供
        pred:   H×W float，None 时只输出三联图（调试数据管线时用）

    所有深度面板共用同一组 vmin/vmax，保证 GT 与预测的颜色可直接比较。
    """
    gt = np.asarray(gt, dtype=np.float32)
    h, w = gt.shape

    if vmin is None or vmax is None:
        valid = gt[gt > 0]
        if valid.size:
            vmin = float(np.percentile(valid, 1)) if vmin is None else vmin
            vmax = float(np.percentile(valid, 99)) if vmax is None else vmax
        else:
            vmin = 0.0 if vmin is None else vmin
            vmax = 1.0 if vmax is None else vmax

    if rgb is None:
        rgb_panel = np.full((h, w, 3), 64, dtype=np.uint8)
    else:
        rgb_panel = rgb_float_to_uint8(rgb)

    if sparse is None:
        sparse_panel = np.zeros((h, w, 3), dtype=np.uint8)
    else:
        sparse_panel = colorize_depth(sparse, vmin, vmax)

    panels = [
        _put_label(rgb_panel, "RGB"),
        _put_label(sparse_panel, "SPARSE"),
        _put_label(colorize_depth(gt, vmin, vmax), "GT"),
    ]
    if pred is not None:
        panels.append(_put_label(colorize_depth(pred, vmin, vmax), "PRED"))

    return np.concatenate(panels, axis=1)


def make_error_map(pred, gt):
    """|pred - gt| 的着色图；无效像素黑。"""
    pred = np.asarray(pred, dtype=np.float32)
    gt = np.asarray(gt, dtype=np.float32)
    mask = gt > 0
    err = np.abs(pred - gt)
    vmax = float(np.percentile(err[mask], 99)) if mask.any() else 1.0
    return _apply_cmap(err, 0.0, max(vmax, 1e-6), mask)


def save_panel(path, panel):
    path = Path(path)
    if path.parent:
        path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(panel).save(path)
    return path
