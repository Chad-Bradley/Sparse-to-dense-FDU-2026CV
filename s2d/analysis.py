"""边界 / 无纹理区域分析。

两个“难度指标”：

    boundary（边界强度）: GT 深度的梯度幅值。值越大 = 越靠近物体轮廓。
    texture （纹理丰富度）: RGB 灰度的局部方差。值越小 = 越无纹理
                             （白墙、桌面、地板等）。

把有效像素按这两个 score 的**全局分位数**分桶，统计各桶的 RMSE / AbsRel，
就能回答「模型在边界和无纹理区域是不是更差」。
"""

from __future__ import annotations

import numpy as np
import cv2


def rgb_to_gray(rgb):
    """H×W×3 -> H×W 灰度。输入可以是 [0,1] 浮点或 0~255 uint8。"""
    rgb = np.asarray(rgb, dtype=np.float32)
    if rgb.ndim != 3 or rgb.shape[2] != 3:
        raise ValueError("rgb 必须是 H×W×3，实际 {}".format(rgb.shape))
    if float(rgb.max()) > 1.5:          # 看起来是 0~255
        rgb = rgb / 255.0
    return 0.2989 * rgb[..., 0] + 0.587 * rgb[..., 1] + 0.114 * rgb[..., 2]


def gradient_magnitude(depth, ksize=3):
    """GT 深度的梯度幅值 sqrt(gx² + gy²)，H×W float32。

    用 Sobel 而不是 cv2.Canny：Canny 带非极大值抑制和二值化，
    会把「边界强度」变成 0/1 掩码，丢掉了强弱信息。
    """
    depth = np.asarray(depth, dtype=np.float32)
    gx = cv2.Sobel(depth, cv2.CV_32F, 1, 0, ksize=ksize)
    gy = cv2.Sobel(depth, cv2.CV_32F, 0, 1, ksize=ksize)
    return np.sqrt(gx * gx + gy * gy)


def local_variance(gray, ksize=7):
    """局部方差（均值滤波的平方 - 平方的均值），H×W float32。

    无纹理区域（白墙等）方差接近 0；有纹理的区域方差大。
    BORDER_REFLECT 避免图像边缘因为补零而出现虚假的“高方差”。
    """
    gray = np.asarray(gray, dtype=np.float32)
    mean = cv2.boxFilter(gray, -1, (ksize, ksize),
                         normalize=True, borderType=cv2.BORDER_REFLECT)
    mean_sq = cv2.boxFilter(gray * gray, -1, (ksize, ksize),
                            normalize=True, borderType=cv2.BORDER_REFLECT)
    return np.maximum(mean_sq - mean * mean, 0.0)


def quantile_edges(values, bins):
    """按全局分位数求桶边界。

    用全体验证像素的分位数，而不是逐图分位数——否则不同图之间的
    「高边界」「低纹理」不可比。
    """
    values = np.asarray(values, dtype=np.float64)
    if values.size == 0:
        return np.array([0.0, 1.0])
    edges = np.quantile(values, np.linspace(0.0, 1.0, bins + 1))
    # 如果 score 大量重复（例如整幅图梯度都为 0），unique 后桶会变少；
    # 这比人为造出空桶更好，下面按实际 edges 返回。
    return np.unique(edges)


def summarize_bins(score, sq_err, abs_rel_err, bins=4, edges=None):
    """对已展平、已对齐的数组分桶统计。

    参数：
        score:        每个像素的难度分数（如梯度幅值）
        sq_err:       (pred - gt)^2
        abs_rel_err:  |pred - gt| / gt
        edges:        可选，外部传入的分桶边界；None 时按全局分位数计算

    返回：
        (rows, edges)
        rows[b] = {"bin", "lo", "hi", "n", "rmse", "absrel"}
    """
    score = np.asarray(score, dtype=np.float64)
    sq_err = np.asarray(sq_err, dtype=np.float64)
    abs_rel_err = np.asarray(abs_rel_err, dtype=np.float64)
    assert score.shape == sq_err.shape == abs_rel_err.shape

    if edges is None:
        edges = quantile_edges(score, bins)
    edges = np.asarray(edges, dtype=np.float64)
    if edges.size < 2:
        edges = np.array([edges[0], edges[0] + 1e-6] if edges.size else [0.0, 1.0])

    # 内部边界决定分桶；最右边界用 <=，保证最大值也有归属
    idx = np.digitize(score, edges[1:-1])

    rows = []
    for b in range(len(edges) - 1):
        sel = idx == b
        n = int(sel.sum())
        row = {
            "bin": b,
            "lo": float(edges[b]),
            "hi": float(edges[b + 1]),
            "n": n,
            "rmse": float("nan"),
            "absrel": float("nan"),
        }
        if n > 0:
            row["rmse"] = float(np.sqrt(sq_err[sel].mean()))
            row["absrel"] = float(abs_rel_err[sel].mean())
        rows.append(row)
    return rows, edges


def stratified_metrics(pred, target, score, bins=4):
    """单张图的分桶指标（便于调试和单测）。

    pred / target / score: H×W。
    """
    pred = np.asarray(pred, dtype=np.float32)
    target = np.asarray(target, dtype=np.float32)
    score = np.asarray(score, dtype=np.float32)
    mask = target > 0
    diff = pred[mask] - target[mask]
    sq = diff ** 2
    abs_rel = np.abs(diff) / target[mask]
    return summarize_bins(score[mask], sq, abs_rel, bins=bins)
