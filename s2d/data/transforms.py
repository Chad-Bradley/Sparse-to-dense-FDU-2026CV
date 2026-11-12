"""几何增广与颜色抖动。

设计原则（也是和上游对齐的地方）：

1. **随机参数在 dataset 里采样一次**，再构造确定性的 transform 对象，
   分别作用于 RGB 和深度。如果在每个算子的 __call__ 里各自随机，
   RGB 和深度的几何变换会不一致，物体轮廓立刻错位。
2. **阶段 A 全部用最近邻**：上游的 scipy.misc.imresize 默认 interp='nearest'，
   旋转也用 order=0。对深度图来说最近邻是必须的（双线性会插出原本不存在的
   深度值）；对 RGB 用最近邻是刻意模仿上游，阶段 B 可以再换双线性做对照。
3. 所有算子同时支持 (H, W) 深度图和 (H, W, 3) RGB，靠 ndim 区分。
"""

"""在末尾增加了平移。"""


from __future__ import annotations

import cv2
import numpy as np
from PIL import Image
from torchvision.transforms import functional as TF

# 字符串到 OpenCV 插值标志的映射。只暴露阶段 A 需要的两种；
# 阶段 B 想给 RGB 换双线性时，直接把 Resize(..., interpolation="bilinear") 即可。
_INTERPOLATION = {
    "nearest": cv2.INTER_NEAREST,
    "bilinear": cv2.INTER_LINEAR,
}


class Compose:
    """按顺序执行一组变换。"""

    def __init__(self, transforms):
        self.transforms = list(transforms)

    def __call__(self, img):
        for t in self.transforms:
            img = t(img)
        return img


class Resize:
    """缩放。

    size:
        float -> 缩放倍数（0.5 表示缩小一半）
        tuple -> 目标尺寸 (H, W)

    注意 cv2.resize 的 dsize 参数顺序是 (宽, 高)，与 numpy 的 (高, 宽) 相反，
    这里显式写成 (new_w, new_h) 防止写反。
    """

    def __init__(self, size, interpolation="nearest"):
        assert isinstance(size, (int, float, tuple, list)), \
            "size 必须是缩放倍数或 (H, W)"
        self.size = size
        self.interpolation = interpolation

    def __call__(self, img):
        h, w = img.shape[:2]
        if isinstance(self.size, (tuple, list)):
            new_h, new_w = int(self.size[0]), int(self.size[1])
        else:
            # 与上游一致：用 int() 截断，而不是 round；保证 250/480 这类
            # 比例在 480×640 上得到 250×333，后续 CenterCrop 才够切。
            new_h, new_w = int(h * self.size), int(w * self.size)
        # 至少保留 1 像素，防止极端缩放得到 0 尺寸
        new_h, new_w = max(1, new_h), max(1, new_w)
        return cv2.resize(img, (new_w, new_h),
                          interpolation=_INTERPOLATION[self.interpolation])


class Rotate:
    """绕图像中心旋转，保持原尺寸（reshape=False），边界补 0。

    用 getRotationMatrix2D + warpAffine 而不是 scipy.ndimage.rotate：
    scipy.ndimage.interpolation 在新版 SciPy 里已经移除，而 cv2 没有这个问题。
    边界补 0 对深度图意味着「无效像素」，后续 depth>0 的掩码会自然过滤掉。
    """

    def __init__(self, angle, interpolation="nearest"):
        self.angle = float(angle)
        self.interpolation = interpolation

    def __call__(self, img):
        h, w = img.shape[:2]
        # 用 (w/2 - 0.5, h/2 - 0.5) 是像素中心的正确表达；
        # 直接用 (w/2, h/2) 会有半个像素的整体偏移。
        center = (w / 2.0 - 0.5, h / 2.0 - 0.5)
        matrix = cv2.getRotationMatrix2D(center, self.angle, 1.0)
        return cv2.warpAffine(
            img, matrix, (w, h),
            flags=_INTERPOLATION[self.interpolation],
            borderMode=cv2.BORDER_CONSTANT,
            borderValue=0,
        )


class CenterCrop:
    """中心裁剪到 (H, W)。

    训练集的 Resize 倍数 s 最小是 1.0，此时 250×333 仍大于 228×304，
    所以中心裁剪不会越界；这里仍然做一次检查，避免尺寸配错时切出空数组。
    """

    def __init__(self, size):
        self.size = (int(size[0]), int(size[1]))

    def __call__(self, img):
        h, w = img.shape[:2]
        crop_h, crop_w = self.size
        if h < crop_h or w < crop_w:
            raise ValueError(
                "CenterCrop: 输入 {}x{} 小于目标 {}x{}".format(
                    h, w, crop_h, crop_w))
        top = (h - crop_h) // 2
        left = (w - crop_w) // 2
        return img[top:top + crop_h, left:left + crop_w]


class HorizontalFlip:
    """水平翻转。

    do_flip 由 dataset 在采样阶段决定并传入，保证 RGB 与深度同时翻/同时不翻。
    """

    def __init__(self, do_flip):
        self.do_flip = bool(do_flip)

    def __call__(self, img):
        # np.fliplr 返回的是负步长视图，dataset 转 tensor 前会 ascontiguousarray
        return np.fliplr(img) if self.do_flip else img


# ImageNet 统计量，与 Lua 原版的 Dataset.meanstdRGB 完全一致
# （datasets/dataset.lua:130-134）。
IMAGENET_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
IMAGENET_STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)


class Normalize:
    """ImageNet 逐通道标准化，输入 H×W×3 浮点图。

    对应 Lua `datasets/transforms.lua:29-37` 的 ColorNormalize：
        x[i] = (x[i] - mean[i]) / std[i]
    原版在训练和验证时都做，所以这里也不区分 split。
    """

    def __init__(self, mean=IMAGENET_MEAN, std=IMAGENET_STD):
        self.mean = np.asarray(mean, dtype=np.float32)
        self.std = np.asarray(std, dtype=np.float32)

    def __call__(self, img):
        assert img.ndim == 3 and img.shape[2] == 3, \
            "Normalize 只接受 H×W×3 的 RGB 图像"
        return (img - self.mean) / self.std


class Lighting:
    """AlexNet 式 PCA 光照噪声，输入 H×W×3 浮点图。

    对应 Lua `datasets/transforms.lua:212-231`：
        alpha ~ N(0, alphastd)          # 三个主成分的系数
        offset = eigvec @ (alpha * eigval)
        img[c] += offset[c]

    注意 offset 在整张图上是**空间恒定**的：它模拟的是整体光照/色温偏移，
    不是逐像素噪声。只在训练时使用。
    """

    # AlexNet 的 RGB PCA 统计量（Lua datasets/dataset.lua:135-142）
    EIGVAL = np.array([0.2175, 0.0188, 0.0045], dtype=np.float32)
    EIGVEC = np.array([
        [-0.5675, 0.7192, 0.4009],
        [-0.5808, -0.0045, -0.8140],
        [-0.5836, -0.6948, 0.4203],
    ], dtype=np.float32)

    def __init__(self, alphastd=0.1):
        self.alphastd = float(alphastd)

    def __call__(self, img):
        if self.alphastd == 0.0:
            return img
        alpha = np.random.normal(0.0, self.alphastd, size=3).astype(np.float32)
        offset = self.EIGVEC @ (alpha * self.EIGVAL)   # (3,)
        return img + offset                            # HWC 广播


class ColorJitter:
    """随机亮度/对比度/饱和度抖动，只用于 RGB。

    只作用于 uint8 的 H×W×3 图像，必须在 /255 之前调用：
    上游也是先 jitter 再归一化，这样亮度因子的物理意义才一致。
    三个因子的取值范围和上游一致（[max(0, 1-a), 1+a]），执行顺序随机。
    """

    def __init__(self, brightness=0.4, contrast=0.4, saturation=0.4):
        self.brightness = brightness
        self.contrast = contrast
        self.saturation = saturation

    def __call__(self, img):
        assert img.ndim == 3 and img.shape[2] == 3, \
            "ColorJitter 只接受 H×W×3 的 RGB 图像"
        # 上一步可能是 HorizontalFlip，留下负步长视图；PIL/cv2 对非连续数组
        # 处理不一致，这里统一转成连续内存，避免偶发的 "unexpected layout"。
        pil = Image.fromarray(np.ascontiguousarray(img))

        ops = []
        if self.brightness > 0:
            factor = np.random.uniform(max(0.0, 1 - self.brightness),
                                       1 + self.brightness)
            ops.append(lambda im, f=factor: TF.adjust_brightness(im, f))
        if self.contrast > 0:
            factor = np.random.uniform(max(0.0, 1 - self.contrast),
                                       1 + self.contrast)
            ops.append(lambda im, f=factor: TF.adjust_contrast(im, f))
        if self.saturation > 0:
            factor = np.random.uniform(max(0.0, 1 - self.saturation),
                                       1 + self.saturation)
            ops.append(lambda im, f=factor: TF.adjust_saturation(im, f))

        np.random.shuffle(ops)          # 与上游一样随机顺序
        for op in ops:
            pil = op(pil)
        return np.array(pil)

class RandomTranslate:
    """随机平移 (tx, ty)。
       对深度图来说，平移出去的区域补 0（无效像素）。
    """
    def __init__(self, tx, ty):
        self.tx = int(tx)
        self.ty = int(ty)

    def __call__(self, img):
        h, w = img.shape[:2]
        matrix = np.float32([[1, 0, self.tx], [0, 1, self.ty]])
        
        # 深度图如果是单通道 (H, W)，cv2.warpAffine 需要特殊处理
        if img.ndim == 2:
            return cv2.warpAffine(
                img, matrix, (w, h),
                flags=cv2.INTER_NEAREST,
                borderMode=cv2.BORDER_CONSTANT,
                borderValue=0
            )
        else:
            return cv2.warpAffine(
                img, matrix, (w, h),
                flags=cv2.INTER_NEAREST,
                borderMode=cv2.BORDER_CONSTANT,
                borderValue=0
            )