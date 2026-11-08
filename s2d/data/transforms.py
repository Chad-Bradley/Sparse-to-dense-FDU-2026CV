"""Minimal transforms for the baseline data interface.

Geometry objects have deterministic parameters so RGB and depth stay aligned.
"""

import cv2
import numpy as np


IMAGENET_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
IMAGENET_STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)


class Resize:
    def __init__(self, size):
        self.size = size

    def __call__(self, image):
        if isinstance(self.size, (tuple, list)):
            height, width = self.size
        else:
            height, width = (int(length * self.size) for length in image.shape[:2])
        if height <= 0 or width <= 0:
            raise ValueError("resize dimensions must be positive")
        return cv2.resize(image, (int(width), int(height)), interpolation=cv2.INTER_NEAREST)


class CenterCrop:
    def __init__(self, size):
        self.height, self.width = size

    def __call__(self, image):
        height, width = image.shape[:2]
        if height < self.height or width < self.width:
            raise ValueError("crop exceeds input dimensions")
        top = (height - self.height) // 2
        left = (width - self.width) // 2
        return image[top:top + self.height, left:left + self.width]


class Rotate:
    def __init__(self, angle):
        self.angle = float(angle)

    def __call__(self, image):
        height, width = image.shape[:2]
        matrix = cv2.getRotationMatrix2D(((width - 1) / 2, (height - 1) / 2), self.angle, 1)
        return cv2.warpAffine(image, matrix, (width, height), flags=cv2.INTER_NEAREST,
                             borderMode=cv2.BORDER_CONSTANT, borderValue=0)


class HorizontalFlip:
    def __init__(self, enabled):
        self.enabled = enabled

    def __call__(self, image):
        return np.flip(image, axis=1) if self.enabled else image


class Normalize:
    def __init__(self, mean=IMAGENET_MEAN, std=IMAGENET_STD):
        self.mean = np.asarray(mean, dtype=np.float32)
        self.std = np.asarray(std, dtype=np.float32)

    def __call__(self, image):
        return (image - self.mean) / self.std


class Lighting:
    def __init__(self, std):
        self.std = float(std)

    def __call__(self, image):
        if self.std == 0:
            return image
        eigenvalues = np.array([0.2175, 0.0188, 0.0045], dtype=np.float32)
        eigenvectors = np.array([[-0.5675, 0.7192, 0.4009],
                                 [-0.5808, -0.0045, -0.8140],
                                 [-0.5836, -0.6948, 0.4203]], dtype=np.float32)
        noise = np.random.normal(0, self.std, 3).astype(np.float32)
        return image + eigenvectors @ (eigenvalues * noise)
