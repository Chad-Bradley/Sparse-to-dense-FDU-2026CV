"""NYU HDF5 adapter implementing the API used by the GitHub baseline.

Implemented locally for missing baseline sources. No ZIP-specific improvement
flags are provided. Inputs are RGB / sparse depth / their concatenation.
"""

from pathlib import Path

import h5py
import numpy as np
from PIL import Image
import torch
from torch.utils.data import Dataset
from torchvision.transforms import functional as TF

from s2d.data.sparse import UniformSampling
from s2d.data.transforms import CenterCrop, HorizontalFlip, Lighting, Normalize, Resize, Rotate


class NYUDataset(Dataset):
    def __init__(self, root, split, modality="rgbd", num_samples=500,
                 max_depth=-1.0, sparsifier="uniform", output_size=(228, 304),
                 rgb_normalize=True, lighting_std=0.1):
        if split not in ("train", "val") or modality not in ("rgb", "d", "rgbd"):
            raise ValueError("unsupported split or modality")
        if sparsifier != "uniform":
            raise ValueError("baseline supports uniform sampling")
        root = Path(root)
        if not root.is_dir():
            raise FileNotFoundError("NYU data directory missing: {}".format(root))
        self.files = [str(path) for path in sorted(root.glob("*/*.h5"))]
        if not self.files:
            raise ValueError("no scene/*.h5 samples found in {}".format(root))
        self.split = split
        self.modality = modality
        self.output_size = tuple(output_size)
        self.normalize = Normalize() if rgb_normalize else None
        self.lighting = Lighting(lighting_std)
        self.sampling = UniformSampling(num_samples, np.inf if max_depth < 0 else max_depth)

    def __len__(self):
        return len(self.files)

    def __getitem__(self, index):
        with h5py.File(self.files[index], "r") as sample:
            rgb = np.asarray(sample["rgb"])
            depth = np.asarray(sample["depth"], dtype=np.float32)
        if rgb.ndim != 3 or rgb.shape[0] != 3 or depth.shape != rgb.shape[1:]:
            raise ValueError("expected aligned CHW RGB and HW depth")
        rgb = rgb.transpose(1, 2, 0)
        depth = np.where(np.isfinite(depth) & (depth > 0), depth, 0).astype(np.float32)
        if self.split == "train":
            scale = np.random.uniform(1, 1.5)
            depth = depth / scale
            geometry = [Resize(250 / 480), Rotate(np.random.uniform(-5, 5)), Resize(scale),
                        CenterCrop(self.output_size), HorizontalFlip(np.random.random() < 0.5)]
        else:
            geometry = [Resize(0.5), CenterCrop(self.output_size)]
        for operation in geometry:
            rgb, depth = operation(rgb), operation(depth)
        if self.split == "train":
            # NumPy controls all augmentation randomness, matching worker_init_fn.
            image = Image.fromarray(np.ascontiguousarray(rgb))
            adjustments = [TF.adjust_brightness, TF.adjust_contrast, TF.adjust_saturation]
            for adjustment in np.random.permutation(adjustments):
                image = adjustment(image, float(np.random.uniform(0.6, 1.4)))
            rgb = np.asarray(image)
        rgb = rgb.astype(np.float32) / 255
        if self.split == "train":
            rgb = self.lighting(rgb)
        if self.normalize is not None:
            rgb = self.normalize(rgb)
        sparse = np.where(self.sampling(depth), depth, 0)
        if self.modality == "rgb":
            inputs = rgb
        elif self.modality == "d":
            inputs = sparse[..., None]
        else:
            inputs = np.concatenate((rgb, sparse[..., None]), axis=2)
        return (torch.from_numpy(np.ascontiguousarray(inputs.transpose(2, 0, 1), dtype=np.float32)),
                torch.from_numpy(np.ascontiguousarray(depth[None], dtype=np.float32)))
