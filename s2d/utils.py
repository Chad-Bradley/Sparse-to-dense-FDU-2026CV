"""通用工具：随机种子、目录、JSON、DataLoader worker 初始化。"""

from __future__ import annotations

import json
import random
from pathlib import Path

import numpy as np
import torch


def set_seed(seed: int) -> None:
    """固定所有随机源。

    注意：这里只保证 Python / NumPy / PyTorch 的随机数一致。
    CUDA 卷积本身仍有非确定性（cudnn benchmark / 原子加），
    如果要做严格可复现实验，还需要：
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    但那会牺牲速度，默认不开，在实验记录里注明即可。
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def worker_init_fn(worker_id: int) -> None:
    """让每个 DataLoader worker 拥有不同的 NumPy 随机流。

    不设置的话，fork 出来的 worker 会继承父进程同一个种子，
    多个 worker 的稀疏采样/增广会完全一致，等于浪费了并行采样。
    把 worker_id 混进 PyTorch 的 base seed，保证每轮、每个 worker 都不同。
    """
    base_seed = torch.initial_seed() % (2 ** 32)
    np.random.seed((base_seed + worker_id) % (2 ** 32))


def ensure_dir(path) -> Path:
    path = Path(path)
    path.mkdir(parents=True, exist_ok=True)
    return path


def save_json(obj, path) -> None:
    path = Path(path)
    if path.parent:
        path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2, ensure_ascii=False)
