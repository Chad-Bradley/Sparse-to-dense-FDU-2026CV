"""Training stages, checked weight initialization and reproducible checkpoints."""
from __future__ import annotations

import random

import numpy as np
import torch

from s2d.config import Config


def stage_for_epoch(cfg, epoch):
    if cfg.refiner == "none":
        return "finetune" if cfg.init_checkpoint else "backbone"
    return "refiner_only" if epoch < cfg.refiner_only_epochs else "joint"


def stage_lr(cfg, epoch):
    stage = stage_for_epoch(cfg, epoch)
    if stage == "refiner_only":
        return cfg.refiner_lr * cfg.lr_gamma ** (epoch // cfg.lr_step)
    offset = cfg.refiner_only_epochs if stage == "joint" else 0
    base = cfg.lr if stage == "backbone" else cfg.joint_lr
    return base * cfg.lr_gamma ** ((epoch - offset) // cfg.lr_step)


def make_optimizer(model, cfg, epoch):
    model.set_backbone_frozen(stage_for_epoch(cfg, epoch) == "refiner_only")
    return torch.optim.SGD([p for p in model.parameters() if p.requires_grad],
                           lr=stage_lr(cfg, epoch), momentum=cfg.momentum,
                           weight_decay=cfg.weight_decay)


def initialize_weights(model, cfg, checkpoint):
    """Start a new experiment; missing/unexpected backbone keys are fatal."""
    source = Config.from_dict(checkpoint["config"])
    backbone_fields = ("arch", "decoder", "modality", "output_size", "conv1_init",
                       "bottleneck_kernel", "multiscale", "unet_skips")
    mismatch = [name for name in backbone_fields if getattr(cfg, name) != getattr(source, name)]
    if mismatch:
        raise ValueError("Initialization backbone config mismatch: " + ", ".join(mismatch))
    if source.refiner != "none":
        raise ValueError("--init-checkpoint expects a coarse backbone checkpoint; use --resume for refinement")
    current = model.state_dict()
    incoming = checkpoint["model"]
    missing = sorted(set(current) - set(incoming))
    unexpected = sorted(set(incoming) - set(current))
    if unexpected or any(not key.startswith("refiner.") for key in missing):
        raise ValueError("Initialization keys differ: missing={}, unexpected={}".format(missing, unexpected))
    for key, value in incoming.items():
        if value.shape != current[key].shape:
            raise ValueError("Initialization tensor shape differs: " + key)
    model.load_state_dict(incoming, strict=not missing)
    return {"loaded": len(incoming), "new_refiner_keys": missing}


def capture_rng(loader=None):
    return {
        "python": random.getstate(), "numpy": np.random.get_state(),
        "torch": torch.get_rng_state(),
        "cuda": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None,
        "loader": loader.generator.get_state() if loader is not None else None,
    }


def restore_rng(state, loader=None):
    if not state:
        return
    random.setstate(state["python"])
    np.random.set_state(state["numpy"])
    torch.set_rng_state(state["torch"])
    if state["cuda"] is not None and torch.cuda.is_available():
        torch.cuda.set_rng_state_all(state["cuda"])
    if loader is not None and state["loader"] is not None:
        loader.generator.set_state(state["loader"])
