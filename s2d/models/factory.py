"""One config-to-model mapping for training, evaluation and analysis."""
from s2d.models.sparse2dense import SparseToDense

IN_CHANNELS = {"rgb": 3, "d": 1, "rgbd": 4}


def build_model(cfg, pretrained=False):
    return SparseToDense(
        in_channels=IN_CHANNELS[cfg.modality], arch=cfg.arch, decoder=cfg.decoder,
        output_size=cfg.output_size, pretrained=pretrained,
        conv1_init=cfg.conv1_init, bottleneck_kernel=cfg.bottleneck_kernel,
        multiscale=cfg.multiscale, unet_skips=cfg.unet_skips, refiner=cfg.refiner,
        prop_kernel=cfg.prop_kernel, prop_steps=cfg.prop_steps,
        preserve_sparse=cfg.preserve_sparse)
