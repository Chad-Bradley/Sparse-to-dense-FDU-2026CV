"""Synthetic checks only. No optimizer steps, data downloads or training runs."""

import unittest
import contextlib
import io
import json
import tempfile
from pathlib import Path
from unittest.mock import patch

import torch

from s2d.config import Config
from s2d.models.invalid_points import InvalidPointAdapter, SparseConv2d
from s2d.models.sparse2dense import SparseToDense
from scripts.task2 import experiments


class TestInvalidPoints(unittest.TestCase):
    def test_encodings_preserve_rgb_and_measurements(self):
        x = torch.zeros(1, 4, 5, 7)
        x[:, :3] = 0.5
        x[:, 3, 2, 3] = 2
        for mode in ("zero", "sentinel", "mask"):
            with self.subTest(mode=mode):
                y = InvalidPointAdapter(4, mode)(x)
                torch.testing.assert_close(y[:, :3], x[:, :3])
                self.assertEqual(y[0, 3, 2, 3], 2)
                self.assertEqual(y[0, 3, 0, 0], -1 if mode == "sentinel" else 0)
                if mode == "mask":
                    self.assertEqual(y[:, 4].sum(), 1)

    def test_masked_average_and_empty_neighborhood(self):
        layer = SparseConv2d(1, 1)
        with torch.no_grad():
            layer.conv.weight.fill_(1)
            layer.bias.fill_(2)
        x = torch.full((1, 1, 7, 7), 100.0)
        mask = torch.zeros_like(x)
        x[:, :, 3, 3] = 4
        mask[:, :, 3, 3] = 1
        y, propagated = layer(x, mask)
        self.assertEqual(y[0, 0, 3, 3], 6)
        self.assertEqual(propagated.sum(), 9)
        self.assertEqual(y[0, 0, 0, 0], 0)
        empty, _ = layer(x, torch.zeros_like(mask))
        self.assertEqual(empty.count_nonzero(), 0)

    def test_nonfinite_invalid_values_are_ignored(self):
        x = torch.zeros(1, 4, 8, 8)
        x[:, 3, 0, 0] = float("nan")
        x[:, 3, 0, 1] = float("inf")
        for mode in ("mask", "sparse", "sentinel"):
            self.assertTrue(torch.isfinite(InvalidPointAdapter(4, mode)(x)).all())

    def test_config_and_old_checkpoint_defaults(self):
        cfg = Config.from_dict({"criterion": "mse", "lighting_std": 0})
        self.assertEqual(cfg.invalid_mode, "zero")
        self.assertFalse(cfg.learned_upsample)
        self.assertEqual(Config.from_dict(cfg.to_dict()).to_dict(), cfg.to_dict())
        for value in (0, 1, float("nan"), float("-inf")):
            with self.assertRaises(ValueError):
                Config(invalid_value=value)
        with self.assertRaises(ValueError):
            Config(modality="rgb", invalid_mode="mask")


class TestTask2Models(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(2)

    def test_matrix_forward_backward_and_reload(self):
        # Small ResNet18 keeps all 24 combinations cheap on CPU.
        for name in experiments():
            _, decoder, head, mode = name.split("_")
            with self.subTest(experiment=name):
                cfg = Config(arch="resnet18", decoder=decoder, pretrained=False,
                             output_size=(33, 45), invalid_mode=mode,
                             learned_upsample=head == "learned")
                model = SparseToDense(**cfg.model_kwargs()).eval()
                x = torch.zeros(1, 4, 33, 45)
                x[:, :3] = torch.rand(1, 3, 33, 45)
                x[:, 3, 10, 10] = 2
                output = model(x)
                self.assertEqual(output.shape, (1, 1, 33, 45))
                self.assertTrue(torch.isfinite(output).all())
                output.square().mean().backward()
                for param in model.parameters():
                    if param.grad is not None:
                        self.assertTrue(torch.isfinite(param.grad).all())
                restored = SparseToDense(**cfg.model_kwargs()).eval()
                restored.load_state_dict(model.state_dict(), strict=True)
                with torch.no_grad():
                    torch.testing.assert_close(restored(x), output)

    def test_empty_sparse_input(self):
        for channels in (1, 4):
            model = SparseToDense(in_channels=channels, arch="resnet18",
                                  pretrained=False, invalid_mode="sparse",
                                  output_size=(32, 40)).eval()
            with torch.no_grad():
                self.assertTrue(torch.isfinite(model(torch.zeros(1, channels, 32, 40))).all())

    def test_resnet50_full_resolution(self):
        cfg = Config(pretrained=False, invalid_mode="sparse", learned_upsample=True)
        model = SparseToDense(**cfg.model_kwargs()).eval()
        x = torch.zeros(1, 4, 228, 304)
        x[:, 3, ::12, ::12] = 2
        with torch.no_grad():
            output = model(x)
        self.assertEqual(output.shape, (1, 1, 228, 304))
        self.assertTrue(torch.isfinite(output).all())

    def test_baseline_state_layout_and_output(self):
        model = SparseToDense(arch="resnet18", pretrained=False).eval()
        self.assertFalse(any(key.startswith("input_adapter.") for key in model.state_dict()))
        x = torch.zeros(1, 4, 32, 40)
        with torch.no_grad():
            original = model.bilinear(model.conv3(model.decoder(
                model.bn2(model.conv2(model.encoder(x))))))
            torch.testing.assert_close(model(x), original, rtol=0, atol=0)

    def test_real_entrypoints_with_synthetic_h5(self):
        import h5py
        import numpy as np
        from scripts import analyze, evaluate, train
        from s2d.data.dataset import NYUDataset
        with tempfile.TemporaryDirectory(dir=Path(__file__).resolve().parents[1]) as directory:
            root = Path(directory)
            scene = root / "val" / "scene"
            scene.mkdir(parents=True)
            with h5py.File(scene / "sample.h5", "w") as sample:
                sample["rgb"] = np.full((3, 480, 640), 128, np.uint8)
                sample["depth"] = np.full((480, 640), 2, np.float32)
            for split in ("train", "val"):
                for modality, channels in (("rgb", 3), ("d", 1), ("rgbd", 4)):
                    dataset = NYUDataset(scene.parent, split=split, modality=modality,
                                         lighting_std=0)
                    np.random.seed(123)
                    inputs, target = dataset[0]
                    np.random.seed(123)
                    repeated, _ = dataset[0]
                    self.assertTrue(torch.equal(inputs, repeated))
                    self.assertEqual(inputs.shape, (channels, 228, 304))
                    if modality != "rgb":
                        valid = inputs[-1] > 0
                        self.assertEqual(valid.sum(), 500)
                        torch.testing.assert_close(inputs[-1][valid], target[0][valid])
            empty_dataset = NYUDataset(scene.parent, split="val", modality="d", num_samples=0)
            self.assertEqual(empty_dataset[0][0].count_nonzero(), 0)
            cfg = Config(data_root=str(root), arch="resnet18", decoder="deconv3",
                         pretrained=False, invalid_mode="sparse", learned_upsample=True,
                         lighting_std=0, criterion="mse")
            # Test CLI parsing without entering the training loop.
            with patch("sys.argv", ["train", "--invalid-mode", "sparse",
                                     "--learned-upsample", "--criterion", "mse",
                                     "--lighting-std", "0"]):
                parsed = train.parse_args()
                self.assertEqual(parsed.invalid_mode, cfg.invalid_mode)
                self.assertTrue(parsed.learned_upsample)
            with patch("sys.argv", ["train", "-a", "resnet50", "-d", "upproj",
                                     "-m", "rgbd", "-s", "500"]):
                parsed = train.parse_args()
                self.assertEqual((parsed.arch, parsed.decoder, parsed.modality, parsed.num_samples),
                                 ("resnet50", "upproj", "rgbd", 500))
            model = SparseToDense(**cfg.model_kwargs())
            checkpoint = root / "model.pth.tar"
            torch.save(dict(config=cfg.to_dict(), model=model.state_dict(), epoch=0), checkpoint)
            common = ["--checkpoint", str(checkpoint), "--workers", "0", "--device", "cpu"]
            with contextlib.redirect_stdout(io.StringIO()):
                with patch("sys.argv", ["evaluate"] + common + ["--save-samples", "0"]):
                    evaluate.main()
                with patch("sys.argv", ["analyze"] + common):
                    analyze.main()
            metrics = json.loads((root / "eval" / "metrics.json").read_text())
            self.assertTrue(np.isfinite(metrics["image"]["rmse"]))
            self.assertTrue((root / "analysis" / "boundary.csv").exists())
            self.assertTrue((root / "analysis" / "texture.csv").exists())


if __name__ == "__main__":
    unittest.main()
