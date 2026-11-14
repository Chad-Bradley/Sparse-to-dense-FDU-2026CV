"""Architecture ablations and stage transitions, without downloads or NYU data."""
import contextlib
import csv
import io
import itertools
import tempfile
import unittest
from pathlib import Path
from unittest import mock

import h5py
import numpy as np
import torch

from s2d.config import Config
from s2d.data.dataset import NYUDataset
from s2d.losses import build_criterion
from s2d.models.enhancements import CSPNLite
from s2d.models.factory import build_model
from s2d.models.sparse2dense import SparseToDense
from s2d.training import (initialize_weights, make_optimizer, stage_for_epoch,
                          stage_lr, capture_rng, restore_rng)
from scripts.train import parse_args, save_checkpoint


class TestEnhancements(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.previous_threads = torch.get_num_threads()
        torch.set_num_threads(2)

    @classmethod
    def tearDownClass(cls):
        torch.set_num_threads(cls.previous_threads)

    def make_input(self, h=64, w=80):
        x = torch.randn(2, 4, h, w)
        x[:, 3] = 0
        x[:, 3, ::16, ::16] = 2
        return x

    def test_disabled_matches_legacy_graph_and_weights(self):
        model = SparseToDense(arch="resnet18", pretrained=False, output_size=(65, 81)).eval()
        x = self.make_input(65, 81)
        with torch.no_grad():
            legacy = model.bilinear(model.conv3(model.decoder(model.bn2(model.conv2(model.encoder(x))))))
            self.assertTrue(torch.equal(model(x), legacy))
            self.assertTrue(torch.equal(model.forward_outputs(x)["coarse"], legacy))
        self.assertFalse(any(k.startswith(("context.", "skip_fusions.", "refiner."))
                             for k in model.state_dict()))
        clone = SparseToDense(arch="resnet18", pretrained=False, output_size=(65, 81))
        clone.load_state_dict(model.state_dict(), strict=True)
        legacy_config = Config.from_dict({"arch": "resnet18", "pretrained": False})
        self.assertEqual(legacy_config.refiner, "none")

    def test_all_eight_combinations_and_backward(self):
        x = self.make_input(65, 81)
        for ms, skip, refine in itertools.product((False, True), repeat=3):
            with self.subTest(ms=ms, skip=skip, refine=refine):
                model = SparseToDense(
                    arch="resnet18", pretrained=False, output_size=(65, 81),
                    multiscale="aspp" if ms else "none", unet_skips=skip,
                    refiner="cspn-lite" if refine else "none")
                y = model(x)
                self.assertEqual(tuple(y.shape), (2, 1, 65, 81))
                self.assertTrue(torch.isfinite(y).all())
                y.square().mean().backward()
                self.assertIsNotNone(model.encoder.layer1[0].conv1.weight.grad)
                if skip:
                    self.assertIsNotNone(model.skip_fusions[0].project.weight.grad)
                if ms:
                    self.assertIsNotNone(model.context.branches[0][0].weight.grad)
                if refine:
                    self.assertIsNotNone(model.refiner.guidance[-1].weight.grad)

    def test_resnet50_full_model_and_feature_channels(self):
        model = SparseToDense(arch="resnet50", pretrained=False, output_size=(65, 81),
                              multiscale="aspp", unet_skips=True, refiner="cspn-lite").eval()
        with torch.no_grad():
            x = self.make_input(65, 81)[:1]
            self.assertEqual(model(x).shape, (1, 1, 65, 81))
            features = model.encoder.forward_features(x)
        self.assertEqual(tuple(f.shape[1] for f in features), (64, 256, 512, 1024, 2048))

    def test_refiner_modalities(self):
        for channels in (1, 3, 4):
            model = SparseToDense(in_channels=channels, arch="resnet18", pretrained=False,
                                  output_size=(64, 80), refiner="cspn-lite").eval()
            with torch.no_grad():
                self.assertEqual(model(torch.zeros(1, channels, 64, 80)).shape, (1, 1, 64, 80))

    def test_propagation_identity_anchors_and_edges(self):
        coarse = torch.ones(1, 1, 5, 7)
        features = torch.randn(1, 4, 3, 4)
        head = CSPNLite(4)
        self.assertTrue(torch.equal(head(coarse, features), coarse))
        sparse = torch.zeros_like(coarse)
        sparse[..., 0, 0] = 5
        out = head(coarse, features, sparse)
        self.assertEqual(out[..., 0, 0].item(), 5)
        head.preserve_sparse = False
        self.assertTrue(torch.equal(head(coarse, features, sparse), coarse))
        # Nonzero affinities must preserve constants at boundaries too.
        with torch.no_grad():
            head.guidance[-1].bias.fill_(0.25)
        self.assertTrue(torch.allclose(head(coarse, features), coarse, atol=1e-6))
        head(coarse.requires_grad_(), features).sum().backward()
        self.assertTrue(torch.isfinite(coarse.grad).all())
        for kernel, steps in ((2, 3), (1, 3), (3, 0)):
            with self.assertRaises(ValueError):
                CSPNLite(4, kernel, steps)
        with self.assertRaises(ValueError):
            head(coarse, features, torch.zeros(1, 1, 6, 7))

    def test_frozen_backbone_including_bn_and_joint_update(self):
        cfg = Config(arch="resnet18", output_size=(64, 80), pretrained=False,
                     refiner="cspn-lite", multiscale="aspp", unet_skips=True,
                     refiner_only_epochs=1)
        model = build_model(cfg)
        optimizer = make_optimizer(model, cfg, 0)
        before = {k: v.clone() for k, v in model.state_dict().items()}
        x = self.make_input()
        model.train()
        loss = build_criterion("l1")(model(x), torch.ones(2, 1, 64, 80))
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        self.assertTrue(all(torch.equal(v, before[k]) for k, v in model.state_dict().items()
                            if not k.startswith("refiner.")))
        self.assertFalse(torch.equal(model.refiner.guidance[-1].weight, before["refiner.guidance.2.weight"]))
        optimizer = make_optimizer(model, cfg, 1)
        model.train()
        optimizer.zero_grad()
        model(x).square().mean().backward()
        optimizer.step()
        self.assertFalse(torch.equal(model.encoder.conv1.weight, before["encoder.conv1.weight"]))
        self.assertTrue(model.encoder.bn1.training)

    def test_initialization_rejects_wrong_backbone_and_keys(self):
        cfg = Config(arch="resnet18", pretrained=False, output_size=(64, 80))
        coarse = build_model(cfg)
        ckpt = {"config": cfg.to_dict(), "model": coarse.state_dict()}
        target = Config.from_dict({**cfg.to_dict(), "refiner": "cspn-lite"})
        refined = build_model(target)
        report = initialize_weights(refined, target, ckpt)
        self.assertTrue(report["new_refiner_keys"])
        self.assertTrue(torch.equal(refined.conv2.weight, coarse.conv2.weight))
        with self.assertRaises(ValueError):
            initialize_weights(refined, Config.from_dict({**target.to_dict(), "unet_skips": True}), ckpt)
        bad = {**ckpt, "model": dict(ckpt["model"])}
        bad["model"].pop("encoder.conv1.weight")
        with self.assertRaises(ValueError):
            initialize_weights(refined, target, bad)

    def test_resume_inside_stage_and_across_boundary(self):
        cfg = Config(arch="resnet18", pretrained=False, output_size=(64, 80),
                     refiner="cspn-lite", refiner_only_epochs=1, epochs=3)
        model = build_model(cfg)
        optimizer = make_optimizer(model, cfg, 0)
        x = self.make_input()

        def step(net, opt):
            net.train()
            opt.zero_grad()
            net(x).square().mean().backward()
            opt.step()

        step(model, optimizer)
        for saved_epoch in (0, 1):
            with self.subTest(saved_epoch=saved_epoch), tempfile.TemporaryDirectory() as directory:
                path = Path(directory) / "last.pth.tar"
                save_checkpoint(path, model, optimizer, saved_epoch, 1.0, cfg)
                ckpt = torch.load(path, weights_only=False)
                resumed = build_model(Config.from_dict(ckpt["config"]))
                resumed.load_state_dict(ckpt["model"])
                resumed_opt = make_optimizer(resumed, cfg, saved_epoch)
                resumed_opt.load_state_dict(ckpt["optimizer"])
                self.assertEqual(ckpt["stage"], stage_for_epoch(cfg, saved_epoch))
                if stage_for_epoch(cfg, saved_epoch + 1) != ckpt["stage"]:
                    optimizer = make_optimizer(model, cfg, saved_epoch + 1)
                    resumed_opt = make_optimizer(resumed, cfg, saved_epoch + 1)
                step(model, optimizer)
                step(resumed, resumed_opt)
                for key, value in model.state_dict().items():
                    self.assertTrue(torch.equal(value, resumed.state_dict()[key]), key)
        state = capture_rng()
        expected = (torch.rand(3), np.random.rand(3))
        restore_rng(state)
        self.assertTrue(torch.equal(torch.rand(3), expected[0]))
        np.testing.assert_array_equal(np.random.rand(3), expected[1])

    def test_cli_validation_and_schedule(self):
        args = parse_args(["-a", "resnet18", "-d", "upproj", "-m", "rgbd", "-s", "200"])
        self.assertEqual(args.num_samples, 200)
        with contextlib.redirect_stderr(io.StringIO()), self.assertRaises(SystemExit):
            parse_args(["--resume", "old", "--init-checkpoint", "new"])
        with contextlib.redirect_stderr(io.StringIO()), self.assertRaises(SystemExit):
            parse_args(["--refiner", "cspn-lite"])
        cfg = Config(refiner="cspn-lite")
        self.assertEqual(stage_for_epoch(cfg, 2), "refiner_only")
        self.assertEqual(stage_for_epoch(cfg, 3), "joint")
        self.assertEqual(stage_lr(cfg, 3), cfg.joint_lr)
        with self.assertRaises(ValueError):
            Config(unet_skips=True, decoder="deconv3")

    def test_ablation_commands_use_matching_initialization(self):
        from scripts.ablate import experiment_commands
        from types import SimpleNamespace
        args = SimpleNamespace(data_root="nyu", out_dir="runs", arch="resnet50",
                               seed=0, val_sampling_seed=0, batch_size=8, workers=0,
                               no_pretrain=True, limit_train=4, limit_val=2,
                               prefix="check", backbone_epochs=1, refiner_epochs=1,
                               joint_epochs=1)
        commands = experiment_commands(args)
        self.assertEqual(len(commands), 12)
        configs = {}
        for name, command in commands:
            values = vars(parse_args(command[3:])).copy()
            for runtime in ("resume", "limit_train", "limit_val"):
                values.pop(runtime)
            configs[name] = Config(**values)
        combinations = set()
        for name, cfg in configs.items():
            combinations.add((cfg.multiscale, cfg.unet_skips, cfg.refiner))
            if cfg.init_checkpoint:
                source_name = Path(cfg.init_checkpoint).parent.name
                source = configs[source_name]
                self.assertEqual((source.multiscale, source.unet_skips),
                                 (cfg.multiscale, cfg.unet_skips))
        self.assertEqual(len(combinations), 8)

    def test_fixed_validation_sampling_and_zero_points(self):
        with tempfile.TemporaryDirectory() as directory:
            scene = Path(directory) / "scene"
            scene.mkdir()
            with h5py.File(scene / "sample.h5", "w") as f:
                f["rgb"] = np.zeros((3, 480, 640), dtype=np.uint8)
                f["depth"] = np.full((480, 640), 2, dtype=np.float32)
            ds = NYUDataset(directory, split="val")
            state = np.random.get_state()
            first, _ = ds[0]
            after = np.random.rand(3)
            np.random.set_state(state)
            np.testing.assert_array_equal(np.random.rand(3), after)
            np.random.seed(876)
            second, _ = ds[0]
            self.assertTrue(torch.equal(first, second))
            self.assertEqual((first[3] > 0).sum().item(), 500)
            empty, _ = NYUDataset(directory, split="val", num_samples=0)[0]
            self.assertEqual(empty[3].count_nonzero().item(), 0)

    @unittest.skipUnless(torch.cuda.is_available(), "CUDA not available")
    def test_cuda_full_model_backward(self):
        model = SparseToDense(arch="resnet18", pretrained=False, output_size=(64, 80),
                              multiscale="aspp", unet_skips=True, refiner="cspn-lite").cuda()
        output = model(self.make_input().cuda())
        output.square().mean().backward()
        torch.cuda.synchronize()
        self.assertTrue(torch.isfinite(output).all())
        self.assertTrue(torch.isfinite(model.encoder.conv1.weight.grad).all())

    def test_training_entrypoint_resume_evaluation_and_analysis(self):
        """Exercise real H5 loaders and driver I/O using temporary fixture images."""
        from scripts import train, evaluate, analyze
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            data = root / "data"
            for split, count in (("train", 2), ("val", 1)):
                scene = data / split / "scene"
                scene.mkdir(parents=True)
                for index in range(count):
                    with h5py.File(scene / (str(index) + ".h5"), "w") as f:
                        f["rgb"] = np.random.RandomState(index).randint(
                            0, 255, (3, 480, 640), dtype=np.uint8)
                        f["depth"] = np.full((480, 640), 2, dtype=np.float32)
            common = ["--data-root", str(data), "--out-dir", str(root / "runs"),
                      "--arch", "resnet18", "--batch-size", "2", "--workers", "0",
                      "--no-pretrain", "--print-freq", "0"]

            def run_train(argv):
                args = parse_args(argv)
                # Small internal Config size keeps this driver test fast.
                args.output_size = (64, 80)
                with mock.patch.object(train, "parse_args", return_value=args):
                    train.main()

            with mock.patch("torch.cuda.is_available", return_value=False), contextlib.redirect_stdout(io.StringIO()):
                run_train(common + ["--run-name", "backbone", "--epochs", "1"])
                initial = root / "runs/backbone/best.pth.tar"
                original_epoch = train.train_one_epoch

                def interrupt_joint(*args, **kwargs):
                    if kwargs["epoch"] == 1:
                        raise RuntimeError("simulated interruption")
                    return original_epoch(*args, **kwargs)

                with mock.patch.object(train, "train_one_epoch", side_effect=interrupt_joint):
                    with self.assertRaisesRegex(RuntimeError, "simulated interruption"):
                        run_train(common + ["--run-name", "refined", "--epochs", "2",
                                             "--init-checkpoint", str(initial), "--refiner", "cspn-lite",
                                             "--refiner-only-epochs", "1"])
                last = root / "runs/refined/last.pth.tar"
                self.assertEqual(torch.load(last, weights_only=False)["stage"], "refiner_only")
                run_train(["--resume", str(last)])
                resumed = torch.load(last, weights_only=False)
                self.assertEqual(resumed["stage"], "joint")
                self.assertEqual(resumed["epoch"], 1)
                with open(root / "runs/refined/metrics.csv", newline="", encoding="utf-8") as f:
                    self.assertEqual([row["stage"] for row in csv.DictReader(f)], ["refiner_only", "joint"])
                for prediction in ("coarse", "refined"):
                    with mock.patch("sys.argv", ["evaluate", "--checkpoint", str(last),
                                                 "--workers", "0", "--device", "cpu",
                                                 "--save-samples", "0", "--prediction", prediction]):
                        evaluate.main()
                with mock.patch("sys.argv", ["analyze", "--checkpoint", str(last),
                                             "--workers", "0", "--device", "cpu", "--bins", "2"]):
                    analyze.main()
                self.assertTrue((root / "runs/refined/eval/metrics.json").is_file())
                self.assertTrue((root / "runs/refined/eval_coarse/metrics.json").is_file())
                self.assertTrue((root / "runs/refined/analysis/boundary.csv").is_file())


if __name__ == "__main__":
    unittest.main()
