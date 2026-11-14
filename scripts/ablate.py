"""Print (default) or execute eight ablations plus four continued-training controls."""
from __future__ import annotations

import argparse
import itertools
import shlex
import subprocess
import sys
from pathlib import Path

from s2d.config import Config


def experiment_commands(args):
    commands = []
    common = [sys.executable, "-m", "scripts.train", "--data-root", str(args.data_root),
              "--out-dir", str(args.out_dir), "--arch", args.arch, "--decoder", "upproj",
              "--modality", "rgbd", "--num-samples", "500", "--seed", str(args.seed),
              "--val-sampling-seed", str(args.val_sampling_seed),
              "--batch-size", str(args.batch_size), "--workers", str(args.workers),
              "--metric-reduction", "image", "--criterion", "l1"]
    if args.no_pretrain:
        common.append("--no-pretrain")
    if args.limit_train:
        common += ["--limit-train", str(args.limit_train)]
    if args.limit_val:
        common += ["--limit-val", str(args.limit_val)]
    for ms, skip in itertools.product((False, True), repeat=2):
        name = "{}_ms{}_skip{}".format(args.prefix, int(ms), int(skip))
        structure = ["--multiscale", "aspp" if ms else "none"]
        if skip:
            structure.append("--unet-skips")
        backbone = name + "_backbone"
        initial = str(Path(args.out_dir) / backbone / "best.pth.tar")
        commands.append((backbone, common + structure + [
            "--run-name", backbone, "--epochs", str(args.backbone_epochs)]))
        refined = name + "_refined"
        commands.append((refined, common + structure + [
            "--run-name", refined, "--init-checkpoint", initial,
            "--refiner", "cspn-lite", "--prop-kernel", "3", "--prop-steps", "3",
            "--refiner-only-epochs", str(args.refiner_epochs),
            "--epochs", str(args.refiner_epochs + args.joint_epochs),
            "--refiner-lr", "0.001", "--joint-lr", "0.0001"]))
        control = name + "_continued"
        commands.append((control, common + structure + [
            "--run-name", control, "--init-checkpoint", initial,
            "--epochs", str(args.joint_epochs), "--joint-lr", "0.0001"]))
    return commands


def main(argv=None):
    cfg = Config()
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", default=cfg.data_root)
    parser.add_argument("--out-dir", default=str(Path(cfg.out_dir) / "ablations"))
    parser.add_argument("--prefix", default="ablation")
    parser.add_argument("--arch", choices=["resnet18", "resnet50"], default="resnet50")
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--workers", type=int, default=10)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--val-sampling-seed", type=int, default=0)
    parser.add_argument("--backbone-epochs", type=int, default=15)
    parser.add_argument("--refiner-epochs", type=int, default=3)
    parser.add_argument("--joint-epochs", type=int, default=5)
    parser.add_argument("--no-pretrain", action="store_true")
    parser.add_argument("--limit-train", type=int, default=0)
    parser.add_argument("--limit-val", type=int, default=0)
    parser.add_argument("--execute", action="store_true", help="Actually train; otherwise only print commands")
    args = parser.parse_args(argv)
    if min(args.backbone_epochs, args.joint_epochs) < 1 or args.refiner_epochs < 0:
        parser.error("Invalid stage epoch counts")
    if not args.prefix or Path(args.prefix).name != args.prefix or args.prefix in (".", ".."):
        parser.error("--prefix must be a single directory name")
    commands = experiment_commands(args)
    if args.execute:
        for name, _ in commands:
            path = Path(args.out_dir) / name
            if path.exists():
                parser.error("Experiment already exists: {}. Choose a new --prefix.".format(path))
    for name, command in commands:
        print("# " + name, flush=True)
        print(subprocess.list2cmdline(command) if sys.platform == "win32" else shlex.join(command), flush=True)
        if args.execute:
            subprocess.run(command, check=True)
            if not (Path(args.out_dir) / name / "best.pth.tar").is_file():
                raise RuntimeError("No valid best checkpoint produced for " + name)


if __name__ == "__main__":
    main()
