"""Task 2 experiment matrix. Default prints commands only, --run launches one."""

import argparse
import shlex
import subprocess
import sys


def experiments():
    rows = {}
    for decoder in ("deconv3", "upconv", "upproj"):
        for learned in (False, True):
            for mode in ("zero", "sentinel", "mask", "sparse"):
                name = "t2_{}_{}_{}".format(
                    decoder, "learned" if learned else "bilinear", mode)
                flags = ["--decoder", decoder, "--invalid-mode", mode]
                if learned:
                    flags.append("--learned-upsample")
                rows[name] = flags
    return rows


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--experiment", choices=list(experiments()))
    parser.add_argument("--run", action="store_true",
                        help="explicitly launch the selected experiment")
    args, extra = parser.parse_known_args()
    if args.run and not args.experiment:
        parser.error("--run requires --experiment")
    rows = experiments()
    names = [args.experiment] if args.experiment else rows
    for name in names:
        command = [sys.executable, "-m", "scripts.train",
                   "--lighting-std", "0", "--criterion", "mse",
                   "--num-samples", "500", "--run-name", name]
        command += rows[name] + extra
        print(shlex.join(command), flush=True)
        if args.run:
            subprocess.run(command, check=True)


if __name__ == "__main__":
    main()
