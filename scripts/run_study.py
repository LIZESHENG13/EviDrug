"""Launch the declared EviDrug run grid sequentially; test evaluation stays separate."""

import argparse
import shlex
import subprocess
import sys
from pathlib import Path

from evidrug.config import ARMS, TRAINING


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument(
        "--datasets",
        nargs="+",
        default=["mimic_iv_full", "nwicu", "sicdb"],
        choices=("mimic_iv_full", "nwicu", "sicdb"),
    )
    parser.add_argument("--seeds", nargs="+", type=int, default=[0, 1, 2])
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--ablations", action="store_true")
    parser.add_argument("--controls", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    arms = ARMS if args.ablations else ("full",)
    for dataset in args.datasets:
        for seed in args.seeds:
            for setting in list(arms) + (
                list(TRAINING["fixed_coefficient_controls"]) if args.controls else []
            ):
                output = args.output_root / dataset / setting / f"seed{seed}"
                if (output / "selected.json").exists():
                    print(f"Completed: {dataset}/{setting}/seed{seed}")
                    continue
                command = [
                    sys.executable,
                    "-m",
                    "evidrug",
                    "train",
                    "--dataset",
                    dataset,
                    "--data",
                    str(args.data_root / dataset / "patients.pkl"),
                    "--output",
                    str(output),
                    "--seed",
                    str(seed),
                    "--device",
                    args.device,
                ]
                if setting in ARMS:
                    command += ["--arm", setting]
                else:
                    initial = (
                        args.output_root / dataset / "full" / f"seed{seed}" / "refinement/best.pt"
                    )
                    command += ["--continuation", setting, "--initial-refinement", str(initial)]
                print(shlex.join(command), flush=True)
                if not args.dry_run:
                    subprocess.run(command, check=True)


if __name__ == "__main__":
    main()
