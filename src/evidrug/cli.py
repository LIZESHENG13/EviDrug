"""One command interface for resource preparation, training and evaluation."""

import argparse

from .config import ARMS, TRAINING


def main():
    parser = argparse.ArgumentParser(prog="evidrug", description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    resources = sub.add_parser("resources", help="Download checksum-verified public mappings")
    resources.add_argument("--output", default="resources")
    prepare = sub.add_parser("prepare", help="Prepare an authorized source cohort")
    prepare.add_argument("--dataset", choices=("mimic_iv_full", "nwicu", "sicdb"), required=True)
    prepare.add_argument("--input", required=True, help="MIMIC/NWICU ZIP or SICdb 1.0.8 directory")
    prepare.add_argument("--output", required=True)
    prepare.add_argument("--resources", default="resources")
    for action in ("train", "evaluate", "benchmark"):
        command = sub.add_parser(action)
        data = command.add_mutually_exclusive_group(required=True)
        data.add_argument("--data", help="Trusted, locally prepared patients.pkl")
        data.add_argument("--encoded", help="Trusted local encoded cohort tuple")
        command.add_argument("--resources", default="resources")
        command.add_argument("--device", default="cpu")
        command.add_argument("--threads", type=int, default=1)
        if action == "train":
            command.add_argument(
                "--dataset", choices=("mimic_iv_full", "nwicu", "sicdb"), required=True
            )
            command.add_argument("--output", required=True)
            command.add_argument("--seed", type=int, default=0)
            command.add_argument("--arm", choices=ARMS, default="full")
            command.add_argument(
                "--continuation",
                choices=("dynamic",) + tuple(TRAINING["fixed_coefficient_controls"]),
                default="dynamic",
            )
            command.add_argument(
                "--initial-refinement", help="Paired main-run refinement/best.pt for controls"
            )
            command.add_argument(
                "--epochs", type=int, default=20, help="Epochs in each training stage"
            )
            command.add_argument("--batch-size", type=int, default=64)
        else:
            command.add_argument("--run", required=True)
            if action == "evaluate":
                command.add_argument("--freeze", required=True)
            else:
                command.add_argument("--patients", type=int, default=128)
                command.add_argument("--batch-size", type=int, default=16)
                command.add_argument("--passes", type=int, default=3)
                command.add_argument("--output", required=True)
    freeze = sub.add_parser("freeze", help="Freeze a completed run group before test evaluation")
    freeze.add_argument("--runs", required=True)
    args = parser.parse_args()
    if hasattr(args, "threads"):
        import torch

        torch.set_num_threads(args.threads)
        torch.set_float32_matmul_precision("highest")
    if args.command == "resources":
        from .data.resources import download_resources

        download_resources(args.output)
    elif args.command == "prepare":
        from .data.resources import prepare_dataset

        prepare_dataset(args.dataset, args.input, args.output, args.resources)
    elif args.command == "benchmark":
        from .benchmark import benchmark

        benchmark(args)
    else:
        from . import training

        getattr(training, args.command)(args)
