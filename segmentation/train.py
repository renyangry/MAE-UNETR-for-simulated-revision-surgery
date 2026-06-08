"""
nnUNet training entry point for two-class femur CT segmentation.

Labels:
    0 — background
    1 — femur

This script wraps the standard nnUNet v2 training interface.
See nnUNet documentation for dataset preparation and fold configuration.
"""

import argparse
import subprocess
import sys


def parse_args():
    parser = argparse.ArgumentParser(
        description="Train nnUNet for two-class femur CT segmentation."
    )
    parser.add_argument("--dataset_id", type=int, required=True,
                        help="nnUNet dataset ID (e.g. 001)")
    parser.add_argument("--configuration", type=str, default="3d_fullres",
                        choices=["3d_fullres", "3d_lowres", "2d"],
                        help="nnUNet configuration to train")
    parser.add_argument("--fold", type=int, default=0,
                        help="Cross-validation fold (0–4, or 'all')")
    parser.add_argument("--trainer", type=str, default="nnUNetTrainer",
                        help="nnUNet trainer class name")
    return parser.parse_args()


def main():
    args = parse_args()

    cmd = [
        "nnUNetv2_train",
        str(args.dataset_id),
        args.configuration,
        str(args.fold),
        "--trainer", args.trainer,
    ]

    print(f"Launching nnUNet training: {' '.join(cmd)}")
    result = subprocess.run(cmd, check=True)
    sys.exit(result.returncode)


if __name__ == "__main__":
    main()
