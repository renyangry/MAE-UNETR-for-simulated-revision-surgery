"""
nnUNet inference for two-class femur CT segmentation.

Produces NIfTI segmentation masks (label 1 = femur) used by the downstream
preprocessing step to crop femur-only regions.
"""

import argparse
import subprocess
import sys


def parse_args():
    parser = argparse.ArgumentParser(
        description="Run nnUNet inference for femur CT segmentation."
    )
    parser.add_argument("--input_dir", type=str, required=True,
                        help="Directory containing input NIfTI CT files")
    parser.add_argument("--output_dir", type=str, required=True,
                        help="Directory to write predicted segmentation masks")
    parser.add_argument("--model_dir", type=str, required=True,
                        help="Path to trained nnUNet model directory")
    parser.add_argument("--dataset_id", type=int, required=True,
                        help="nnUNet dataset ID")
    parser.add_argument("--configuration", type=str, default="3d_fullres",
                        help="nnUNet configuration used during training")
    parser.add_argument("--folds", type=str, default="0",
                        help="Fold(s) to use for ensemble inference (e.g. '0 1 2')")
    parser.add_argument("--device", type=str, default="cuda",
                        choices=["cuda", "cpu", "mps"])
    return parser.parse_args()


def main():
    args = parse_args()

    cmd = [
        "nnUNetv2_predict",
        "-i", args.input_dir,
        "-o", args.output_dir,
        "-d", str(args.dataset_id),
        "-c", args.configuration,
        "-f", *args.folds.split(),
        "--device", args.device,
    ]

    print(f"Running nnUNet inference: {' '.join(cmd)}")
    result = subprocess.run(cmd, check=True)
    sys.exit(result.returncode)


if __name__ == "__main__":
    main()
