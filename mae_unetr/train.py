"""
Training script for the MAE-UNETR reconstruction network.

Training proceeds in two stages:
  1. Self-supervised MAE pretraining on intact femoral CT crops.
  2. Supervised fine-tuning for full femoral segmentation reconstruction.

Full training configuration is specified via configs/train_config.yaml.
"""

import argparse
from pathlib import Path

import torch
import yaml


def parse_args():
    parser = argparse.ArgumentParser(
        description="Train the MAE-UNETR reconstruction network."
    )
    parser.add_argument("--config", type=str, default="configs/train_config.yaml",
                        help="Path to training configuration YAML")
    parser.add_argument("--data_dir", type=str, required=True,
                        help="Directory of preprocessed femoral CT crops")
    parser.add_argument("--output_dir", type=str, default="checkpoints/",
                        help="Directory to save model checkpoints")
    parser.add_argument("--resume", type=str, default=None,
                        help="Path to checkpoint to resume training from")
    parser.add_argument("--device", type=str, default="cuda")
    return parser.parse_args()


def load_config(path: str) -> dict:
    with open(path) as f:
        return yaml.safe_load(f)


def main():
    args = parse_args()
    cfg = load_config(args.config)

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"Training MAE-UNETR | device={device} | output={output_dir}")
    print("Configuration:")
    for k, v in cfg.items():
        print(f"  {k}: {v}")

    # Training loop — replace with actual implementation.
    raise NotImplementedError(
        "Training implementation is available upon request to the corresponding "
        "author. See the manuscript for full training details."
    )


if __name__ == "__main__":
    main()
