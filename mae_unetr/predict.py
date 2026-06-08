"""
Inference script for the MAE-UNETR reconstruction network.

Accepts femur-cropped CT volumes (output of preprocessing/crop_femur.py) and
produces full femoral 3D segmentation masks.
"""

import argparse
from pathlib import Path

import nibabel as nib
import numpy as np
import torch

from mae_unetr.models.mae_unetr import MAEUNETR


def parse_args():
    parser = argparse.ArgumentParser(
        description="Run MAE-UNETR inference for femoral reconstruction."
    )
    parser.add_argument("--input_dir", type=str, required=True,
                        help="Directory of femur-cropped CT NIfTI volumes")
    parser.add_argument("--output_dir", type=str, required=True,
                        help="Directory to write predicted segmentation masks")
    parser.add_argument("--weights", type=str, required=True,
                        help="Path to trained MAE-UNETR weights (.pth)")
    parser.add_argument("--config", type=str, default="configs/predict_config.yaml")
    parser.add_argument("--device", type=str, default="cuda")
    return parser.parse_args()


def preprocess(ct_data: np.ndarray, target_shape=(96, 96, 96)) -> torch.Tensor:
    """Normalise and resize a CT crop for model input."""
    from scipy.ndimage import zoom
    scale = [t / s for t, s in zip(target_shape, ct_data.shape)]
    resized = zoom(ct_data, scale, order=1)
    # Clip HU range typical for bone
    resized = np.clip(resized, -200, 1500)
    resized = (resized - resized.mean()) / (resized.std() + 1e-8)
    return torch.from_numpy(resized).float().unsqueeze(0).unsqueeze(0)


def main():
    args = parse_args()
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")

    in_dir = Path(args.input_dir)
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    model = MAEUNETR.from_pretrained(args.weights)
    model.to(device).eval()
    print(f"Model loaded from {args.weights}")

    nifti_files = sorted(in_dir.glob("*.nii.gz")) + sorted(in_dir.glob("*.nii"))
    print(f"Found {len(nifti_files)} volumes for inference.")

    with torch.no_grad():
        for fp in nifti_files:
            img = nib.load(fp)
            ct_data = img.get_fdata(dtype=np.float32)
            x = preprocess(ct_data).to(device)

            logits = model(x)
            pred = torch.argmax(logits, dim=1).squeeze(0).cpu().numpy().astype(np.uint8)

            out_img = nib.Nifti1Image(pred, img.affine)
            nib.save(out_img, str(out_dir / fp.name))
            print(f"  {fp.name} → {out_dir / fp.name}")

    print("Inference complete.")


if __name__ == "__main__":
    main()
