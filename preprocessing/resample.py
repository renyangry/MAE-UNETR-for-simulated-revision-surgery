"""
Resamples CT volumes to a uniform isotropic voxel spacing before segmentation.
"""

import argparse
from pathlib import Path

import nibabel as nib
import numpy as np
from scipy.ndimage import zoom


def resample_volume(img: nib.Nifti1Image, target_spacing: float) -> nib.Nifti1Image:
    """Resample a NIfTI volume to isotropic voxel spacing."""
    zooms = np.array(img.header.get_zooms()[:3])
    scale = zooms / target_spacing
    data = zoom(img.get_fdata(), scale, order=1)
    new_affine = img.affine.copy()
    for i in range(3):
        new_affine[i, i] = np.sign(new_affine[i, i]) * target_spacing
    return nib.Nifti1Image(data.astype(np.float32), new_affine)


def parse_args():
    parser = argparse.ArgumentParser(
        description="Resample CT volumes to isotropic voxel spacing."
    )
    parser.add_argument("--input_dir", type=str, required=True)
    parser.add_argument("--output_dir", type=str, required=True)
    parser.add_argument("--spacing", type=float, default=1.0,
                        help="Target isotropic voxel spacing in mm")
    return parser.parse_args()


def main():
    args = parse_args()
    in_dir = Path(args.input_dir)
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    nifti_files = sorted(in_dir.glob("*.nii.gz")) + sorted(in_dir.glob("*.nii"))
    print(f"Found {len(nifti_files)} volumes to resample.")

    for fp in nifti_files:
        img = nib.load(fp)
        resampled = resample_volume(img, args.spacing)
        out_path = out_dir / fp.name
        nib.save(resampled, str(out_path))
        print(f"  {fp.name} → {out_path.name}")

    print("Resampling complete.")


if __name__ == "__main__":
    main()
