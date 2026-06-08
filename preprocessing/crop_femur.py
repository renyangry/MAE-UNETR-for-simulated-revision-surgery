"""
Crops CT volumes to femur-only regions using label=1 from nnUNet segmentation masks.

For each CT/segmentation pair the bounding box of the femur label is computed
and a padded crop is extracted from the original CT volume.
"""

import argparse
from pathlib import Path

import nibabel as nib
import numpy as np


def get_femur_bbox(seg_data: np.ndarray, label: int = 1, pad: int = 10):
    """Return (min, max) index bounds for the femur label with optional padding."""
    coords = np.argwhere(seg_data == label)
    if coords.size == 0:
        raise ValueError("No voxels found for femur label in segmentation mask.")
    mins = np.maximum(coords.min(axis=0) - pad, 0)
    maxs = np.minimum(coords.max(axis=0) + pad + 1, np.array(seg_data.shape))
    return mins, maxs


def crop_volume(ct_data: np.ndarray, mins, maxs) -> np.ndarray:
    return ct_data[mins[0]:maxs[0], mins[1]:maxs[1], mins[2]:maxs[2]]


def parse_args():
    parser = argparse.ArgumentParser(
        description="Crop CT volumes to femur-only bounding box."
    )
    parser.add_argument("--ct_dir", type=str, required=True,
                        help="Directory of input CT NIfTI files")
    parser.add_argument("--seg_dir", type=str, required=True,
                        help="Directory of nnUNet segmentation masks")
    parser.add_argument("--out_dir", type=str, required=True,
                        help="Output directory for cropped femur volumes")
    parser.add_argument("--pad", type=int, default=10,
                        help="Voxel padding around the femur bounding box")
    return parser.parse_args()


def main():
    args = parse_args()
    ct_dir = Path(args.ct_dir)
    seg_dir = Path(args.seg_dir)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    ct_files = sorted(ct_dir.glob("*.nii.gz")) + sorted(ct_dir.glob("*.nii"))
    print(f"Found {len(ct_files)} CT volumes to crop.")

    for ct_fp in ct_files:
        seg_fp = seg_dir / ct_fp.name
        if not seg_fp.exists():
            print(f"  [SKIP] No segmentation found for {ct_fp.name}")
            continue

        ct_img = nib.load(ct_fp)
        seg_img = nib.load(seg_fp)

        ct_data = ct_img.get_fdata(dtype=np.float32)
        seg_data = np.asarray(seg_img.dataobj, dtype=np.uint8)

        try:
            mins, maxs = get_femur_bbox(seg_data, label=1, pad=args.pad)
        except ValueError as e:
            print(f"  [WARN] {ct_fp.name}: {e}")
            continue

        cropped = crop_volume(ct_data, mins, maxs)
        out_img = nib.Nifti1Image(cropped, ct_img.affine)
        nib.save(out_img, str(out_dir / ct_fp.name))
        print(f"  {ct_fp.name}: cropped to {cropped.shape}")

    print("Cropping complete.")


if __name__ == "__main__":
    main()
