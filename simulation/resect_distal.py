"""
Simulates distal femoral resection by zeroing out the distal portion of a
femoral CT crop.

The resection plane is placed at a fraction of the total femoral length
(measured along the superior-inferior axis), up to a maximum of 10% of the
total length as reported in the manuscript.
"""

import argparse
from pathlib import Path

import nibabel as nib
import numpy as np


def apply_distal_resection(
    ct_data: np.ndarray,
    seg_data: np.ndarray,
    fraction: float = 0.10,
    femur_label: int = 1,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Zero out the distal `fraction` of the femoral volume.

    Parameters
    ----------
    ct_data : np.ndarray
        CT intensity volume (D, H, W).
    seg_data : np.ndarray
        Femur segmentation mask (D, H, W), label=femur_label.
    fraction : float
        Fraction of femoral length to resect from the distal end (0 < fraction <= 0.10).
    femur_label : int
        Voxel label for the femur in seg_data.

    Returns
    -------
    ct_resected, seg_resected : np.ndarray
        CT and segmentation volumes with the distal region zeroed out.
    """
    if not (0 < fraction <= 0.10):
        raise ValueError(f"fraction must be in (0, 0.10], got {fraction}")

    femur_coords = np.argwhere(seg_data == femur_label)
    if femur_coords.size == 0:
        raise ValueError("No femur voxels found in segmentation.")

    # Superior-inferior axis assumed to be axis=0 (first spatial dimension).
    si_min = femur_coords[:, 0].min()
    si_max = femur_coords[:, 0].max()
    femoral_length = si_max - si_min

    resection_boundary = int(si_max - np.floor(fraction * femoral_length))

    ct_resected = ct_data.copy()
    seg_resected = seg_data.copy()
    ct_resected[resection_boundary:, :, :] = 0
    seg_resected[resection_boundary:, :, :] = 0

    return ct_resected, seg_resected


def parse_args():
    parser = argparse.ArgumentParser(
        description="Simulate distal femoral resection on femur CT crops."
    )
    parser.add_argument("--input_dir", type=str, required=True,
                        help="Directory of femur-cropped CT NIfTI volumes")
    parser.add_argument("--seg_dir", type=str, required=True,
                        help="Directory of corresponding segmentation masks")
    parser.add_argument("--out_dir", type=str, required=True,
                        help="Output directory for resected volumes")
    parser.add_argument("--fraction", type=float, default=0.10,
                        help="Fraction of femoral length to resect (max 0.10)")
    return parser.parse_args()


def main():
    args = parse_args()
    in_dir = Path(args.input_dir)
    seg_dir = Path(args.seg_dir)
    out_dir = Path(args.out_dir)
    out_ct_dir = out_dir / "ct"
    out_seg_dir = out_dir / "seg"
    out_ct_dir.mkdir(parents=True, exist_ok=True)
    out_seg_dir.mkdir(parents=True, exist_ok=True)

    ct_files = sorted(in_dir.glob("*.nii.gz")) + sorted(in_dir.glob("*.nii"))
    print(f"Applying {args.fraction*100:.0f}% distal resection to {len(ct_files)} volumes.")

    for fp in ct_files:
        seg_fp = seg_dir / fp.name
        if not seg_fp.exists():
            print(f"  [SKIP] No segmentation for {fp.name}")
            continue

        ct_img = nib.load(fp)
        seg_img = nib.load(seg_fp)
        ct_data = ct_img.get_fdata(dtype=np.float32)
        seg_data = np.asarray(seg_img.dataobj, dtype=np.uint8)

        try:
            ct_r, seg_r = apply_distal_resection(ct_data, seg_data, args.fraction)
        except ValueError as e:
            print(f"  [WARN] {fp.name}: {e}")
            continue

        nib.save(nib.Nifti1Image(ct_r, ct_img.affine), str(out_ct_dir / fp.name))
        nib.save(nib.Nifti1Image(seg_r, seg_img.affine), str(out_seg_dir / fp.name))
        print(f"  {fp.name}: resection applied.")

    print("Distal resection simulation complete.")


if __name__ == "__main__":
    main()
