"""
Simulates metallic implant signal loss by applying a synthetic implant-shaped
mask to femoral CT volumes.

The implant mask is derived from a segmentation of an implant-alike structure
and is used to zero out (nullify) CT signal in the corresponding region,
replicating the effect of metallic hardware on image quality.
"""

import argparse
from pathlib import Path

import nibabel as nib
import numpy as np


def apply_implant_mask(
    ct_data: np.ndarray,
    implant_mask: np.ndarray,
    fill_value: float = 0.0,
) -> np.ndarray:
    """
    Zero out CT signal within the implant mask region.

    Parameters
    ----------
    ct_data : np.ndarray
        CT intensity volume.
    implant_mask : np.ndarray
        Binary mask (1 = implant region, 0 = background).
    fill_value : float
        Replacement value for masked voxels (default: 0.0).

    Returns
    -------
    np.ndarray
        CT volume with implant region signal removed.
    """
    if ct_data.shape != implant_mask.shape:
        raise ValueError(
            f"CT shape {ct_data.shape} and mask shape {implant_mask.shape} must match."
        )
    masked = ct_data.copy()
    masked[implant_mask > 0] = fill_value
    return masked


def parse_args():
    parser = argparse.ArgumentParser(
        description="Apply synthetic implant mask to femoral CT volumes."
    )
    parser.add_argument("--input_dir", type=str, required=True,
                        help="Directory of femur-cropped CT NIfTI volumes")
    parser.add_argument("--mask_dir", type=str, required=True,
                        help="Directory of implant segmentation masks (NIfTI, label=1)")
    parser.add_argument("--out_dir", type=str, required=True,
                        help="Output directory for masked CT volumes")
    parser.add_argument("--fill_value", type=float, default=0.0,
                        help="Value to fill masked voxels with")
    return parser.parse_args()


def main():
    args = parse_args()
    in_dir = Path(args.input_dir)
    mask_dir = Path(args.mask_dir)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    ct_files = sorted(in_dir.glob("*.nii.gz")) + sorted(in_dir.glob("*.nii"))
    print(f"Applying implant mask to {len(ct_files)} volumes.")

    for fp in ct_files:
        mask_fp = mask_dir / fp.name
        if not mask_fp.exists():
            print(f"  [SKIP] No implant mask for {fp.name}")
            continue

        ct_img = nib.load(fp)
        mask_img = nib.load(mask_fp)
        ct_data = ct_img.get_fdata(dtype=np.float32)
        mask_data = np.asarray(mask_img.dataobj, dtype=np.uint8)

        try:
            masked = apply_implant_mask(ct_data, mask_data, args.fill_value)
        except ValueError as e:
            print(f"  [WARN] {fp.name}: {e}")
            continue

        nib.save(nib.Nifti1Image(masked, ct_img.affine), str(out_dir / fp.name))
        print(f"  {fp.name}: implant mask applied.")

    print("Implant masking complete.")


if __name__ == "__main__":
    main()
