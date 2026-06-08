"""
Batch evaluation script.

Computes DSC, HD95, and volumetric error for all prediction/ground-truth
NIfTI pairs in the specified directories and writes a CSV summary.
"""

import argparse
import csv
from pathlib import Path

import nibabel as nib
import numpy as np

from evaluation.metrics import dice_coefficient, hausdorff_distance_95, volume_similarity


def parse_args():
    parser = argparse.ArgumentParser(
        description="Evaluate MAE-UNETR predictions against ground-truth masks."
    )
    parser.add_argument("--pred_dir", type=str, required=True,
                        help="Directory of predicted segmentation NIfTI files")
    parser.add_argument("--gt_dir", type=str, required=True,
                        help="Directory of ground-truth segmentation NIfTI files")
    parser.add_argument("--output_csv", type=str, default="results/evaluation.csv",
                        help="Path to write the per-case CSV summary")
    parser.add_argument("--label", type=int, default=1,
                        help="Foreground label to evaluate (default: 1 = femur)")
    return parser.parse_args()


def main():
    args = parse_args()
    pred_dir = Path(args.pred_dir)
    gt_dir = Path(args.gt_dir)
    out_csv = Path(args.output_csv)
    out_csv.parent.mkdir(parents=True, exist_ok=True)

    pred_files = sorted(pred_dir.glob("*.nii.gz")) + sorted(pred_dir.glob("*.nii"))
    print(f"Evaluating {len(pred_files)} cases (label={args.label}).")

    rows = []
    for fp in pred_files:
        gt_fp = gt_dir / fp.name
        if not gt_fp.exists():
            print(f"  [SKIP] No ground truth for {fp.name}")
            continue

        pred_img = nib.load(fp)
        gt_img = nib.load(gt_fp)

        pred = (np.asarray(pred_img.dataobj) == args.label).astype(np.uint8)
        gt = (np.asarray(gt_img.dataobj) == args.label).astype(np.uint8)

        zooms = pred_img.header.get_zooms()[:3]
        spacing = tuple(float(z) for z in zooms)
        voxel_vol = float(np.prod(zooms))

        dsc = dice_coefficient(pred, gt)
        hd95 = hausdorff_distance_95(pred, gt, spacing)
        vol = volume_similarity(pred, gt, voxel_vol)

        row = {
            "case": fp.name,
            "DSC": round(dsc, 4),
            "HD95_mm": round(hd95, 2),
            "pred_volume_mm3": round(vol["pred_volume_mm3"], 1),
            "gt_volume_mm3": round(vol["target_volume_mm3"], 1),
            "volume_error_mm3": round(vol["volume_error_mm3"], 1),
        }
        rows.append(row)
        print(f"  {fp.name}: DSC={row['DSC']:.4f}, HD95={row['HD95_mm']:.1f} mm")

    if rows:
        with open(out_csv, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=rows[0].keys())
            writer.writeheader()
            writer.writerows(rows)

        dscs = [r["DSC"] for r in rows]
        hd95s = [r["HD95_mm"] for r in rows]
        print(f"\nSummary ({len(rows)} cases):")
        print(f"  Mean DSC : {np.mean(dscs):.4f} ± {np.std(dscs):.4f}")
        print(f"  Mean HD95: {np.mean(hd95s):.2f} ± {np.std(hd95s):.2f} mm")
        print(f"\nResults written to {out_csv}")


if __name__ == "__main__":
    main()
