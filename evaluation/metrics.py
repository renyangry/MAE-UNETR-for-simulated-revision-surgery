"""
Volumetric segmentation evaluation metrics.

Implements Dice Similarity Coefficient (DSC) and 95th-percentile
Hausdorff Distance (HD95) for binary 3D segmentation masks.
"""

import numpy as np
from scipy.ndimage import binary_erosion, distance_transform_edt


def dice_coefficient(pred: np.ndarray, target: np.ndarray) -> float:
    """Compute Dice Similarity Coefficient for binary masks."""
    pred = pred.astype(bool)
    target = target.astype(bool)
    intersection = np.logical_and(pred, target).sum()
    denom = pred.sum() + target.sum()
    if denom == 0:
        return 1.0 if intersection == 0 else 0.0
    return 2.0 * intersection / denom


def surface_distances(pred: np.ndarray, target: np.ndarray,
                      spacing: tuple = (1.0, 1.0, 1.0)) -> np.ndarray:
    """Compute symmetric surface distances between two binary masks."""
    def _surface(mask):
        eroded = binary_erosion(mask)
        return np.logical_xor(mask, eroded)

    pred_surface = _surface(pred.astype(bool))
    target_surface = _surface(target.astype(bool))

    dt_pred = distance_transform_edt(~pred_surface, sampling=spacing)
    dt_target = distance_transform_edt(~target_surface, sampling=spacing)

    d_pred_to_target = dt_target[pred_surface]
    d_target_to_pred = dt_pred[target_surface]

    return np.concatenate([d_pred_to_target, d_target_to_pred])


def hausdorff_distance_95(pred: np.ndarray, target: np.ndarray,
                           spacing: tuple = (1.0, 1.0, 1.0)) -> float:
    """Compute the 95th-percentile Hausdorff Distance (HD95) in mm."""
    distances = surface_distances(pred, target, spacing)
    if distances.size == 0:
        return 0.0
    return float(np.percentile(distances, 95))


def volume_similarity(pred: np.ndarray, target: np.ndarray,
                      voxel_volume_mm3: float = 1.0) -> dict:
    """
    Return a dict with predicted volume, target volume, and their absolute
    difference in mm³.
    """
    pred_vol = pred.astype(bool).sum() * voxel_volume_mm3
    target_vol = target.astype(bool).sum() * voxel_volume_mm3
    return {
        "pred_volume_mm3": pred_vol,
        "target_volume_mm3": target_vol,
        "volume_error_mm3": abs(pred_vol - target_vol),
    }
