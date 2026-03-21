from __future__ import annotations

from pathlib import Path

import nibabel as nib
import numpy as np
from nibabel.processing import resample_from_to
from scipy import ndimage


def load_resampled_array(path: str | Path, ref_image: nib.Nifti1Image, order: int) -> np.ndarray:
    image = nib.load(str(path))
    if image.shape != ref_image.shape or not np.allclose(image.affine, ref_image.affine, atol=1e-3):
        image = resample_from_to(image, ref_image, order=order)
    return np.asarray(image.get_fdata(), dtype=np.float32)


def gaussian_smooth_confidence(array: np.ndarray, sigma: float = 1.0) -> np.ndarray:
    if sigma <= 0:
        return np.clip(array, 0.0, 1.0)
    return np.clip(ndimage.gaussian_filter(array.astype(np.float32), sigma=sigma), 0.0, 1.0)


def build_overlap_confidence(
    anchor_liver_mask: np.ndarray,
    moving_liver_mask: np.ndarray | None,
    *,
    smooth_sigma: float = 1.0,
) -> np.ndarray:
    if moving_liver_mask is None:
        return np.zeros_like(anchor_liver_mask, dtype=np.float32)
    anchor = (anchor_liver_mask > 0).astype(np.float32)
    moving = (moving_liver_mask > 0).astype(np.float32)
    overlap = anchor * moving
    return gaussian_smooth_confidence(overlap, sigma=smooth_sigma)


def build_anchor_confidence(anchor_liver_mask: np.ndarray) -> np.ndarray:
    return (anchor_liver_mask > 0).astype(np.float32)
