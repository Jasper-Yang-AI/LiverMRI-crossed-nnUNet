from __future__ import annotations

import math
import re
from pathlib import Path
from typing import Iterable, Tuple

import nibabel as nib
import numpy as np
from nibabel.processing import resample_from_to
from scipy import ndimage


def sanitize_subset_name(value: str) -> str:
    return re.sub(r"[^A-Za-z0-9._-]+", "_", str(value)).strip("_") or "unknown"


def load_nifti(path: str | Path) -> nib.Nifti1Image:
    return nib.load(str(path))


def load_float_image(path: str | Path) -> Tuple[np.ndarray, nib.Nifti1Image]:
    image = load_nifti(path)
    array = np.asarray(image.get_fdata(), dtype=np.float32)
    return array, image


def load_label_like(path: str | Path, ref_image: nib.Nifti1Image) -> np.ndarray:
    label_image = load_nifti(path)
    if label_image.shape != ref_image.shape or not np.allclose(label_image.affine, ref_image.affine, atol=1e-3):
        label_image = resample_from_to(label_image, ref_image, order=0)
    return (np.asarray(label_image.get_fdata()) > 0.5).astype(np.uint8)


def load_mask_like(path: str | Path, ref_image: nib.Nifti1Image) -> np.ndarray:
    mask_image = load_nifti(path)
    if mask_image.shape != ref_image.shape or not np.allclose(mask_image.affine, ref_image.affine, atol=1e-3):
        mask_image = resample_from_to(mask_image, ref_image, order=0)
    return (np.asarray(mask_image.get_fdata()) > 0.5).astype(np.uint8)


def build_nifti(
    data: np.ndarray,
    ref_image: nib.Nifti1Image,
    *,
    affine: np.ndarray | None = None,
    dtype: np.dtype | type | str | None = None,
) -> nib.Nifti1Image:
    header = ref_image.header.copy()
    target_dtype = np.dtype(dtype) if dtype is not None else data.dtype
    header.set_data_dtype(target_dtype)
    target_affine = affine if affine is not None else ref_image.affine
    return nib.Nifti1Image(np.asarray(data, dtype=target_dtype), target_affine, header=header)


def save_nifti(image: nib.Nifti1Image, path: str | Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    nib.save(image, str(path))


def largest_connected_component(mask: np.ndarray) -> np.ndarray:
    labeled, num = ndimage.label(mask > 0)
    if num <= 1:
        return (mask > 0).astype(np.uint8)
    component_sizes = ndimage.sum(mask > 0, labeled, index=np.arange(1, num + 1))
    keep_label = int(np.argmax(component_sizes)) + 1
    return (labeled == keep_label).astype(np.uint8)


def fill_mask_holes(mask: np.ndarray) -> np.ndarray:
    return ndimage.binary_fill_holes(mask > 0).astype(np.uint8)


def ellipsoid_structure(spacing: Iterable[float], radius_mm: float) -> np.ndarray:
    spacing = np.asarray(list(spacing), dtype=float)
    if radius_mm <= 0:
        return np.ones((1, 1, 1), dtype=bool)
    radius_vox = np.maximum(1, np.ceil(radius_mm / spacing).astype(int))
    grid = np.ogrid[
        -radius_vox[0] : radius_vox[0] + 1,
        -radius_vox[1] : radius_vox[1] + 1,
        -radius_vox[2] : radius_vox[2] + 1,
    ]
    distance = (
        (grid[0] * spacing[0]) ** 2
        + (grid[1] * spacing[1]) ** 2
        + (grid[2] * spacing[2]) ** 2
    )
    return distance <= (radius_mm**2)


def dilate_mask_mm(mask: np.ndarray, spacing: Iterable[float], radius_mm: float) -> np.ndarray:
    structure = ellipsoid_structure(spacing, radius_mm)
    return ndimage.binary_dilation(mask > 0, structure=structure).astype(np.uint8)


def compute_bbox_slices(mask: np.ndarray, spacing: Iterable[float], margin_mm: float = 0.0) -> Tuple[slice, slice, slice]:
    coords = np.argwhere(mask > 0)
    if coords.size == 0:
        raise ValueError("ROI mask is empty; cannot compute crop bbox.")
    spacing = np.asarray(list(spacing), dtype=float)
    mins = coords.min(axis=0)
    maxs = coords.max(axis=0) + 1
    if margin_mm > 0:
        margin_vox = np.ceil(margin_mm / spacing).astype(int)
        mins = np.maximum(0, mins - margin_vox)
        maxs = np.minimum(mask.shape, maxs + margin_vox)
    return tuple(slice(int(mins[i]), int(maxs[i])) for i in range(3))


def affine_after_crop(affine: np.ndarray, crop_slices: Tuple[slice, slice, slice]) -> np.ndarray:
    origin_shift = np.array([crop_slices[0].start, crop_slices[1].start, crop_slices[2].start], dtype=float)
    new_affine = affine.copy()
    new_affine[:3, 3] = affine[:3, 3] + affine[:3, :3] @ origin_shift
    return new_affine


def apply_mask_to_image(image: np.ndarray, mask: np.ndarray, outside_value: float = 0.0) -> np.ndarray:
    return np.where(mask > 0, image, np.asarray(outside_value, dtype=np.float32)).astype(np.float32)


def clip_label_to_mask(label: np.ndarray, mask: np.ndarray) -> np.ndarray:
    return np.where(mask > 0, label, 0).astype(np.uint8)


def transform_case_with_roi(
    *,
    image_path: str | Path,
    label_path: str | Path,
    roi_path: str | Path,
    roi_mode: str,
    outside_value: float = 0.0,
    crop_margin_mm: float = 0.0,
    clip_label: bool = True,
) -> Tuple[nib.Nifti1Image, nib.Nifti1Image, dict]:
    image_arr, image_obj = load_float_image(image_path)
    label_arr = load_label_like(label_path, image_obj)
    roi_mask = load_mask_like(roi_path, image_obj)

    if roi_mask.sum() == 0:
        raise ValueError(f"ROI mask is empty: {roi_path}")

    spacing = np.asarray(image_obj.header.get_zooms()[:3], dtype=float)

    if roi_mode == "masked":
        out_image = apply_mask_to_image(image_arr, roi_mask, outside_value=outside_value)
        out_label = clip_label_to_mask(label_arr, roi_mask) if clip_label else label_arr.astype(np.uint8)
        image_nifti = build_nifti(out_image, image_obj, dtype=np.float32)
        label_nifti = build_nifti(out_label, image_obj, dtype=np.uint8)
        return image_nifti, label_nifti, {
            "roi_mode": roi_mode,
            "bbox_start_zyx": "",
            "bbox_stop_zyx": "",
            "roi_voxels": int(roi_mask.sum()),
        }

    if roi_mode != "cropped":
        raise ValueError(f"Unsupported roi_mode: {roi_mode}")

    crop_slices = compute_bbox_slices(roi_mask, spacing, margin_mm=float(crop_margin_mm))
    cropped_image = image_arr[crop_slices].astype(np.float32)
    cropped_label = label_arr[crop_slices].astype(np.uint8)
    if clip_label:
        cropped_roi = roi_mask[crop_slices]
        cropped_label = clip_label_to_mask(cropped_label, cropped_roi)

    cropped_affine = affine_after_crop(image_obj.affine, crop_slices)
    image_nifti = build_nifti(cropped_image, image_obj, affine=cropped_affine, dtype=np.float32)
    label_nifti = build_nifti(cropped_label, image_obj, affine=cropped_affine, dtype=np.uint8)
    bbox_start = [int(crop_slices[i].start) for i in range(3)]
    bbox_stop = [int(crop_slices[i].stop) for i in range(3)]
    return image_nifti, label_nifti, {
        "roi_mode": roi_mode,
        "bbox_start_zyx": ",".join(map(str, bbox_start)),
        "bbox_stop_zyx": ",".join(map(str, bbox_stop)),
        "roi_voxels": int(roi_mask.sum()),
    }


def ensure_binary_prediction(path: str | Path) -> Tuple[np.ndarray, nib.Nifti1Image]:
    pred_image = load_nifti(path)
    pred_array = (np.asarray(pred_image.get_fdata()) > 0.5).astype(np.uint8)
    return pred_array, pred_image


def mm_tag(value: float) -> str:
    if math.isclose(value, round(value)):
        return f"{int(round(value))}mm"
    return f"{str(value).replace('.', 'p')}mm"

